"""The run planner — platform slice 6.

The missing verb in the goal + data + hardware → runnable-plan loop: ``profiler`` describes the host
(:class:`EnvironmentProfile`), ``probes`` proves what actually works on it (:class:`CapabilityReport`),
and the ``supervisor`` / ``TrainingRunner`` execute a plan — but a :class:`RunPlan` had to be
hand-authored until now. :func:`build_run_plan` composes those inputs plus a small
:class:`PlannerConstraints` (the user intent the host can't decide) into ONE valid, immutable,
``plan_hash``-sealed RunPlan, resolving every ambiguous field AHEAD OF TIME against what PROVED to
work on this host — the runtime decisions ``training.trainer.resolve_run_plan`` /
``resolve_attention_implementation`` make late, moved forward.

Honesty non-negotiables baked in:
* Blackwell (GPU ``compute_capability_major >= 12``) forces ``attention_backend = math`` — asserted
  from the profile, independent of probe output (the flash probe short-circuits to KERNEL_STALL on
  sm_120 without executing, so flash is correctly absent from the proven set).
* Nothing is claimed that wasn't PROVEN: ``bf16`` only when it's in the effective precision modes;
  ``nf4`` only when bitsandbytes passed; resolve against ``effective_capabilities``, never the raw
  profile or a backend's declared surface.
* ``sequence_len`` flows from the constraints — never a hardcoded calibration value.
* ``cpu_toy`` is never a silent downgrade of a real-training intent: an unready host raises
  :class:`PlannerError` unless cpu-toy was explicitly requested.
* A *planned* fit is NOT a proven fit — this slice picks a VALID plan; whether it FITS the VRAM is a
  separate calibrator/measured-run concern and is deliberately not asserted here.

Dependency-light: stdlib + platform contracts only, no torch at module load.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import ValidationError

from corpus_studio.platform.backends import (
    backend_manifest_ref,
    builtin_backends,
    compatible_backends,
    get_backend,
    unmet_physical_requirements,
    unmet_requirements,
)
from corpus_studio.platform.contracts import (
    AttentionExecutionPolicy,
    CapabilityReport,
    DeviceMapEntry,
    EnvironmentProfile,
    ExecutionInputBinding,
    ExecutionInputs,
    ParameterAccountingReport,
    PhysicalExecutionSpec,
    PhysicalResource,
    PhysicalScopeSelector,
    ParallelismSpec,
    RankBinding,
    ResolvedExecutionConfiguration,
    RunPlan,
    StatePlacement,
    StorageProfile,
    TrainerInterfacePolicy,
    TrainingDataPolicy,
    TrainingSchedule,
)
from corpus_studio.platform.common import HashRef, PackageLock, Ref
from corpus_studio.platform.enums import (
    AdapterMethod,
    AttentionImpl,
    AttentionKernel,
    DeviceKind,
    ExecutionVerificationRequirement,
    ExportFormat,
    FailureTaxonomy,
    MemoryTier,
    OffloadStrategy,
    OperatingSystem,
    Optimizer,
    PhysicalStateKind,
    PlacementRole,
    TaskType,
)
from corpus_studio.platform.execution_config import (
    capability_report_ref_for,
    execution_configuration_hash_for,
    formatter_identity,
    huggingface_input_ref,
)
from corpus_studio.platform.host_platform import flash_sdpa_deadlocks
from corpus_studio.platform.parameter_accounting import verify_parameter_accounting_hash
from corpus_studio.platform.objectives import get_objective

# attn_implementation strings the trainer passes to from_pretrained. math / sdpa / mem_efficient /
# xformers are NOT from_pretrained values (they are SDPA backends toggled inside the trainer), so we
# leave attn_implementation unset for those and let the trainer's own proven Blackwell path fire.
_LORA_FAMILY = frozenset({"lora", "qlora", "dora"})
# The attention backends NOT guaranteed safe on Blackwell (sm_120): the fused/flash family deadlocks
# outright, and plain `sdpa` can DISPATCH to the deadlocking flash kernel (its safety would depend on
# the trainer disabling flash at runtime — a detail the plan must not assume; it also lets Unsloth,
# which declares sdpa but no math, slip past the sm_120 refusal). Only math + eager are sealable here.
_FUSED_ATTN_UNSAFE_ON_BLACKWELL = frozenset(
    {"flash_attention_2", "flash_attention_3", "mem_efficient", "xformers", "sdpa"}
)
_BLACKWELL_MAJOR = 12
_EXECUTION_CONTRACT_VERSION = "1.0.0"


class PlannerError(Exception):
    """A request the host cannot honor — the ahead-of-time twin of ``trainer.TrainerError`` (not
    ready, cpu-toy-only without opt-in, or an unsupported constraint)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PlannerConstraints:
    """The user intent the host can't decide — everything else the planner resolves from the
    environment + proven capabilities. ``sequence_len`` etc. flow into the plan verbatim (never a
    hardcoded calibration value)."""

    base_model: str
    dataset_path: str
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    model_content_sha256: str | None = None
    tokenizer_content_sha256: str | None = None
    dataset_content_sha256: str | None = None
    task_type: str = "sft"
    dataset_format: str = "instruction"
    adapter_method: str | None = None  # None → auto: qlora when quantized, else lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: tuple[str, ...] = ("all-linear",)
    sequence_len: int = 4096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "linear"
    warmup_ratio: float = 0.0
    seed: int = 42
    data_seed: int | None = None
    output_dir: str = "output"
    supervised_token_accumulation_target: int | None = None
    attention_backend: str | None = None  # explicit override; else resolved from the host
    verification_requirement: str = "require_verified"
    export_format: str = "adapter_peft"
    backend: str = "corpus_studio"  # the training framework to run on (see platform.backends)
    # Memory / spill-avoidance levers (opt-in), validated against the backend's declared surface: a
    # paged optimizer (spill optimizer state to RAM) + a fused-CE loss (drop the long-seq logits spike).
    optim: str = "adamw_torch"
    use_liger: bool = False
    max_steps: int | None = None
    num_train_epochs: float = 1.0
    # Intermediate first-party checkpoints remain disabled until a future execution contract can
    # seal compatible resume lineage. Non-null values are refused rather than written unusably.
    checkpoint_steps: int | None = None
    checkpoint_keep_last: int | None = None
    truncation_allowed: bool = False
    chat_template_sha256: str | None = None
    allow_cpu_toy: bool = False


def _require_enum(value: str, enum_cls: type[Enum], label: str) -> None:
    valid = {member.value for member in enum_cls}
    if value not in valid:
        raise PlannerError(
            f"unsupported {label} '{value}'; expected one of: {', '.join(sorted(valid))}"
        )


def _max_cc_major(profile: EnvironmentProfile) -> int | None:
    majors = [g.compute_capability_major for g in profile.gpus if g.compute_capability_major is not None]
    return max(majors) if majors else None


def _attention_policy(
    *,
    kernel: AttentionKernel,
    kernel_probe_ref: Ref,
    evidence_kind: str,
    safety_mandate: str | None = None,
    flash_attention_package: PackageLock | None = None,
) -> AttentionExecutionPolicy:
    model_api = {
        AttentionKernel.eager: "eager",
        AttentionKernel.torch_sdpa_math: "sdpa",
        AttentionKernel.torch_sdpa_flash: "sdpa",
        AttentionKernel.torch_sdpa_mem_efficient: "sdpa",
        AttentionKernel.flash_attention_2: "flash_attention_2",
        AttentionKernel.flash_attention_3: "flash_attention_3",
        AttentionKernel.xformers: "xformers",
    }[kernel]
    return AttentionExecutionPolicy.model_validate(
        {
            "model_attention_api": model_api,
            "effective_backend_required": kernel.value,
            "flash_sdp_enabled": kernel == AttentionKernel.torch_sdpa_flash,
            "mem_efficient_sdp_enabled": kernel
            == AttentionKernel.torch_sdpa_mem_efficient,
            "math_sdp_enabled": kernel not in {
                AttentionKernel.torch_sdpa_flash,
                AttentionKernel.torch_sdpa_mem_efficient,
            },
            "flash_attention_package": (
                flash_attention_package.model_dump(mode="json")
                if flash_attention_package is not None
                else None
            ),
            "kernel_probe_ref": kernel_probe_ref.model_dump(mode="json"),
            "evidence_kind": evidence_kind,
            "safety_mandate": safety_mandate,
            "verification_requirement": "require_verified",
            "fallback_policy": "refuse",
        }
    )


def _resolve_attention(
    explicit: str | None,
    cc_major: int | None,
    proven_attn: set[str],
    proven_kernels: set[str],
    *,
    os_value: OperatingSystem,
    evidence_ref: Ref,
    flash_attention_package: PackageLock | None,
) -> tuple[str, AttentionExecutionPolicy]:
    """Resolve an API request to one exact, enforceable kernel."""

    wddm_blackwell = flash_sdpa_deadlocks(os_value, cc_major)
    if explicit is not None:
        _require_enum(explicit, AttentionImpl, "attention_backend")
        if wddm_blackwell and explicit in _FUSED_ATTN_UNSAFE_ON_BLACKWELL:
            raise PlannerError(
                f"attention_backend '{explicit}' is not guaranteed safe on native Windows + Blackwell "
                f"(sm_120, cc_major>={_BLACKWELL_MAJOR}) - it can hit the deadlocking flash kernel under "
                "the Windows WDDM driver; use math/eager, or use a non-WDDM host only after its "
                "exact attention-kernel probe passes."
            )
    candidates: dict[str, list[AttentionKernel]] = {
        "math": [AttentionKernel.torch_sdpa_math],
        "eager": [AttentionKernel.eager],
        "sdpa": [
            AttentionKernel.torch_sdpa_flash,
            AttentionKernel.torch_sdpa_mem_efficient,
            AttentionKernel.torch_sdpa_math,
        ],
        "mem_efficient": [AttentionKernel.torch_sdpa_mem_efficient],
        "flash_attention_2": [AttentionKernel.flash_attention_2],
        "flash_attention_3": [AttentionKernel.flash_attention_3],
        "xformers": [AttentionKernel.xformers],
    }
    if wddm_blackwell:
        wddm_requested = explicit or AttentionImpl.math.value
        wddm_kernel = candidates[wddm_requested][0]
        if wddm_requested not in proven_attn or wddm_kernel.value not in proven_kernels:
            raise PlannerError(
                f"native Windows + Blackwell requires proven {wddm_requested} attention, but its exact "
                "kernel has no passing functional probe in this capability report"
            )
        return wddm_requested, _attention_policy(
            kernel=wddm_kernel,
            kernel_probe_ref=evidence_ref,
            evidence_kind="functional_probe",
            safety_mandate="native_windows_blackwell_math_or_eager_only",
        )

    requested = explicit
    if requested is None:
        if (
            AttentionImpl.math.value in proven_attn
            and AttentionKernel.torch_sdpa_math.value in proven_kernels
        ):
            # Until a complete flash/memory-efficient training tuple passes, prefer the full math
            # tuple over a faster kernel demonstrated only by an isolated attention probe.
            requested = AttentionImpl.math.value
        elif (
            AttentionImpl.sdpa.value in proven_attn
            and AttentionKernel.torch_sdpa_flash.value in proven_kernels
        ):
            requested = AttentionImpl.sdpa.value
        elif (
            AttentionImpl.sdpa.value in proven_attn
            and AttentionKernel.torch_sdpa_mem_efficient.value in proven_kernels
        ):
            requested = AttentionImpl.sdpa.value
        elif AttentionImpl.eager.value in proven_attn:
            requested = AttentionImpl.eager.value
        else:
            raise PlannerError(
                "no exact attention backend has functional evidence in this capability report"
            )
    if requested not in proven_attn:
        raise PlannerError(
            f"attention_backend '{requested}' was requested explicitly but is not functionally proven"
        )
    chosen_kernel = next(
        (item for item in candidates[requested] if item.value in proven_kernels),
        None,
    )
    if chosen_kernel is None:
        raise PlannerError(
            f"attention_backend '{requested}' has no exact proven runtime kernel"
        )
    kernel = chosen_kernel
    if kernel in {AttentionKernel.flash_attention_2, AttentionKernel.flash_attention_3} and (
        flash_attention_package is None or flash_attention_package.version is None
    ):
        raise PlannerError(
            "external FlashAttention requires an exact installed flash-attn package version"
        )
    summary = (
        AttentionImpl.sdpa
        if kernel
        in {
            AttentionKernel.torch_sdpa_flash,
            AttentionKernel.torch_sdpa_mem_efficient,
        }
        else AttentionImpl(requested)
    )
    return summary.value, _attention_policy(
        kernel=kernel,
        kernel_probe_ref=evidence_ref,
        evidence_kind="functional_probe",
        flash_attention_package=flash_attention_package,
    )


def compute_plan_hash(plan_body: Mapping[str, Any]) -> str:
    """The immutability seal: sha256 over the canonicalized plan body. Mirrors
    ``profiler._environment_signature`` (compact, key-sorted JSON) — the engine-wide content-identity
    convention. The caller MUST exclude ``plan_hash`` (a hash can't include itself) and ``created_at``
    (volatile) so two byte-identical plans minted at different instants seal to the same hash."""
    canonical = json.dumps(plan_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_plan_hash_payload(plan: RunPlan) -> dict[str, Any]:
    """Return the canonical seal payload. A missing ``physical_execution`` is omitted so legacy
    plans retain their historical hash payload; every new planner-produced plan includes the spec."""

    payload = plan.model_dump(mode="json", exclude={"plan_hash", "created_at"})
    if plan.physical_execution is None:
        payload.pop("physical_execution", None)
    return payload


def verify_run_plan_hash(plan: RunPlan) -> bool:
    return compute_plan_hash(run_plan_hash_payload(plan)) == plan.plan_hash


def storage_profile_hash_for(profile: StorageProfile) -> str:
    """Content identity for the exact StorageProfile snapshot consumed by a physical plan."""

    return compute_plan_hash(profile.model_dump(mode="json"))


def storage_profile_ref_for(profile: StorageProfile) -> Ref:
    digest = storage_profile_hash_for(profile)
    return Ref(id=f"storage-profile-{digest[:12]}", hash=HashRef(value=digest))


def default_physical_execution(
    profile: EnvironmentProfile,
    *,
    cpu_toy: bool,
) -> PhysicalExecutionSpec:
    """Resolve today's supported physical path: one explicit compute resource and rank, no offload.

    Whole-model placement is scheduling intent only; it does not become N_resident evidence.
    """

    if cpu_toy or not profile.gpus:
        tier = MemoryTier.pageable_ram
        device_kind = DeviceKind.cpu
        device_id = "cpu:0"
    else:
        gpu = min(profile.gpus, key=lambda item: item.index)
        tier = MemoryTier.gpu
        device_kind = gpu.kind
        device_id = f"{gpu.kind.value}:{gpu.index}"
    return PhysicalExecutionSpec(
        resources=[
            PhysicalResource(
                resource_id="compute-0",
                tier=tier,
                device_kind=device_kind,
                device_id=device_id,
            )
        ],
        placements=[
            StatePlacement(
                placement_id="parameters-authoritative",
                state=PhysicalStateKind.parameters,
                selector=PhysicalScopeSelector(whole_model=True),
                resource_id="compute-0",
                role=PlacementRole.authoritative,
            )
        ],
        parallelism=ParallelismSpec(
            world_size=1,
            ranks=[RankBinding(rank=0, resource_id="compute-0")],
        ),
    )


def is_trivial_physical_execution(spec: PhysicalExecutionSpec | None) -> bool:
    """Whether current runners/calibration can safely consume the physical spec without ignoring it."""

    if spec is None:
        return True
    return (
        spec.route_fidelity == "preserve_or_fail"
        and spec.semantic_fallback_policy_ref is None
        and spec.storage_profile_ref is None
        and len(spec.resources) == 1
        and len(spec.placements) == 1
        and spec.placements[0].state == PhysicalStateKind.parameters
        and spec.placements[0].selector.whole_model
        and spec.placements[0].role == PlacementRole.authoritative
        and spec.placements[0].resource_id == spec.resources[0].resource_id
        and not spec.offload_rules
        and spec.parallelism.world_size == 1
        and len(spec.parallelism.ranks) == 1
        and spec.parallelism.ranks[0].resource_id == spec.resources[0].resource_id
        and not spec.parallelism.groups
    )


def _offload_summary(spec: PhysicalExecutionSpec) -> OffloadStrategy:
    if not spec.offload_rules:
        return OffloadStrategy.none
    resources = {item.resource_id: item for item in spec.resources}
    targets = {resources[item.target_resource_id].tier for item in spec.offload_rules}
    states = {item.state for item in spec.offload_rules}
    if targets.issubset({MemoryTier.nvme, MemoryTier.sata, MemoryTier.remote}):
        return OffloadStrategy.disk_offload
    if states == {PhysicalStateKind.activations}:
        return OffloadStrategy.controlled_activation_offload
    if states == {PhysicalStateKind.optimizer_state}:
        return OffloadStrategy.controlled_optimizer_offload
    if states.issubset({PhysicalStateKind.parameters, PhysicalStateKind.gradients}):
        return OffloadStrategy.controlled_parameter_offload
    return OffloadStrategy.cpu_offload


def _validate_parameter_accounting(
    report: ParameterAccountingReport,
    spec: PhysicalExecutionSpec,
) -> Ref:
    try:
        report = ParameterAccountingReport.model_validate(report.model_dump(mode="json"))
    except (ValueError, TypeError, RecursionError) as exc:
        raise PlannerError(f"parameter-accounting report is structurally invalid: {exc}") from exc
    if not verify_parameter_accounting_hash(report):
        raise PlannerError("parameter-accounting report hash mismatch")
    scopes = {item.scope.scope_id for item in report.observations} | {
        item.scope.scope_id for item in report.gaps
    }
    components = {
        component
        for item in report.observations
        for component in item.scope.component_ids
    } | {
        component for item in report.gaps for component in item.scope.component_ids
    }
    experts = {
        expert for item in report.observations for expert in item.scope.expert_ids
    } | {expert for item in report.gaps for expert in item.scope.expert_ids}
    selectors = [
        *(item.selector for item in spec.placements),
        *(item.selector for item in spec.offload_rules),
    ]
    requested_scopes = {
        scope for selector in selectors for scope in selector.parameter_scope_ids
    } | {
        scope
        for group in spec.parallelism.groups
        for scope in group.parameter_scope_ids
    }
    requested_components = {
        component for selector in selectors for component in selector.component_ids
    }
    requested_experts = {expert for selector in selectors for expert in selector.expert_ids}
    for label, requested, available in (
        ("parameter scope", requested_scopes, scopes),
        ("component", requested_components, components),
        ("expert", requested_experts, experts),
    ):
        missing = sorted(requested - available)
        if missing:
            raise PlannerError(
                f"physical plan references {label} IDs absent from the sealed report: "
                + ", ".join(missing)
            )
    return Ref(id=report.report_id, hash=HashRef(value=report.report_hash))


def _validate_storage_profile(
    spec: PhysicalExecutionSpec,
    profile: StorageProfile | None,
    *,
    allow_marginal: bool,
    allow_unknown: bool,
) -> None:
    bindings = [item.storage for item in spec.resources if item.storage is not None]
    if not bindings:
        if profile is not None:
            raise PlannerError("a StorageProfile was supplied but the physical plan uses no storage")
        return
    if profile is None:
        raise PlannerError("storage-backed physical planning requires the exact StorageProfile")
    try:
        profile = StorageProfile.model_validate(profile.model_dump(mode="json"))
    except (ValueError, TypeError, RecursionError) as exc:
        raise PlannerError(f"StorageProfile is structurally invalid: {exc}") from exc
    expected_ref = storage_profile_ref_for(profile)
    if spec.storage_profile_ref != expected_ref:
        raise PlannerError("physical plan StorageProfile ref does not match the supplied profile")
    for binding in bindings:
        if binding is None:  # narrowed above; keeps mypy explicit.
            continue
        if binding.assessment not in profile.assessments:
            raise PlannerError("physical plan embeds a storage assessment absent from the profile")
        verdict = binding.assessment.suitability.value
        if verdict == "marginal" and not allow_marginal:
            raise PlannerError("marginal storage requires explicit allow_marginal_storage")
        if verdict == "unknown" and not allow_unknown:
            raise PlannerError("unknown storage requires explicit allow_unknown_storage")


def _validate_environment_resources(
    spec: PhysicalExecutionSpec,
    profile: EnvironmentProfile,
) -> None:
    known_accelerators = {
        (item.kind, f"{item.kind.value}:{item.index}") for item in profile.gpus
    }
    for resource in spec.resources:
        if resource.tier != MemoryTier.gpu:
            continue
        identity = (resource.device_kind, resource.device_id)
        if identity not in known_accelerators:
            raise PlannerError(
                f"physical resource '{resource.resource_id}' references an accelerator absent from "
                "the EnvironmentProfile"
            )


def _resolved_execution_inputs(
    constraints: PlannerConstraints,
    dataset_ref: Ref,
) -> ExecutionInputs:
    if dataset_ref.hash is None or dataset_ref.hash.value is None:
        raise PlannerError("dataset_ref must be hash-pinned before execution planning")
    dataset_digest = constraints.dataset_content_sha256 or dataset_ref.hash.value
    dataset = ExecutionInputBinding.model_validate(
        {
            "kind": "dataset",
            "ref": dataset_ref.model_dump(mode="json"),
            "source": "local_file",
            "location": constraints.dataset_path,
            "content_sha256": dataset_digest,
        }
    )

    if constraints.model_content_sha256 is not None:
        model_ref = Ref(
            id=f"model-{constraints.model_content_sha256[:12]}",
            hash=HashRef(value=constraints.model_content_sha256),
        )
        model = ExecutionInputBinding(
            kind="model",
            ref=model_ref,
            source="local_directory",
            location=constraints.base_model,
            content_sha256=constraints.model_content_sha256,
        )
        tokenizer_digest = constraints.tokenizer_content_sha256 or constraints.model_content_sha256
        tokenizer = ExecutionInputBinding(
            kind="tokenizer",
            ref=Ref(
                id=f"tokenizer-{tokenizer_digest[:12]}",
                hash=HashRef(value=tokenizer_digest),
            ),
            source="local_directory",
            location=constraints.base_model,
            content_sha256=tokenizer_digest,
        )
    else:
        revision = constraints.model_revision
        if revision is None:
            raise PlannerError(
                "base_model must be pinned with model_revision (immutable Hub commit) or a local "
                "model_content_sha256"
            )
        tokenizer_revision = constraints.tokenizer_revision or revision
        model = ExecutionInputBinding(
            kind="model",
            ref=huggingface_input_ref("model", constraints.base_model, revision),
            source="huggingface",
            location=constraints.base_model,
            resolved_revision=revision,
        )
        tokenizer = ExecutionInputBinding(
            kind="tokenizer",
            ref=huggingface_input_ref("tokenizer", constraints.base_model, tokenizer_revision),
            source="huggingface",
            location=constraints.base_model,
            resolved_revision=tokenizer_revision,
        )
    return ExecutionInputs(dataset=dataset, model=model, tokenizer=tokenizer)


def _has_verified_package_integrity(package: PackageLock) -> bool:
    record_hash = package.hash
    artifact_hash = package.artifact_hash
    installed_files_hash = package.installed_files_hash
    return (
        package.has_complete_record_count_evidence()
        and record_hash is not None
        and record_hash.value is not None
        and installed_files_hash is not None
        and installed_files_hash.value is not None
        and package.installed_file_count is not None
        and package.installed_file_count == package.record_entries
        and artifact_hash is not None
        and artifact_hash.value is not None
    )


def _trainer_interface(
    capabilities: CapabilityReport,
    *,
    cpu_toy: bool,
    quantized: bool,
    use_liger: bool,
    use_max_steps: bool,
    require_package_integrity: bool,
    external_attention_package: PackageLock | None,
) -> TrainerInterfacePolicy:
    effective = capabilities.effective_capabilities
    fields = set(effective.trainer_fields) if effective else set()
    init_fields = set(effective.trainer_init_fields) if effective else set()
    sequence_field = (
        "max_length"
        if "max_length" in fields
        else "max_seq_length"
        if "max_seq_length" in fields
        else None
    )
    tokenizer_parameter = (
        "processing_class"
        if "processing_class" in init_fields
        else "tokenizer"
        if "tokenizer" in init_fields
        else None
    )
    if sequence_field is None or tokenizer_parameter is None:
        raise PlannerError(
            "the capability report does not prove the exact SFTConfig/SFTTrainer field surface"
        )
    required = {
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "bf16",
        "data_seed",
        "dataset_text_field",
        "disable_tqdm",
        "fp16",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "learning_rate",
        "logging_nan_inf_filter",
        "logging_strategy",
        "logging_steps",
        "lr_scheduler_type",
        "optim",
        "output_dir",
        "packing",
        "per_device_train_batch_size",
        "report_to",
        "save_strategy",
        "seed",
        "warmup_ratio",
        "weight_decay",
        sequence_field,
        "max_steps" if use_max_steps else "num_train_epochs",
        "max_grad_norm",
    }
    if cpu_toy:
        required.add("use_cpu")
    if use_liger:
        required.add("use_liger_kernel")
    missing_fields = sorted(required - fields)
    if missing_fields:
        raise PlannerError(
            "the installed trainer cannot accept required semantic fields: "
            + ", ".join(missing_fields)
        )

    required_packages = {"accelerate", "datasets", "peft", "torch", "transformers", "trl"}
    if quantized:
        required_packages.add("bitsandbytes")
    if use_liger:
        required_packages.add("liger-kernel")
    if external_attention_package is not None:
        required_packages.add(external_attention_package.name.lower())
    installed = {
        item.name.lower(): item
        for item in capabilities.installed_packages
        if item.version is not None
    }
    missing_packages = sorted(required_packages - set(installed))
    if missing_packages:
        raise PlannerError(
            "the capability report lacks exact trainer package versions: "
            + ", ".join(missing_packages)
        )
    if require_package_integrity:
        unverified_packages = sorted(
            name
            for name in required_packages
            if not _has_verified_package_integrity(installed[name])
        )
        if unverified_packages:
            raise PlannerError(
                "managed trainer packages lack verified artifact, RECORD, or installed-file "
                "integrity evidence: " + ", ".join(unverified_packages)
            )
    return TrainerInterfacePolicy.model_validate(
        {
            "package_versions": [
                installed[name].model_dump(mode="json") for name in sorted(required_packages)
            ],
            "required_sft_config_fields": sorted(required),
            "sequence_length_field": sequence_field,
            "tokenizer_parameter": tokenizer_parameter,
            "logging_strategy": "steps",
            "logging_steps": 1,
            "logging_nan_inf_filter": False,
        }
    )


def _precision_policy(precision: str, quantization: str, optimizer: str) -> dict[str, Any]:
    quantized = quantization != "none"
    return {
        "weight_storage_dtype": None if quantized else precision,
        "quantized_storage_format": quantization,
        "dequantization_dtype": precision,
        "forward_compute_dtype": precision,
        "gradient_dtype": "fp32",
        "optimizer_state_dtype": "int8" if "8bit" in optimizer else "fp32",
        "optimizer_auxiliary_dtype": "fp32",
        "master_weight_dtype": "fp32",
    }


def build_run_plan(
    *,
    profile: EnvironmentProfile,
    capabilities: CapabilityReport,
    dataset_ref: Ref,
    constraints: PlannerConstraints,
    plan_id: str,
    environment_ref: Ref | None = None,
    parameter_accounting: ParameterAccountingReport | None = None,
    physical_execution: PhysicalExecutionSpec | None = None,
    storage_profile: StorageProfile | None = None,
    allow_marginal_storage: bool = False,
    allow_unknown_storage: bool = False,
    now: str | None = None,
) -> RunPlan:
    """Resolve one immutable, hash-sealed :class:`RunPlan` from the host profile + proven
    capabilities + dataset + user constraints. Raises :class:`PlannerError` when the host can't honor
    the request (not ready; cpu-toy-only without ``allow_cpu_toy``; an unsupported constraint)."""
    _require_enum(constraints.task_type, TaskType, "task_type")
    _require_enum(constraints.export_format, ExportFormat, "export_format")
    _require_enum(constraints.optim, Optimizer, "optimizer")
    _require_enum(
        constraints.verification_requirement,
        ExecutionVerificationRequirement,
        "verification_requirement",
    )
    if constraints.verification_requirement != "require_verified":
        raise PlannerError(
            "the first-party executor currently requires verified capability evidence; "
            "allow_unverified is represented for future research workers but is not executable"
        )

    effective = capabilities.effective_capabilities
    if capabilities.environment_ref.id != profile.environment_signature:
        raise PlannerError(
            "capability report environment does not match the profiled execution environment"
        )
    proven_precisions = {p.value for p in effective.precision_modes} if effective else set()
    proven_attn = {a.value for a in effective.attention_impls} if effective else set()
    proven_kernels = {item.value for item in effective.attention_kernels} if effective else set()
    proven_quantization = (
        {item.value for item in effective.quantization_modes} if effective else set()
    )
    proven_adapters = {item.value for item in effective.adapter_methods} if effective else set()
    proven_optimizers = {item.value for item in effective.optimizers} if effective else set()
    proven_losses = {item.value for item in effective.loss_impls} if effective else set()
    proven_checkpoints = (
        {item.value for item in effective.checkpoint_impls} if effective else set()
    )
    proven_execution_contracts = (
        set(effective.execution_contract_versions) if effective else set()
    )
    capability_ref = capability_report_ref_for(capabilities)
    cc_major = _max_cc_major(profile)

    # --- run mode (honest, readiness-driven) ---
    if capabilities.readiness == "ready":
        cpu_toy = False
    elif capabilities.readiness == "cpu_toy_only":
        if not constraints.allow_cpu_toy:
            raise PlannerError(
                "only the CPU-toy smoke path is available on this host; pass allow_cpu_toy to plan "
                "it, or provision a GPU + [train] runtime for a real run."
            )
        cpu_toy = True
    else:  # not_ready
        missing = ", ".join(capabilities.missing_packages) or "the training runtime"
        raise PlannerError(
            f"the environment is not ready for training (missing: {missing}); "
            "run 'corpus-studio train-check' to see what's needed."
        )

    # --- resolve the ambiguous fields against PROVEN capabilities ---
    if cpu_toy:
        if "fp32" not in proven_precisions:
            raise PlannerError("the CPU-toy path lacks a passing FP32 training-step probe")
        precision = "fp32"
        quantization = "none"
        attention_backend = AttentionImpl.eager.value
        attention_policy = _attention_policy(
            kernel=AttentionKernel.eager,
            kernel_probe_ref=capability_ref,
            evidence_kind="cpu_reference",
        )
    else:
        if "bf16" in proven_precisions:
            precision = "bf16"
        elif "fp32" in proven_precisions:
            precision = "fp32"
        else:
            raise PlannerError("no functionally proven training precision is available")
        quantization = (
            "nf4"
            if capabilities.bitsandbytes_ok and "nf4" in proven_quantization
            else "none"
        )
        attention_backend, attention_policy = _resolve_attention(
            constraints.attention_backend,
            cc_major,
            proven_attn,
            proven_kernels,
            os_value=profile.host.os,
            evidence_ref=capability_ref,
            flash_attention_package=next(
                (
                    item
                    for item in capabilities.installed_packages
                    if item.name == "flash-attn" and item.version is not None
                ),
                None,
            ),
        )

    adapter_method = constraints.adapter_method or ("qlora" if quantization == "nf4" else "lora")
    _require_enum(adapter_method, AdapterMethod, "adapter_method")
    if adapter_method not in proven_adapters and not cpu_toy:
        raise PlannerError(f"adapter '{adapter_method}' is not functionally proven")

    loss_impl = "liger_fused_ce" if constraints.use_liger else "cross_entropy"
    checkpoint_impl = "adapter_only"
    for label, value, proven in (
        ("optimizer", constraints.optim, proven_optimizers),
        ("loss", loss_impl, proven_losses),
        ("checkpoint", checkpoint_impl, proven_checkpoints),
    ):
        if value not in proven:
            raise PlannerError(f"{label} '{value}' is not functionally proven")
    expected_combination = {
        "runtime_mode": "cpu_toy" if cpu_toy else "training",
        "device": "cpu" if cpu_toy else "cuda",
        "precision": precision,
        "quantization": quantization,
        "adapter_method": adapter_method,
        "attention_impl": attention_backend,
        "attention_kernel": attention_policy.effective_backend_required.value,
        "optimizer": constraints.optim,
        "loss_impl": loss_impl,
        "checkpoint_impl": checkpoint_impl,
        "export_format": constraints.export_format,
        "execution_contract_version": _EXECUTION_CONTRACT_VERSION,
    }
    exact_combination = next(
        (
            item
            for item in (effective.execution_combinations if effective else [])
            if all(
                item.model_dump(mode="json")[field] == expected
                for field, expected in expected_combination.items()
            )
        ),
        None,
    )
    if exact_combination is None:
        rendered = ", ".join(
            f"{key}={value}" for key, value in expected_combination.items()
        )
        raise PlannerError(
            "no bounded functional probe demonstrated the complete requested execution tuple "
            f"({rendered})"
        )
    exact_probe_result = next(
        (
            result
            for result in capabilities.probe_results
            if result.probe == exact_combination.probe
            and result.outcome == FailureTaxonomy.PASS
            and exact_combination in result.execution_combinations
        ),
        None,
    )
    if exact_probe_result is None:
        raise PlannerError(
            "the selected execution combination is not embedded in its named passing probe result"
        )
    if _EXECUTION_CONTRACT_VERSION not in proven_execution_contracts:
        raise PlannerError(
            "the capability report does not prove resolved execution contract 1.0.0"
        )

    # Validate the chosen training backend can actually run the RESOLVED plan (declared support), so a
    # plan is never sealed for a framework that would silently downgrade or refuse it. This is where
    # "pick your framework" is enforced honestly — e.g. Unsloth (flash/sdpa only) is rejected for a
    # Blackwell math plan, and the fitting alternatives are named.
    backend = get_backend(constraints.backend)
    if backend is None:
        known = ", ".join(b.backend_id for b in builtin_backends())
        raise PlannerError(f"unknown backend '{constraints.backend}'; available: {known}.")
    if capabilities.backend_id != backend.backend_id:
        raise PlannerError(
            f"capability report belongs to backend '{capabilities.backend_id}', not "
            f"'{backend.backend_id}'"
        )
    if capabilities.backend_version != backend.backend_version:
        raise PlannerError(
            "capability report backend version does not match the selected manifest "
            f"(report={capabilities.backend_version!r}, manifest={backend.backend_version!r})"
        )
    device = "cpu" if cpu_toy else ("cuda" if profile.gpus else "cpu")
    host_os = profile.host.os.value
    unmet = unmet_requirements(
        backend,
        os=host_os,
        device=device,
        task_type=constraints.task_type,
        precision=precision,
        quantization=quantization,
        adapter_method=adapter_method,
        attention=attention_backend,
        attention_kernel=attention_policy.effective_backend_required.value,
        optimizer=constraints.optim,
        loss=loss_impl,
        checkpoint=checkpoint_impl,
        export_format=constraints.export_format,
        execution_contract_version=_EXECUTION_CONTRACT_VERSION,
    )
    if unmet:
        alternatives = [
            b.backend_id
            for b in compatible_backends(
                os=host_os,
                device=device,
                task_type=constraints.task_type,
                precision=precision,
                quantization=quantization,
                adapter_method=adapter_method,
                attention=attention_backend,
                attention_kernel=attention_policy.effective_backend_required.value,
                optimizer=constraints.optim,
                loss=loss_impl,
                checkpoint=checkpoint_impl,
                export_format=constraints.export_format,
                execution_contract_version=_EXECUTION_CONTRACT_VERSION,
            )
        ]
        hint = (
            f" Backends that fit: {', '.join(alternatives)}."
            if alternatives
            else " No registered backend fits this configuration."
        )
        raise PlannerError(f"backend '{backend.backend_id}' can't run this plan: {'; '.join(unmet)}.{hint}")

    resolved_physical = physical_execution or default_physical_execution(profile, cpu_toy=cpu_toy)
    try:
        resolved_physical = PhysicalExecutionSpec.model_validate(
            resolved_physical.model_dump(mode="json")
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise PlannerError(f"physical execution spec is structurally invalid: {exc}") from exc
    _validate_environment_resources(resolved_physical, profile)
    _validate_storage_profile(
        resolved_physical,
        storage_profile,
        allow_marginal=allow_marginal_storage,
        allow_unknown=allow_unknown_storage,
    )
    if resolved_physical.requires_parameter_accounting() and parameter_accounting is None:
        raise PlannerError(
            "scope-specific physical planning requires a hash-pinned parameter-accounting report"
        )
    parameter_accounting_ref = (
        _validate_parameter_accounting(parameter_accounting, resolved_physical)
        if parameter_accounting is not None
        else None
    )
    offload_strategy = _offload_summary(resolved_physical)
    if not is_trivial_physical_execution(resolved_physical):
        physical_unmet = unmet_physical_requirements(
            backend,
            effective,
            resolved_physical,
            offload_strategy=offload_strategy,
        )
        if physical_unmet:
            raise PlannerError(
                f"backend '{backend.backend_id}' can't run the physical plan: "
                + "; ".join(physical_unmet)
            )

    token_target = constraints.supervised_token_accumulation_target or max(
        1, constraints.sequence_len * constraints.micro_batch_size * constraints.gradient_accumulation_steps
    )

    resolved_environment_ref = environment_ref or Ref(
        id=profile.environment_signature,
        hash=HashRef(value=profile.environment_signature),
    )
    if (
        resolved_environment_ref.hash is None
        or resolved_environment_ref.hash.value is None
    ):
        raise PlannerError("the execution environment must be hash-pinned")
    execution_inputs = _resolved_execution_inputs(constraints, dataset_ref)

    adapter: dict[str, Any] = {"method": adapter_method}
    if adapter_method in _LORA_FAMILY:
        adapter["lora_r"] = constraints.lora_r
        adapter["lora_alpha"] = constraints.lora_alpha
        adapter["lora_dropout"] = constraints.lora_dropout
        adapter["target_modules"] = sorted(set(constraints.lora_target_modules))
        adapter["bias"] = constraints.lora_bias

    optimizer = {
        "impl": constraints.optim,
        "learning_rate": constraints.learning_rate,
        "weight_decay": constraints.weight_decay,
        "adam_beta1": constraints.adam_beta1,
        "adam_beta2": constraints.adam_beta2,
        "adam_epsilon": constraints.adam_epsilon,
        "max_grad_norm": constraints.max_grad_norm,
        "lr_scheduler": constraints.lr_scheduler,
        "warmup_ratio": constraints.warmup_ratio,
    }
    sequence = {
        "max_sequence_len": constraints.sequence_len,
        "packing": False,
        "truncation_allowed": constraints.truncation_allowed,
    }
    batching = {
        "micro_batch_size": constraints.micro_batch_size,
        "supervised_token_accumulation_target": token_target,
        "fallback_grad_accumulation_steps": constraints.gradient_accumulation_steps,
    }
    if (
        constraints.checkpoint_steps is not None
        or constraints.checkpoint_keep_last is not None
    ):
        raise PlannerError(
            "the first-party sealed runner cannot write intermediate checkpoints until exact "
            "resume compatibility and lineage are implemented"
        )
    checkpoint_policy = {
        "impl": checkpoint_impl,
        "cadence_optimizer_steps": None,
        "keep_last": None,
        "reload_verify": False,
    }
    schedule = TrainingSchedule(
        max_steps=(constraints.max_steps or 3) if cpu_toy else constraints.max_steps,
        num_train_epochs=(
            None
            if cpu_toy or constraints.max_steps is not None
            else constraints.num_train_epochs
        ),
    )
    trainer_interface = _trainer_interface(
        capabilities,
        cpu_toy=cpu_toy,
        quantized=quantization != "none",
        use_liger=constraints.use_liger,
        use_max_steps=schedule.max_steps is not None,
        require_package_integrity=environment_ref is not None,
        external_attention_package=attention_policy.flash_attention_package,
    )
    missing_manifest_fields = sorted(
        set(trainer_interface.required_sft_config_fields) - set(backend.trainer_fields)
    )
    if missing_manifest_fields:
        raise PlannerError(
            "the backend manifest does not declare required trainer fields: "
            + ", ".join(missing_manifest_fields)
        )
    if trainer_interface.tokenizer_parameter not in backend.trainer_init_fields:
        raise PlannerError(
            "the backend manifest does not declare the required trainer initializer field "
            f"{trainer_interface.tokenizer_parameter!r}"
        )
    formatter_id, formatter_hash = formatter_identity(constraints.dataset_format)
    data_policy = TrainingDataPolicy.model_validate(
        {
            "dataset_format": constraints.dataset_format,
            "formatter_id": formatter_id,
            "formatter_sha256": formatter_hash,
            "chat_template_sha256": constraints.chat_template_sha256,
            "truncation_policy": "allow" if constraints.truncation_allowed else "refuse",
            "packing": False,
        }
    )
    objective = get_objective("qlora" if adapter_method == "qlora" else "lora")
    if objective is None:  # pragma: no cover - sealed built-in catalog invariant
        raise PlannerError("the selected training objective is absent from the sealed registry")
    objective_ref = Ref(id=objective.objective_id, hash=HashRef(value=objective.objective_hash))

    root_device = resolved_physical.resources[0].device_id
    if root_device is None:
        raise PlannerError("the current dense trainer requires one explicit compute device")
    device_map = [
        DeviceMapEntry(module="", device="cpu" if root_device == "cpu:0" else root_device)
    ]
    try:
        execution_draft = ResolvedExecutionConfiguration.model_validate(
            {
            "configuration_id": f"{plan_id}-execution",
            "configuration_hash": "0" * 64,
            "backend_ref": backend_manifest_ref(backend).model_dump(mode="json"),
            "environment_ref": resolved_environment_ref.model_dump(mode="json"),
            "environment_binding": (
                "managed_lock" if environment_ref is not None else "profile_snapshot"
            ),
            "capability_report_ref": capability_ref.model_dump(mode="json"),
            "inputs": execution_inputs.model_dump(mode="json"),
            "objective_ref": objective_ref.model_dump(mode="json"),
            "runtime_mode": "cpu_toy" if cpu_toy else "training",
            "precision": _precision_policy(precision, quantization, constraints.optim),
            "attention": attention_policy.model_dump(mode="json"),
            "device_map": [item.model_dump(mode="json") for item in device_map],
            "adapter": adapter,
            "optimizer": optimizer,
            "loss_impl": loss_impl,
            "sequence": sequence,
            "batching": batching,
            "checkpoint_policy": checkpoint_policy,
            "schedule": schedule.model_dump(mode="json"),
            "data": data_policy.model_dump(mode="json"),
            "trainer_interface": trainer_interface.model_dump(mode="json"),
            "export_format": constraints.export_format,
            "trust_remote_code": False,
            "use_safetensors": True,
            "bnb_4bit_use_double_quant": quantization != "none",
            "adapter_task_type": "CAUSAL_LM",
            "save_strategy": "no",
            "gradient_checkpointing": True,
            "output_dir": constraints.output_dir,
            "output_layout": "run_scoped_v1",
            "seed": constraints.seed,
            "data_seed": constraints.data_seed if constraints.data_seed is not None else constraints.seed,
            }
        )
    except ValidationError as exc:
        raise PlannerError(f"the resolved execution configuration is invalid: {exc}") from exc
    execution = execution_draft.model_copy(
        update={"configuration_hash": execution_configuration_hash_for(execution_draft)}
    )

    body: dict[str, Any] = {
        "plan_id": plan_id,
        "plan_hash": "0" * 64,  # placeholder — replaced by the real seal below
        "backend_ref": backend_manifest_ref(backend).model_dump(mode="json"),
        "environment_ref": resolved_environment_ref.model_dump(mode="json"),
        "dataset_ref": dataset_ref.model_dump(mode="json"),
        "task_type": constraints.task_type,
        "base_model": constraints.base_model,
        "precision": precision,
        "quantization": quantization,
        "adapter": adapter,
        "optimizer": optimizer,
        "loss_impl": loss_impl,
        "attention_backend": attention_backend,
        "sequence": sequence,
        "batching": batching,
        "checkpoint_policy": checkpoint_policy,
        "offload_strategy": offload_strategy.value,
        "gradient_checkpointing": True,
        "export": {"format": constraints.export_format, "output_dir": constraints.output_dir},
        "seed": constraints.seed,
        "training_config_snapshot": {},
        "resolved_execution": execution.model_dump(mode="json"),
        "parameter_accounting_ref": (
            parameter_accounting_ref.model_dump(mode="json")
            if parameter_accounting_ref is not None
            else None
        ),
        "physical_execution": resolved_physical.model_dump(mode="json"),
    }

    try:
        draft = RunPlan.model_validate({**body, "created_at": None})
    except ValidationError as exc:
        raise PlannerError(f"the resolved plan is invalid: {exc}") from exc

    # Seal over the FULLY-DEFAULTED canonical plan, excluding the seal itself + the volatile stamp.
    plan_hash = compute_plan_hash(run_plan_hash_payload(draft))
    return draft.model_copy(update={"plan_hash": plan_hash, "created_at": now or _now_iso()})
