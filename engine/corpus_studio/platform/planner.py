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
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
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
    CustomModelCodeSpec,
    DeviceMapEntry,
    EnvironmentProfile,
    ExecutionInputBinding,
    ExecutionInputs,
    ModelInitializationSpec,
    ParameterAccountingReport,
    PhysicalExecutionSpec,
    PhysicalResource,
    PhysicalScopeSelector,
    ParallelismSpec,
    PreferenceDataPolicy,
    PreferenceOptimizationSpec,
    PretrainingDataPolicy,
    RankBinding,
    ReferenceModelBinding,
    ResolvedExecutionConfiguration,
    ResolvedFullFinetuneExecutionConfiguration,
    ResolvedPreferenceExecutionConfiguration,
    ResolvedRewardExecutionConfiguration,
    ResolvedRolloutExecutionConfiguration,
    RewardModelingSpec,
    RewardSourceRef,
    RolloutSpec,
    ExperienceSource,
    PolicyOptimizationSpec,
    StabilityController,
    ResolvedPretrainingExecutionConfiguration,
    RunManifest,
    RunPlan,
    StatePlacement,
    StorageProfile,
    TokenizerSourceSpec,
    TrainerInterfacePolicy,
    TrainingDataPolicy,
    TrainingSchedule,
)
from corpus_studio.platform.common import HashRef, PackageLock, Ref
from corpus_studio.platform.enums import (
    AdapterMethod,
    AllocatorPolicy,
    AttentionImpl,
    AttentionKernel,
    DeviceKind,
    ExecutionVerificationRequirement,
    ExportFormat,
    FailureTaxonomy,
    MemoryTier,
    ObjectiveKind,
    OffloadStrategy,
    OperatingSystem,
    Optimizer,
    PhysicalStateKind,
    PlacementRole,
    QuantizationMode,
    TaskType,
)
from corpus_studio.platform.execution_config import (
    ExecutionConfigurationError,
    canonical_sha256,
    capability_report_ref_for,
    execution_configuration_hash_for,
    full_finetune_execution_configuration_hash_for,
    formatter_identity,
    rollout_formatter_identity,
    huggingface_input_ref,
    preference_execution_configuration_hash_for,
    reward_execution_configuration_hash_for,
    preference_formatter_identity,
    pretraining_execution_configuration_hash_for,
    rollout_execution_configuration_hash_for,
    run_scoped_training_output,
    stable_file_sha256,
)
from corpus_studio.schemas.project_schemas import resolve_schema
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
    # The specific training objective. Required for a preference task (DPO/IPO/KTO/ORPO are distinct and
    # not interchangeable); the harness maps only built ones and refuses the rest fail-closed. SFT
    # resolves its objective from the adapter method, so this stays optional there.
    objective_id: str | None = None
    dataset_format: str = "instruction"
    adapter_method: str | None = None  # None → auto: qlora when quantized, else lora
    # Explicit precision/quantization override (nf4 | int8 | fp4 | none); None → auto-select the proven
    # default (nf4 when bitsandbytes + a passing nf4 probe, else none). A selected quantized mode must be
    # BOTH runnable (bitsandbytes present) AND proven by a capability probe on this host - else the planner
    # fails closed, so an unproven mode can never be sealed into a plan. 'none' (16-/32-bit on an
    # unquantized base) needs no quantization proof, only a proven precision.
    quantization: str | None = None
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
    # On-policy RL (S5b) reward-source provenance binding: the reward run's RunManifest (the admitted-run
    # proof) + its RunPlan (for the reward base model + adapter location). Required for a grpo plan.
    reward_source_manifest: str | None = None
    reward_source_plan: str | None = None
    supervised_token_accumulation_target: int | None = None
    attention_backend: str | None = None  # explicit override; else resolved from the host
    verification_requirement: str = "require_verified"
    export_format: str = "adapter_peft"
    backend: str = "corpus_studio"  # the training framework to run on (see platform.backends)
    # Memory / spill-avoidance levers (opt-in), validated against the backend's declared surface: a
    # paged optimizer (spill optimizer state to RAM) + a fused-CE loss (drop the long-seq logits spike).
    optim: str = "adamw_torch"
    use_liger: bool = False
    # Sealed CUDA allocator policy + its numeric parameter (PYTORCH_CUDA_ALLOC_CONF). max_split_size
    # requires allocator_max_split_size_mb; garbage_collection requires allocator_gc_threshold. The
    # seq-4096 paged config uses max_split_size (expandable_segments COLLIDES with a paged optimizer's
    # managed memory). Sealed into the plan hash so it is no longer smuggled via the dispatch env.
    allocator_policy: str = "default"
    allocator_max_split_size_mb: int | None = None
    allocator_gc_threshold: float | None = None
    max_steps: int | None = None
    num_train_epochs: float = 1.0
    # Intermediate first-party checkpoints remain disabled until a future execution contract can
    # seal compatible resume lineage. Non-null values are refused rather than written unusably.
    checkpoint_steps: int | None = None
    checkpoint_keep_last: int | None = None
    truncation_allowed: bool = False
    chat_template_sha256: str | None = None
    allow_cpu_toy: bool = False
    # Preference (DPO) knobs - consumed only by the preference resolver; ignored on the SFT/pretraining
    # paths. ``preference_max_prompt_length`` defaults to half the sequence window when None.
    preference_beta: float = 0.1
    preference_label_smoothing: float = 0.0
    preference_max_prompt_length: int | None = None
    # Pretraining (S3a-2): from-scratch / continued corpus + init + tokenizer knobs. ``init_mode`` None
    # marks a NON-pretraining plan; a pretraining task requires them (checked in _build_pretraining_plan).
    init_mode: str | None = None  # "random" | "continued"
    architecture_ref_id: str | None = None
    architecture_ref_sha256: str | None = None
    init_vocab_size: int | None = None
    init_seed: int | None = None
    init_initializer_range: float | None = None
    source_checkpoint_ref_id: str | None = None
    source_checkpoint_ref_sha256: str | None = None
    tokenizer_source_mode: str | None = None  # "train" | "import" | "freeze"
    tokenizer_algorithm: str | None = None
    tokenizer_vocab_size: int | None = None
    tokenizer_special_tokens: tuple[str, ...] | None = None
    tokenizer_min_frequency: int | None = None
    tokenizer_location: str | None = None  # import/freeze: where the worker loads the pinned tokenizer
    # Custom-block (mode 3): a hash-pinned, ADMITTED local code bundle. Set only for a custom_decoder
    # architecture; the platform-plan layer verifies the vetting report admitted these exact bytes first.
    custom_code_bundle_ref_id: str | None = None
    custom_code_bundle_ref_sha256: str | None = None
    custom_code_entry_symbol: str | None = None
    custom_code_interface_version: str | None = None
    custom_code_vetting_ref_id: str | None = None
    custom_code_vetting_ref_sha256: str | None = None


def _require_enum(value: str, enum_cls: type[Enum], label: str) -> None:
    valid = {member.value for member in enum_cls}
    if value not in valid:
        raise PlannerError(
            f"unsupported {label} '{value}'; expected one of: {', '.join(sorted(valid))}"
        )


def _validate_allocator_constraints(constraints: PlannerConstraints) -> None:
    """Fail closed at PLAN time on an allocator policy that cannot be honestly sealed + applied - the
    same bar the worker enforces, caught before dispatch so no wheel/GPU is spent. A parameterized
    policy needs its parameter, and a parameter without its policy is refused rather than silently
    ignored. ``expandable_segments`` with a paged optimizer is refused: the two collide in CUDA managed
    memory (the measured seq-4096 lesson) - a paged run must use ``max_split_size``."""
    policy = constraints.allocator_policy
    megabytes = constraints.allocator_max_split_size_mb
    threshold = constraints.allocator_gc_threshold
    if policy == AllocatorPolicy.max_split_size.value and megabytes is None:
        raise PlannerError("allocator_policy 'max_split_size' requires a max_split_size_mb value")
    if policy == AllocatorPolicy.garbage_collection.value and threshold is None:
        raise PlannerError("allocator_policy 'garbage_collection' requires a gc_threshold value")
    if megabytes is not None and policy != AllocatorPolicy.max_split_size.value:
        raise PlannerError("max_split_size_mb is only valid with allocator_policy 'max_split_size'")
    if threshold is not None and policy != AllocatorPolicy.garbage_collection.value:
        raise PlannerError("gc_threshold is only valid with allocator_policy 'garbage_collection'")
    if policy == AllocatorPolicy.expandable_segments.value and "paged" in (constraints.optim or ""):
        raise PlannerError(
            "allocator_policy 'expandable_segments' collides with a paged optimizer (CUDA "
            "managed-memory illegal access); use 'max_split_size' with a max_split_size_mb for a "
            "paged run"
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
    # Key by the PEP 503 normalized distribution name so a required package matches regardless of the
    # installed spelling (e.g. required "liger-kernel" vs the dist's reported "liger_kernel"). The
    # required_packages literals are already in normalized (hyphen) form.
    installed = {
        item.name.lower().replace("_", "-").replace(".", "-"): item
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


def _resolve_preference_execution(
    *,
    plan_id: str,
    shared_fields: dict[str, Any],
    constraints: PlannerConstraints,
    resolved_physical: PhysicalExecutionSpec,
    chat_template_sha256: str | None,
    project_dir: Path | str | None,
) -> ResolvedPreferenceExecutionConfiguration:
    """Lower a preference + ``dpo_qlora`` plan into a SEALED
    :class:`ResolvedPreferenceExecutionConfiguration` - the DPO sibling of the SFT resolver. Reuses every
    shared execution sub-spec already assembled in ``shared_fields`` and adds only what DPO needs: a
    :class:`PreferenceDataPolicy` (resolved ``preference`` schema identity + formatter + prompt/response
    budget) and a :class:`PreferenceOptimizationSpec` (beta / sigmoid loss / label-smoothing + frozen-base
    reference), bound to the sealed ``dpo_qlora`` objective. Admitted at planning; the runner refuses it at
    EXECUTION until the DPOTrainer branch + workload-verified evidence + milestone wheel land.

    ``beta`` / ``label_smoothing`` / ``max_prompt_length`` come from the operator's ``PlannerConstraints``
    (``preference_*`` knobs), defaulting to 0.1 / 0.0 / half-the-window when unset."""
    objective = get_objective("dpo_qlora")
    if objective is None:  # pragma: no cover - sealed built-in catalog invariant
        raise PlannerError("the dpo_qlora objective is absent from the sealed registry")
    objective_ref = Ref(id=objective.objective_id, hash=HashRef(value=objective.objective_hash))

    # Deferred #779 finding: the resolved device_map must reconcile with the sealed physical execution -
    # the single dense compute device is the placement root, never an unrelated device.
    root_device = resolved_physical.resources[0].device_id
    expected_device = "cpu" if root_device == "cpu:0" else root_device
    mapped_devices = {entry["device"] for entry in shared_fields["device_map"]}
    if expected_device is not None and mapped_devices != {expected_device}:
        raise PlannerError(
            f"preference device_map {sorted(mapped_devices)} does not reconcile with the sealed physical "
            f"execution device {expected_device!r}"
        )

    # Resolved 'preference' dataset-schema identity: id + version + a content digest of the schema that
    # ACTUALLY governs the dataset (a project-local schema shadows the builtin of the same id), so a
    # row-layout drift fails closed even under a project-local override.
    schema, _schema_source = resolve_schema(project_dir, "preference")
    # A project-local schema can make prompt/chosen/rejected optional or retype them; the sealed
    # chosen_rejected pair layout + format_preference_pair REQUIRE all three as text. Refuse an
    # incompatible resolved schema rather than sealing a config the formatter would reject at execution.
    _string_field_types = {"text", "markdown", "string"}
    schema_fields = {field.name: field for field in schema.fields}
    unusable = sorted(
        name
        for name in ("prompt", "chosen", "rejected")
        if (field := schema_fields.get(name)) is None
        or not getattr(field, "required", False)
        or field.type not in _string_field_types
    )
    if unusable:
        raise PlannerError(
            f"the resolved 'preference' schema is incompatible with the chosen_rejected pair formatter: "
            f"field(s) {unusable} must each be present, required, and text/markdown (a project-local "
            f"schema may have made them optional or retyped them)"
        )
    # The preference-pair formatter (prompt/chosen/rejected), NOT the SFT format_example_text.
    formatter_id, formatter_hash = preference_formatter_identity()
    max_length = constraints.sequence_len
    # Validate the operator's DPO knobs fail-closed (a clean PlannerError, never a silent clamp or a raw
    # pydantic error): an out-of-range or non-finite value hides operator intent rather than surfacing it.
    beta = constraints.preference_beta
    if not math.isfinite(beta) or beta <= 0:
        raise PlannerError(f"preference_beta must be a finite positive number (got {beta!r})")
    smoothing = constraints.preference_label_smoothing
    if not math.isfinite(smoothing) or not 0.0 <= smoothing < 0.5:
        raise PlannerError(f"preference_label_smoothing must be in [0, 0.5) (got {smoothing!r})")
    cap = constraints.preference_max_prompt_length
    if cap is not None and not 1 <= cap < max_length:
        raise PlannerError(
            f"preference_max_prompt_length must be in [1, {max_length}) - below the sequence window so "
            f"the response has room (got {cap!r})"
        )
    # The sealed prompt cap is the (validated) operator knob, else half the window.
    max_prompt_length = cap if cap is not None else max(1, max_length // 2)
    data_policy = PreferenceDataPolicy(
        schema_id=schema.id,
        schema_version=schema.version,
        schema_sha256=canonical_sha256(schema.model_dump(mode="json")),
        pair_schema="chosen_rejected",
        formatter_id=formatter_id,
        formatter_sha256=formatter_hash,
        chat_template_sha256=chat_template_sha256,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        truncation_policy="allow" if constraints.truncation_allowed else "refuse",
        data_seed=constraints.data_seed if constraints.data_seed is not None else constraints.seed,
    )
    preference_spec = PreferenceOptimizationSpec(
        beta=beta,
        label_smoothing=smoothing,
        reference_model=ReferenceModelBinding(mode="frozen_base", precompute_ref_log_probs=False),
    )
    try:
        draft = ResolvedPreferenceExecutionConfiguration.model_validate(
            {
                **shared_fields,
                "configuration_hash": "0" * 64,
                "objective_ref": objective_ref.model_dump(mode="json"),
                "data": data_policy.model_dump(mode="json"),
                "preference": preference_spec.model_dump(mode="json"),
            }
        )
    except ValidationError as exc:
        raise PlannerError(
            f"the resolved preference execution configuration is invalid: {exc}"
        ) from exc
    return draft.model_copy(
        update={"configuration_hash": preference_execution_configuration_hash_for(draft)}
    )


def _resolve_reward_execution(
    *,
    plan_id: str,
    shared_fields: dict[str, Any],
    constraints: PlannerConstraints,
    resolved_physical: PhysicalExecutionSpec,
    chat_template_sha256: str | None,
    project_dir: Path | str | None,
) -> ResolvedRewardExecutionConfiguration:
    """Lower a reward + ``reward_model`` plan into a SEALED :class:`ResolvedRewardExecutionConfiguration`
    - the reward sibling of the SFT/DPO resolvers (RL slice S5a-1). Reuses every shared execution sub-spec
    already assembled in ``shared_fields``, reuses :class:`PreferenceDataPolicy` (a reward model trains on
    the SAME chosen/rejected pairs), and adds a :class:`RewardModelingSpec` (pairwise Bradley-Terry), bound
    to the sealed ``reward_model`` objective. Overrides the export family to ``reward_model`` and the head
    to ``SEQ_CLS`` (a scalar score head, not the CAUSAL_LM policy). Admitted at planning; the runner refuses
    it at EXECUTION until the reward-head trainer branch + workload-verified evidence + milestone wheel."""
    del plan_id  # signature parity with the sibling resolvers; the id is sealed via shared_fields
    objective = get_objective("reward_model")
    if objective is None:  # pragma: no cover - sealed built-in catalog invariant
        raise PlannerError("the reward_model objective is absent from the sealed registry")
    objective_ref = Ref(id=objective.objective_id, hash=HashRef(value=objective.objective_hash))

    # The resolved device_map must reconcile with the sealed physical execution (mirrors the DPO resolver).
    root_device = resolved_physical.resources[0].device_id
    expected_device = "cpu" if root_device == "cpu:0" else root_device
    mapped_devices = {entry["device"] for entry in shared_fields["device_map"]}
    if expected_device is not None and mapped_devices != {expected_device}:
        raise PlannerError(
            f"reward device_map {sorted(mapped_devices)} does not reconcile with the sealed physical "
            f"execution device {expected_device!r}"
        )

    # A reward model scores prompt+chosen vs prompt+rejected - the SAME chosen_rejected pair layout as DPO,
    # so it resolves + validates the 'preference' schema identically (a project-local schema shadows the
    # builtin; the pair formatter requires prompt/chosen/rejected present, required, text/markdown).
    schema, _schema_source = resolve_schema(project_dir, "preference")
    _string_field_types = {"text", "markdown", "string"}
    schema_fields = {field.name: field for field in schema.fields}
    unusable = sorted(
        name
        for name in ("prompt", "chosen", "rejected")
        if (field := schema_fields.get(name)) is None
        or not getattr(field, "required", False)
        or field.type not in _string_field_types
    )
    if unusable:
        raise PlannerError(
            f"the resolved 'preference' schema is incompatible with the chosen_rejected pair formatter: "
            f"field(s) {unusable} must each be present, required, and text/markdown"
        )
    formatter_id, formatter_hash = preference_formatter_identity()
    max_length = constraints.sequence_len
    # The prompt cap is half the sealed window so the response has room; a reward-specific operator knob is
    # a later follow-up (this slice seals the shape at defaults, as the DPO slice initially did).
    max_prompt_length = max(1, max_length // 2)
    data_policy = PreferenceDataPolicy(
        schema_id=schema.id,
        schema_version=schema.version,
        schema_sha256=canonical_sha256(schema.model_dump(mode="json")),
        pair_schema="chosen_rejected",
        formatter_id=formatter_id,
        formatter_sha256=formatter_hash,
        chat_template_sha256=chat_template_sha256,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        truncation_policy="allow" if constraints.truncation_allowed else "refuse",
        data_seed=constraints.data_seed if constraints.data_seed is not None else constraints.seed,
    )
    reward_spec = RewardModelingSpec()
    try:
        draft = ResolvedRewardExecutionConfiguration.model_validate(
            {
                **shared_fields,
                "configuration_hash": "0" * 64,
                "objective_ref": objective_ref.model_dump(mode="json"),
                "data": data_policy.model_dump(mode="json"),
                "reward": reward_spec.model_dump(mode="json"),
                "export_format": ExportFormat.reward_model.value,
                "adapter_task_type": "SEQ_CLS",
            }
        )
    except ValidationError as exc:
        raise PlannerError(f"the resolved reward execution configuration is invalid: {exc}") from exc
    return draft.model_copy(
        update={"configuration_hash": reward_execution_configuration_hash_for(draft)}
    )


def _resolve_reward_source(constraints: PlannerConstraints) -> RewardSourceRef:
    """Bind the served reward source BY PROVENANCE (the chosen on-policy design, RL slice S5b): load the
    reward run's ``RunManifest`` (whose supervisor-admitted ``reward_success_evidence`` PROVES it came from
    an admitted reward run) + its ``RunPlan`` (for the reward base model + adapter location), cross-check
    they pair, and seal a hash-pinned, loadable :class:`RewardSourceRef`. Fail-closed on a missing /
    non-admitted / mismatched source - an unproven reward function must never silently drive an RL run."""
    from pathlib import Path  # noqa: PLC0415

    manifest_path = constraints.reward_source_manifest
    plan_path = constraints.reward_source_plan
    if not manifest_path or not plan_path:
        raise PlannerError(
            "an on-policy RL plan requires reward_source_manifest + reward_source_plan (the reward run's "
            "RunManifest + RunPlan) to bind a provenance-verified reward source"
        )
    try:
        manifest = RunManifest.model_validate_json(Path(manifest_path).read_bytes())
    except (OSError, ValidationError) as exc:
        raise PlannerError(f"the reward source RunManifest is unreadable or invalid: {exc}") from exc
    if manifest.state != "succeeded" or manifest.reward_success_evidence is None:
        raise PlannerError(
            "the reward source RunManifest is not an admitted reward run (needs state='succeeded' + "
            "supervisor-admitted reward_success_evidence)"
        )
    adapter_sha256 = manifest.reward_success_evidence.adapter_safetensors_sha256
    manifest_sha256 = stable_file_sha256(manifest_path)
    try:
        source_plan = RunPlan.model_validate_json(Path(plan_path).read_bytes())
    except (OSError, ValidationError) as exc:
        raise PlannerError(f"the reward source RunPlan is unreadable or invalid: {exc}") from exc
    reward_cfg = source_plan.resolved_reward_execution
    if reward_cfg is None:
        raise PlannerError("the reward source RunPlan carries no resolved reward execution")
    # Provenance integrity: the plan must be self-consistent (its body matches its own plan_hash) AND the
    # manifest must name THIS plan by CONTENT HASH, not merely share a plan_id string. A same-id but
    # content-swapped plan (a different base model / adapter location) is refused fail-closed.
    if not verify_run_plan_hash(source_plan):
        raise PlannerError("the reward source RunPlan is not self-consistent (plan_hash mismatch)")
    plan_ref = manifest.plan_ref
    if (
        plan_ref is None
        or plan_ref.id != source_plan.plan_id
        or plan_ref.hash is None
        or plan_ref.hash.value != source_plan.plan_hash
    ):
        raise PlannerError(
            "the reward source RunManifest does not bind this exact RunPlan (plan_ref id/hash mismatch)"
        )
    try:
        adapter_location = str(
            run_scoped_training_output(reward_cfg, manifest.run_id, leaf="adapter")
        )
    except ExecutionConfigurationError as exc:
        raise PlannerError(f"cannot resolve the reward adapter location: {exc}") from exc
    return RewardSourceRef(
        kind="served_reward_model",
        reward_ref=Ref(
            id=f"reward-adapter-{adapter_sha256[:12]}", hash=HashRef(value=adapter_sha256)
        ),
        reward_base_model=reward_cfg.inputs.model.location,
        reward_adapter_location=adapter_location,
        provenance_manifest_ref=Ref(
            id=f"reward-run-{manifest.run_id}", hash=HashRef(value=manifest_sha256)
        ),
    )


def _resolve_rollout_execution(
    *,
    plan_id: str,
    shared_fields: dict[str, Any],
    constraints: PlannerConstraints,
    resolved_physical: PhysicalExecutionSpec,
    chat_template_sha256: str | None,
    project_dir: Path | str | None,
) -> ResolvedRolloutExecutionConfiguration:
    """Lower a grpo + on-policy RL plan into a SEALED :class:`ResolvedRolloutExecutionConfiguration` - the
    on-policy sibling of the reward resolver (RL slice S5b). Reuses every shared execution sub-spec, binds
    the sealed ``grpo`` objective, and adds the on-policy specs: an :class:`ExperienceSource` (chat prompt
    stream), a :class:`RolloutSpec` (generation), a provenance-verified served :class:`RewardSourceRef`, a
    :class:`StabilityController` (KL/entropy/clip), and a :class:`PolicyOptimizationSpec` (GRPO). Keeps the
    CAUSAL_LM policy head + the ``adapter_peft`` policy export. Admitted at planning; the runner refuses it
    at EXECUTION until the rollout+reward+GRPO worker + workload-verified evidence + wheel land. Hyperparameters
    are sealed at sane GRPO defaults this slice (operator knobs are a follow-up, as reward/DPO initially did)."""
    del plan_id  # signature parity with the sibling resolvers; the id is sealed via shared_fields
    objective = get_objective("grpo")
    if objective is None:  # pragma: no cover - sealed built-in catalog invariant
        raise PlannerError("the grpo objective is absent from the sealed registry")
    objective_ref = Ref(id=objective.objective_id, hash=HashRef(value=objective.objective_hash))

    root_device = resolved_physical.resources[0].device_id
    expected_device = "cpu" if root_device == "cpu:0" else root_device
    mapped_devices = {entry["device"] for entry in shared_fields["device_map"]}
    if expected_device is not None and mapped_devices != {expected_device}:
        raise PlannerError(
            f"rollout device_map {sorted(mapped_devices)} does not reconcile with the sealed physical "
            f"execution device {expected_device!r}"
        )

    # On-policy RL draws PROMPTS (the model generates the completion); resolve + validate the 'chat' schema.
    # Seal the ROLLOUT prompt formatter (add_generation_prompt=True), NOT the SFT 'chat' formatter that
    # renders a full example - the worker generates from a prompt, so its sealed identity must hash the
    # generation-prompt formatter it actually runs (reproducible-from-seal).
    schema, _schema_source = resolve_schema(project_dir, "chat")
    formatter_id, formatter_hash = rollout_formatter_identity()
    max_length = constraints.sequence_len
    # The prompt cap + the generation budget must both fit the window; seal sane defaults (half the window
    # for the prompt, a quarter for the generation) this slice.
    max_prompt_length = max(1, max_length // 2)
    max_new_tokens = max(1, max_length // 4)
    experience = ExperienceSource(
        schema_id=schema.id,
        schema_version=schema.version,
        schema_sha256=canonical_sha256(schema.model_dump(mode="json")),
        formatter_id=formatter_id,
        formatter_sha256=formatter_hash,
        chat_template_sha256=chat_template_sha256,
        max_prompt_length=max_prompt_length,
        truncation_policy="allow" if constraints.truncation_allowed else "refuse",
        data_seed=constraints.data_seed if constraints.data_seed is not None else constraints.seed,
    )
    rollout = RolloutSpec(
        sampling_temperature=1.0,
        sampling_top_p=0.95,
        max_new_tokens=max_new_tokens,
        rollouts_per_prompt=4,
    )
    stability = StabilityController(kl_coefficient=0.05, clip_range=0.2)
    policy = PolicyOptimizationSpec(algorithm="grpo", use_critic=False)
    reward_source = _resolve_reward_source(constraints)
    try:
        draft = ResolvedRolloutExecutionConfiguration.model_validate(
            {
                **shared_fields,
                "configuration_hash": "0" * 64,
                "objective_ref": objective_ref.model_dump(mode="json"),
                "experience": experience.model_dump(mode="json"),
                "rollout": rollout.model_dump(mode="json"),
                "reward_source": reward_source.model_dump(mode="json"),
                "stability": stability.model_dump(mode="json"),
                "policy_optimization": policy.model_dump(mode="json"),
                "export_format": ExportFormat.adapter_peft.value,
                "adapter_task_type": "CAUSAL_LM",
            }
        )
    except ValidationError as exc:
        # A metric-driven lr_scheduler (reduce_lr_on_plateau) is refused by the rollout CONTRACT validator so
        # the refusal also covers imported/hand-built plans, not just this resolver; surfaces here as a
        # PlannerError via this ValidationError wrap.
        raise PlannerError(f"the resolved rollout execution configuration is invalid: {exc}") from exc
    return draft.model_copy(
        update={"configuration_hash": rollout_execution_configuration_hash_for(draft)}
    )


def _custom_code_spec(constraints: PlannerConstraints) -> CustomModelCodeSpec | None:
    """The ADMITTED custom-block spec when platform-plan verified + set its fields (all-or-nothing); else
    None. Any partial set trips the contract validators, surfaced as a PlannerError by the caller."""
    if constraints.custom_code_bundle_ref_id is None:
        return None
    return CustomModelCodeSpec(
        code_bundle_ref=Ref(
            id=constraints.custom_code_bundle_ref_id,
            hash=HashRef(value=constraints.custom_code_bundle_ref_sha256),
        ),
        entry_symbol=constraints.custom_code_entry_symbol or "",
        interface_version=constraints.custom_code_interface_version,  # type: ignore[arg-type]
        vetting_ref=Ref(
            id=constraints.custom_code_vetting_ref_id or "",
            hash=HashRef(value=constraints.custom_code_vetting_ref_sha256),
        ),
        vetting_verdict="admitted",
    )


def _pretraining_init_spec(constraints: PlannerConstraints) -> ModelInitializationSpec:
    """Lower the operator's init knobs into a sealed :class:`ModelInitializationSpec`, fail-closed with a
    clean :class:`PlannerError` (never a raw pydantic error) so missing intent surfaces, not hides."""
    try:
        if constraints.init_mode == "random":
            if constraints.architecture_ref_id is None or constraints.architecture_ref_sha256 is None:
                raise PlannerError("random-init pretraining requires an architecture ref (id + sha256)")
            # A from-scratch model sizes its embedding to its tokenizer: default the model vocab to the
            # trained tokenizer's vocab when the operator does not pin one explicitly (the config seal
            # then enforces they agree).
            vocab_size = constraints.init_vocab_size
            if vocab_size is None and constraints.tokenizer_source_mode == "train":
                vocab_size = constraints.tokenizer_vocab_size
            if vocab_size is None:
                raise PlannerError(
                    "random-init pretraining requires init_vocab_size (or a trained tokenizer vocab_size)"
                )
            return ModelInitializationSpec(
                mode="random",
                architecture_ref=Ref(
                    id=constraints.architecture_ref_id,
                    hash=HashRef(value=constraints.architecture_ref_sha256),
                ),
                vocab_size=vocab_size,
                init_seed=(
                    constraints.init_seed if constraints.init_seed is not None else constraints.seed
                ),
                initializer_range=constraints.init_initializer_range,
                custom_code=_custom_code_spec(constraints),
            )
        if (
            constraints.source_checkpoint_ref_id is None
            or constraints.source_checkpoint_ref_sha256 is None
        ):
            raise PlannerError("continued pretraining requires a source checkpoint ref (id + sha256)")
        return ModelInitializationSpec(
            mode="continued",
            source_checkpoint_ref=Ref(
                id=constraints.source_checkpoint_ref_id,
                hash=HashRef(value=constraints.source_checkpoint_ref_sha256),
            ),
        )
    except ValidationError as exc:
        raise PlannerError(f"the pretraining model initialization is invalid: {exc}") from exc


def _tokenizer_source_spec(constraints: PlannerConstraints) -> TokenizerSourceSpec:
    """Lower the operator's tokenizer knobs into a sealed :class:`TokenizerSourceSpec`, fail-closed."""
    # model_validate (a dict) rather than the typed constructor: the operator's raw CLI strings are
    # validated against the sealed Literals at RUNTIME (fail-closed below), not asserted by the type
    # checker over an untrusted str.
    try:
        if constraints.tokenizer_source_mode == "train":
            return TokenizerSourceSpec.model_validate(
                {
                    "mode": "train",
                    "algorithm": constraints.tokenizer_algorithm,
                    "vocab_size": constraints.tokenizer_vocab_size,
                    "special_tokens": (
                        list(constraints.tokenizer_special_tokens)
                        if constraints.tokenizer_special_tokens is not None
                        else None
                    ),
                    "min_frequency": constraints.tokenizer_min_frequency,
                }
            )
        return TokenizerSourceSpec.model_validate(
            {
                "mode": constraints.tokenizer_source_mode,
                "tokenizer_content_sha256": constraints.tokenizer_content_sha256,
                "tokenizer_location": constraints.tokenizer_location,
            }
        )
    except ValidationError as exc:
        raise PlannerError(f"the pretraining tokenizer source is invalid: {exc}") from exc


def _build_pretraining_plan(
    *,
    profile: EnvironmentProfile,
    capabilities: CapabilityReport,
    constraints: PlannerConstraints,
    plan_id: str,
    environment_ref: Ref | None,
    physical_execution: PhysicalExecutionSpec | None,
    pretraining_data: PretrainingDataPolicy | None,
    now: str | None,
) -> RunPlan:
    """Lower a from-scratch / continued PRETRAINING request into a sealed RunPlan carrying a
    :class:`ResolvedPretrainingExecutionConfiguration`. FULL-PARAMETER (no adapter, no 4-bit base, no
    single dataset file): the corpus is the sharded :class:`PretrainingDataPolicy`, the model an init
    spec, the tokenizer a source spec. Admitted at planning and, once the 'pretraining' variant is
    workload_verified for the backend, executed by the first-party PretrainingRunner lane through
    platform-run. Lives OUTSIDE the SFT/DPO ``build_run_plan`` body so the byte-locked SFT seal path
    stays byte-identical."""
    if pretraining_data is None:
        raise PlannerError(
            "a pretraining plan requires a corpus (a PretrainingDataPolicy of content-hashed shards)"
        )
    if constraints.init_mode not in {"random", "continued"}:
        raise PlannerError("pretraining requires init_mode 'random' or 'continued'")
    if constraints.tokenizer_source_mode not in {"train", "import", "freeze"}:
        raise PlannerError("pretraining requires tokenizer_source 'train', 'import', or 'freeze'")

    effective = capabilities.effective_capabilities
    if capabilities.environment_ref.id != profile.environment_signature:
        raise PlannerError(
            "capability report environment does not match the profiled execution environment"
        )
    proven_precisions = {p.value for p in effective.precision_modes} if effective else set()
    proven_attn = {a.value for a in effective.attention_impls} if effective else set()
    proven_kernels = {item.value for item in effective.attention_kernels} if effective else set()
    capability_ref = capability_report_ref_for(capabilities)
    cc_major = _max_cc_major(profile)

    if capabilities.readiness == "ready":
        cpu_toy = False
    elif capabilities.readiness == "cpu_toy_only":
        if not constraints.allow_cpu_toy:
            raise PlannerError(
                "only the CPU-toy smoke path is available on this host; pass allow_cpu_toy to plan it"
            )
        cpu_toy = True
    else:  # not_ready
        missing = ", ".join(capabilities.missing_packages) or "the training runtime"
        raise PlannerError(f"the environment is not ready for training (missing: {missing})")

    if cpu_toy:
        if "fp32" not in proven_precisions:
            raise PlannerError("the CPU-toy path lacks a passing FP32 training-step probe")
        precision = "fp32"
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

    backend = get_backend(constraints.backend)
    if backend is None:
        raise PlannerError(f"unknown training backend '{constraints.backend}'")
    resolved_physical = physical_execution or default_physical_execution(profile, cpu_toy=cpu_toy)
    root_device = resolved_physical.resources[0].device_id
    if root_device is None:
        raise PlannerError("the current dense trainer requires one explicit compute device")
    device_map = [DeviceMapEntry(module="", device="cpu" if root_device == "cpu:0" else root_device)]
    resolved_environment_ref = environment_ref or Ref(
        id=profile.environment_signature,
        hash=HashRef(value=profile.environment_signature),
    )
    if resolved_environment_ref.hash is None or resolved_environment_ref.hash.value is None:
        raise PlannerError("the execution environment must be hash-pinned")

    token_target = constraints.supervised_token_accumulation_target or max(
        1,
        constraints.sequence_len
        * constraints.micro_batch_size
        * constraints.gradient_accumulation_steps,
    )
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
    # Pretraining PACKS documents (concat-and-split with boundaries per the PretrainingDataPolicy).
    sequence = {
        "max_sequence_len": constraints.sequence_len,
        "packing": True,
        "truncation_allowed": constraints.truncation_allowed,
    }
    batching = {
        "micro_batch_size": constraints.micro_batch_size,
        "supervised_token_accumulation_target": token_target,
        "fallback_grad_accumulation_steps": constraints.gradient_accumulation_steps,
    }
    # Full-parameter pretraining checkpoints the WHOLE model (never adapter-only). Cadence lives on the
    # PretrainingDataPolicy's later worker slice; the plan seals a checkpoint-free policy for now.
    checkpoint_policy = {
        "impl": "full_state",
        "cadence_optimizer_steps": None,
        "keep_last": None,
        "reload_verify": False,
    }
    schedule = TrainingSchedule(
        max_steps=(constraints.max_steps or 3) if cpu_toy else constraints.max_steps,
        num_train_epochs=(
            None if cpu_toy or constraints.max_steps is not None else constraints.num_train_epochs
        ),
    )
    trainer_interface = _trainer_interface(
        capabilities,
        cpu_toy=cpu_toy,
        quantized=False,
        use_liger=constraints.use_liger,
        use_max_steps=schedule.max_steps is not None,
        require_package_integrity=environment_ref is not None,
        external_attention_package=attention_policy.flash_attention_package,
    )
    loss_impl = "liger_fused_ce" if constraints.use_liger else "cross_entropy"

    continued = constraints.init_mode == "continued"
    objective = get_objective("continued_pretraining" if continued else "pretraining")
    if objective is None:  # pragma: no cover - sealed built-in catalog invariant
        raise PlannerError("the pretraining objective is absent from the sealed registry")
    objective_ref = Ref(id=objective.objective_id, hash=HashRef(value=objective.objective_hash))
    init = _pretraining_init_spec(constraints)
    tokenizer_source = _tokenizer_source_spec(constraints)
    # Full-parameter pretraining emits a full model, never a PEFT adapter.
    export_format = (
        constraints.export_format
        if constraints.export_format != ExportFormat.adapter_peft.value
        else ExportFormat.merged_safetensors.value
    )

    try:
        draft_config = ResolvedPretrainingExecutionConfiguration.model_validate(
            {
                "configuration_id": f"{plan_id}-execution",
                "configuration_hash": "0" * 64,
                "backend_ref": backend_manifest_ref(backend).model_dump(mode="json"),
                "environment_ref": resolved_environment_ref.model_dump(mode="json"),
                "environment_binding": (
                    "managed_lock" if environment_ref is not None else "profile_snapshot"
                ),
                "capability_report_ref": capability_ref.model_dump(mode="json"),
                "objective_ref": objective_ref.model_dump(mode="json"),
                "runtime_mode": "cpu_toy" if cpu_toy else "training",
                "init": init.model_dump(mode="json"),
                "tokenizer_source": tokenizer_source.model_dump(mode="json"),
                "precision": _precision_policy(precision, "none", constraints.optim),
                "attention": attention_policy.model_dump(mode="json"),
                "device_map": [item.model_dump(mode="json") for item in device_map],
                "optimizer": optimizer,
                "loss_impl": loss_impl,
                "sequence": sequence,
                "batching": batching,
                "checkpoint_policy": checkpoint_policy,
                "schedule": schedule.model_dump(mode="json"),
                "data": pretraining_data.model_dump(mode="json"),
                "trainer_interface": trainer_interface.model_dump(mode="json"),
                "export_format": export_format,
                "gradient_checkpointing": True,
                "output_dir": constraints.output_dir,
                "seed": constraints.seed,
                "data_seed": pretraining_data.data_seed,
            }
        )
    except ValidationError as exc:
        raise PlannerError(f"the resolved pretraining execution configuration is invalid: {exc}") from exc
    pretraining_execution = draft_config.model_copy(
        update={"configuration_hash": pretraining_execution_configuration_hash_for(draft_config)}
    )

    # The RunPlan's single dataset_ref is a hash-pinned handle over the whole corpus (the real shard set
    # + its per-shard digests live in the sealed PretrainingDataPolicy on the execution config).
    corpus_ref = Ref(
        id=f"corpus:{plan_id}",
        hash=HashRef(value=canonical_sha256(pretraining_data.model_dump(mode="json"))),
    )
    body: dict[str, Any] = {
        "plan_id": plan_id,
        "plan_hash": "0" * 64,
        "backend_ref": backend_manifest_ref(backend).model_dump(mode="json"),
        "environment_ref": resolved_environment_ref.model_dump(mode="json"),
        "dataset_ref": corpus_ref.model_dump(mode="json"),
        "task_type": "pretraining",
        "base_model": constraints.base_model,
        "precision": precision,
        "quantization": "none",
        "adapter": {"method": "full_finetune"},
        "optimizer": optimizer,
        "loss_impl": loss_impl,
        "attention_backend": attention_backend,
        "sequence": sequence,
        "batching": batching,
        "checkpoint_policy": checkpoint_policy,
        "gradient_checkpointing": True,
        "export": {"format": export_format, "output_dir": constraints.output_dir},
        "seed": constraints.seed,
        "training_config_snapshot": {},
        "resolved_execution": None,
        "resolved_preference_execution": None,
        "resolved_pretraining_execution": pretraining_execution.model_dump(mode="json"),
        "physical_execution": resolved_physical.model_dump(mode="json"),
    }
    try:
        draft = RunPlan.model_validate({**body, "created_at": None})
    except ValidationError as exc:
        raise PlannerError(f"the resolved pretraining plan is invalid: {exc}") from exc
    plan_hash = compute_plan_hash(run_plan_hash_payload(draft))
    return draft.model_copy(update={"plan_hash": plan_hash, "created_at": now or _now_iso()})


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
    project_dir: Path | str | None = None,
    pretraining_data: PretrainingDataPolicy | None = None,
    now: str | None = None,
) -> RunPlan:
    """Resolve one immutable, hash-sealed :class:`RunPlan` from the host profile + proven
    capabilities + dataset + user constraints. Raises :class:`PlannerError` when the host can't honor
    the request (not ready; cpu-toy-only without ``allow_cpu_toy``; an unsupported constraint)."""
    _require_enum(constraints.task_type, TaskType, "task_type")
    # Fail-closed execution-variant admission (#484 wired): refuse at planning any task whose execution
    # shape the first-party harness cannot execute - only the dense-QLoRA-SFT shape is workload_verified;
    # pretraining/MoE are declared-only and every other objective has no built execution path yet. Never
    # falls back to dense_qlora_sft. The per-variant executable seal + worker land with each variant (S2+).
    from corpus_studio.platform.execution_variants import (  # noqa: PLC0415
        ExecutionVariantRefused,
        ExecutionVariantSupport,
        admit_task_execution_variant,
        reference_execution_variants,
    )

    # A MoE model routes to the declared-only 'moe' execution shape (refused), never a dense shape:
    # detect it from the sealed parameter accounting's expert/router scopes when one is provided.
    plan_targets_moe = parameter_accounting is not None and any(
        observation.scope.kind.value in {"expert_group", "expert_set", "router"}
        for observation in parameter_accounting.observations
    )
    # A preference request resolves by its EXPLICIT objective, not the task alone (DPO/IPO/KTO/ORPO
    # differ). The caller names it via constraints.objective_id; admission maps only the built ones
    # (dpo_qlora -> preference_dpo, declared at contract_validated so it is refused at execution until
    # the worker lands) and refuses the rest fail-closed. SFT/pretraining resolve by task alone.
    # A preference OBJECTIVE requires the preference TASK. Without this guard a plan that names a
    # preference objective (e.g. dpo_qlora) while leaving task_type=sft would silently lower to a dense-SFT
    # run - build_run_plan ignores objective_id for non-preference tasks - retaining a misleading DPO
    # identity in the resolution. Refuse the contradiction fail-closed.
    if constraints.objective_id is not None and constraints.task_type != TaskType.preference.value:
        named_objective = get_objective(constraints.objective_id)
        if named_objective is not None and named_objective.kind == ObjectiveKind.preference_optimization:
            raise PlannerError(
                f"objective '{constraints.objective_id}' is a preference objective but task_type is "
                f"'{constraints.task_type}' - a preference objective requires task_type='preference'"
            )
    # The same guard for the reward family: a reward objective on a non-reward task would silently lower
    # to dense-SFT (build_run_plan ignores objective_id off its family task), keeping a misleading reward
    # identity. Refuse the contradiction fail-closed.
    if constraints.objective_id is not None and constraints.task_type != TaskType.reward.value:
        named_objective = get_objective(constraints.objective_id)
        if named_objective is not None and named_objective.kind == ObjectiveKind.reward_modeling:
            raise PlannerError(
                f"objective '{constraints.objective_id}' is a reward objective but task_type is "
                f"'{constraints.task_type}' - a reward objective requires task_type='reward'"
            )
    # The same guard for the on-policy RL family: a grpo (on_policy_rl) objective on a non-grpo task would
    # silently lower to dense-SFT, keeping a misleading RL identity. Refuse the contradiction fail-closed.
    if constraints.objective_id is not None and constraints.task_type != TaskType.grpo.value:
        named_objective = get_objective(constraints.objective_id)
        if named_objective is not None and named_objective.kind == ObjectiveKind.on_policy_rl:
            raise PlannerError(
                f"objective '{constraints.objective_id}' is an on-policy RL objective but task_type is "
                f"'{constraints.task_type}' - an on-policy RL objective requires task_type='grpo'"
            )
    # Preference AND reward are objective-keyed families (their specific objective selects the shape);
    # SFT / pretraining resolve by task alone.
    admission_objective_id = (
        constraints.objective_id
        if constraints.task_type
        in (TaskType.preference.value, TaskType.reward.value, TaskType.grpo.value)
        else None
    )
    # QLoRA-DPO is admitted AT PLANNING (contract_validated) so the resolver can seal a reviewable
    # ResolvedPreferenceExecutionConfiguration; it is then refused AT EXECUTION by the runner (the
    # DPOTrainer branch + workload-verified evidence + wheel are the gated milestone). Every other shape
    # keeps the workload_verified bar - MoE / non-DPO preference / unbuilt objectives stay refused here.
    is_preference_dpo = (
        constraints.task_type == TaskType.preference.value
        and admission_objective_id == "dpo_qlora"
        and not plan_targets_moe
    )
    # Pairwise reward model (RL slice S5a-1): admitted AT PLANNING (contract_validated) so the resolver
    # seals a reviewable ResolvedRewardExecutionConfiguration; refused AT EXECUTION until the reward-head
    # worker + evidence + wheel land. Keyed on the specific reward objective (only reward_model is built).
    is_reward_model = (
        constraints.task_type == TaskType.reward.value
        and admission_objective_id == "reward_model"
        and not plan_targets_moe
    )
    # On-policy RL (RL slice S5b): admitted AT PLANNING (contract_validated) so the resolver seals a
    # reviewable ResolvedRolloutExecutionConfiguration; refused AT EXECUTION until the rollout+reward+GRPO
    # worker + evidence + wheel land. Keyed on the specific on-policy objective (only grpo is built).
    is_rollout = (
        constraints.task_type == TaskType.grpo.value
        and admission_objective_id == "grpo"
        and not plan_targets_moe
    )
    # Pretraining (from-scratch / continued) resolves by task alone (no objective_id); its dedicated
    # builder seals a reviewable config, and once the 'pretraining' variant is workload_verified for the
    # backend the first-party PretrainingRunner lane executes it. A MoE topology routes to the
    # declared-only 'moe' shape and stays refused here.
    is_pretraining = constraints.task_type == TaskType.pretraining.value and not plan_targets_moe
    # A full-parameter dense SFT (--adapter-method full_finetune) seals a reviewable full-model config and
    # maps to the dense_full_finetune shape. Admitted at planning (contract_validated) and refused at
    # EXECUTION until that variant is workload_verified - the full-parameter worker + full-model evidence
    # are the gated milestone, exactly as DPO/pretraining. All OTHER dense SFT stays the QLoRA shape.
    is_full_finetune = (
        constraints.task_type == TaskType.sft.value
        and constraints.adapter_method == AdapterMethod.full_finetune.value
        and not plan_targets_moe
    )
    try:
        admit_task_execution_variant(
            TaskType(constraints.task_type),
            is_moe=plan_targets_moe,
            objective_id=admission_objective_id,
            is_full_parameter=is_full_finetune,
            declared_variants=reference_execution_variants(),
            required_support=(
                ExecutionVariantSupport.contract_validated
                if (
                    is_preference_dpo
                    or is_pretraining
                    or is_full_finetune
                    or is_reward_model
                    or is_rollout
                )
                else ExecutionVariantSupport.workload_verified
            ),
        )
    except ExecutionVariantRefused as exc:
        raise PlannerError(str(exc)) from exc
    # Pretraining is lowered by its OWN builder so the byte-locked SFT/DPO body below stays untouched.
    if is_pretraining:
        return _build_pretraining_plan(
            profile=profile,
            capabilities=capabilities,
            constraints=constraints,
            plan_id=plan_id,
            environment_ref=environment_ref,
            physical_execution=physical_execution,
            pretraining_data=pretraining_data,
            now=now,
        )
    _require_enum(constraints.export_format, ExportFormat, "export_format")
    _require_enum(constraints.optim, Optimizer, "optimizer")
    if constraints.quantization is not None:
        # A typo fails closed here with the known set, before the proven/bitsandbytes gate below.
        _require_enum(constraints.quantization, QuantizationMode, "quantization")
    _require_enum(constraints.allocator_policy, AllocatorPolicy, "allocator_policy")
    _validate_allocator_constraints(constraints)
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
        if is_full_finetune:
            # Full-parameter fine-tuning trains all weights, so they must stay in a trainable dtype - never
            # 4-bit frozen. Force 'none' and reject any quantized override rather than silently ignore it.
            if constraints.quantization not in (None, "none"):
                raise PlannerError(
                    f"full-parameter fine-tuning cannot be quantized; drop --quantization "
                    f"'{constraints.quantization}' (it trains all weights in a full-precision dtype)."
                )
            quantization = "none"
        elif constraints.quantization is not None:
            # An explicit precision/quantization override. Honor it ONLY when honestly runnable here:
            # 'none' (16-/32-bit on an unquantized base) needs no quantization proof; any quantized mode
            # needs bitsandbytes AND a passing capability probe on this host. An unproven mode fails closed
            # instead of sealing a plan that would break at execution ("declared" is not "proven").
            requested = constraints.quantization
            if requested != "none":
                if not capabilities.bitsandbytes_ok:
                    raise PlannerError(
                        f"quantization '{requested}' needs bitsandbytes, which is not available in this "
                        "environment; run 'corpus-studio train-check' to see what is missing."
                    )
                if requested not in proven_quantization:
                    proven_list = ", ".join(sorted(proven_quantization)) or "none"
                    raise PlannerError(
                        f"quantization '{requested}' is not proven on this host - a capability probe must "
                        f"pass for it before it can be planned (proven quantized modes: {proven_list})."
                    )
            quantization = requested
        else:
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

    # qlora = a LoRA adapter over a QUANTIZED frozen base, for ANY quant type (nf4/fp4/int8) - not just
    # nf4. An unquantized base ('none') gets a plain lora adapter. (nf4 is unchanged; this only sharpens
    # the newly-selectable int8/fp4 so their adapter reads 'qlora', matching the int8 execution probe.)
    adapter_method = constraints.adapter_method or ("lora" if quantization == "none" else "qlora")
    _require_enum(adapter_method, AdapterMethod, "adapter_method")
    # full_finetune is admitted at PLANNING even though the backend cannot yet prove it (the full-parameter
    # worker + wheel are the gated milestone); it is refused at EXECUTION by required_runner_lane. Every
    # other adapter must be functionally proven here.
    if adapter_method not in proven_adapters and not cpu_toy and not is_full_finetune:
        raise PlannerError(f"adapter '{adapter_method}' is not functionally proven")
    if is_full_finetune and constraints.export_format == ExportFormat.adapter_peft.value:
        raise PlannerError(
            "full-parameter fine-tuning produces a full model, not an adapter - pass "
            "--export-format merged_safetensors"
        )

    loss_impl = "liger_fused_ce" if constraints.use_liger else "cross_entropy"
    # A full fine-tune checkpoints the FULL model state; adapter runs checkpoint only the adapter.
    checkpoint_impl = "full_state" if is_full_finetune else "adapter_only"
    axis_checks = [
        ("optimizer", constraints.optim, proven_optimizers),
        ("loss", loss_impl, proven_losses),
    ]
    if not is_full_finetune:
        # full_state checkpoints are not yet functionally proven for the SFT backend (the milestone wheel
        # proves them); admitted at planning, refused at execution alongside the adapter/export reasons.
        axis_checks.append(("checkpoint", checkpoint_impl, proven_checkpoints))
    for label, value, proven in axis_checks:
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
    if exact_combination is None and not is_full_finetune:
        rendered = ", ".join(
            f"{key}={value}" for key, value in expected_combination.items()
        )
        raise PlannerError(
            "no bounded functional probe demonstrated the complete requested execution tuple "
            f"({rendered})"
        )
    # full_finetune is admitted at planning WITHOUT an exact-combination proof - the tuple cannot be proven
    # until the full-parameter worker + wheel exist (refused at execution by required_runner_lane). For every
    # OTHER path a combination WAS found, so its probe-embedding + execution-contract checks still apply.
    if exact_combination is not None:
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
    # A preference plan's loss IS the DPO loss - fit-check that, not the SFT loss_impl, so the backend
    # capability evidence reflects the plan the resolver actually seals.
    fit_loss = (
        "grpo"
        if is_rollout
        else ("reward_bt" if is_reward_model else ("dpo" if is_preference_dpo else loss_impl))
    )
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
        loss=fit_loss,
        checkpoint=checkpoint_impl,
        export_format=constraints.export_format,
        execution_contract_version=_EXECUTION_CONTRACT_VERSION,
    )
    if is_preference_dpo:
        # The corpus_studio manifest cannot yet DECLARE the preference task OR the dpo loss: both are
        # content-hashed into backend_manifest_ref, which is sealed into the byte-locked SFT configuration,
        # so declaring either is coupled to a new manifest + milestone wheel. A preference+dpo_qlora plan is
        # therefore admitted at planning despite exactly those not-yet-declarable reasons (every OTHER
        # fit-check still applies), and the runner refuses it at execution until the wheel promotes it.
        not_yet_declarable = {
            f"task '{constraints.task_type}' not supported",
            f"loss '{fit_loss}' not supported",
        }
        unmet = [reason for reason in unmet if reason not in not_yet_declarable]
    if is_reward_model:
        # The manifest cannot yet DECLARE the reward task OR the reward_bt loss (both are content-hashed
        # into the byte-locked SFT config's backend_manifest_ref, so declaring either is coupled to a new
        # manifest + milestone wheel). A reward plan is admitted at planning despite exactly those
        # not-yet-declarable reasons (every OTHER fit-check applies); the runner refuses it at execution.
        not_yet_declarable = {
            f"task '{constraints.task_type}' not supported",
            f"loss '{fit_loss}' not supported",
        }
        unmet = [reason for reason in unmet if reason not in not_yet_declarable]
    if is_rollout:
        # Same for on-policy RL (RL slice S5b): the manifest cannot yet DECLARE the grpo task OR the grpo
        # loss (content-hashed into the byte-locked SFT config's backend_manifest_ref), so a rollout plan is
        # admitted at planning despite exactly those not-yet-declarable reasons - every OTHER fit-check
        # applies - and the runner refuses it at execution until the worker + wheel promote it.
        not_yet_declarable = {
            f"task '{constraints.task_type}' not supported",
            f"loss '{fit_loss}' not supported",
        }
        unmet = [reason for reason in unmet if reason not in not_yet_declarable]
    if is_full_finetune:
        # The manifest cannot yet DECLARE the full_finetune adapter method or a full-model export format
        # (declaring either is coupled to a new manifest + milestone wheel), so a full-parameter SFT plan is
        # admitted at planning despite exactly those not-yet-declarable reasons - every OTHER fit-check still
        # applies - and required_runner_lane refuses it at execution until the worker + wheel promote it.
        not_yet_declarable = {
            f"adapter '{adapter_method}' not supported",
            f"export format '{constraints.export_format}' not supported",
            f"checkpoint '{checkpoint_impl}' not supported",
        }
        unmet = [reason for reason in unmet if reason not in not_yet_declarable]
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
    # Intermediate checkpoint WRITING is wired for the adapter SFT lane (run_training drives the reviewed
    # CheckpointCoordinator); the full-parameter worker does not consume a cadence yet, so refuse it there
    # rather than seal a plan its runner would reject. keep_last is meaningless without a cadence.
    checkpoint_cadence = constraints.checkpoint_steps
    checkpoint_keep_last = constraints.checkpoint_keep_last
    if (checkpoint_cadence is not None or checkpoint_keep_last is not None) and is_full_finetune:
        raise PlannerError(
            "full-parameter fine-tuning cannot write intermediate checkpoints yet; drop "
            "--checkpoint-cadence (the coordinator wires the adapter SFT lane)"
        )
    # The reward (pairwise RM) and rollout (on-policy RL) workers have no CheckpointCoordinator, so a
    # sealed cadence would set save_strategy="steps" yet write nothing - a silent broken promise. Refuse it
    # fail-closed rather than seal an unkept guarantee (the coordinator wires only the adapter SFT lane).
    if (checkpoint_cadence is not None or checkpoint_keep_last is not None) and (
        is_reward_model or is_rollout
    ):
        lane = "reward-model" if is_reward_model else "on-policy RL"
        raise PlannerError(
            f"the {lane} lane cannot write intermediate checkpoints; drop --checkpoint-cadence "
            "(the CheckpointCoordinator wires only the adapter SFT lane)"
        )
    if checkpoint_cadence is None and checkpoint_keep_last is not None:
        raise PlannerError("--checkpoint-keep-last requires --checkpoint-cadence")
    checkpoint_policy = {
        "impl": checkpoint_impl,
        "cadence_optimizer_steps": checkpoint_cadence,
        "keep_last": checkpoint_keep_last,
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
    # The shared SFT TrainingDataPolicy renders instruction/chat/trace; a preference (DPO) plan's data is a
    # prompt/chosen/rejected pair sealed into its OWN PreferenceDataPolicy (in _resolve_preference_execution),
    # so the SFT policy is neither representable nor used for preference. Build it only for the paths that
    # consume it (SFT adapter + full-parameter SFT). The conformance preflight already validated the pairs.
    data_policy: TrainingDataPolicy | None = None
    if not is_preference_dpo and not is_reward_model and not is_rollout:
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
    objective = get_objective(
        "full_parameter_sft"
        if is_full_finetune
        else "qlora"
        if adapter_method == "qlora"
        else "lora"
    )
    if objective is None:  # pragma: no cover - sealed built-in catalog invariant
        raise PlannerError("the selected training objective is absent from the sealed registry")
    objective_ref = Ref(id=objective.objective_id, hash=HashRef(value=objective.objective_hash))

    root_device = resolved_physical.resources[0].device_id
    if root_device is None:
        raise PlannerError("the current dense trainer requires one explicit compute device")
    device_map = [
        DeviceMapEntry(module="", device="cpu" if root_device == "cpu:0" else root_device)
    ]
    # Shared execution sub-specs common to the SFT and preference(DPO) seals. Assembling them once keeps
    # the two resolvers consistent by construction; the SFT seal stays byte-identical because its
    # configuration_hash is over the validated MODEL (field values), not this input dict's shape.
    shared_fields: dict[str, Any] = {
        "configuration_id": f"{plan_id}-execution",
        "backend_ref": backend_manifest_ref(backend).model_dump(mode="json"),
        "environment_ref": resolved_environment_ref.model_dump(mode="json"),
        "environment_binding": (
            "managed_lock" if environment_ref is not None else "profile_snapshot"
        ),
        "capability_report_ref": capability_ref.model_dump(mode="json"),
        "inputs": execution_inputs.model_dump(mode="json"),
        "runtime_mode": "cpu_toy" if cpu_toy else "training",
        "precision": _precision_policy(precision, quantization, constraints.optim),
        "attention": attention_policy.model_dump(mode="json"),
        "device_map": [item.model_dump(mode="json") for item in device_map],
        "adapter": adapter,
        "optimizer": optimizer,
        "sequence": sequence,
        "batching": batching,
        "checkpoint_policy": checkpoint_policy,
        "schedule": schedule.model_dump(mode="json"),
        "trainer_interface": trainer_interface.model_dump(mode="json"),
        "export_format": constraints.export_format,
        "trust_remote_code": False,
        "use_safetensors": True,
        "bnb_4bit_use_double_quant": quantization != "none",
        "adapter_task_type": "CAUSAL_LM",
        # "steps" signals the sealed cadence to resolve_checkpoint_execution_policy; HF's own saver stays
        # off (build_training_kwargs forces it) so the CheckpointCoordinator owns exact-lineage writing.
        "save_strategy": "steps" if checkpoint_cadence is not None else "no",
        "gradient_checkpointing": True,
        "output_dir": constraints.output_dir,
        "output_layout": "run_scoped_v1",
        "seed": constraints.seed,
        "data_seed": constraints.data_seed if constraints.data_seed is not None else constraints.seed,
    }
    resolved_execution_field: dict[str, Any] | None = None
    resolved_preference_field: dict[str, Any] | None = None
    resolved_full_finetune_field: dict[str, Any] | None = None
    resolved_reward_field: dict[str, Any] | None = None
    resolved_rollout_field: dict[str, Any] | None = None
    if is_preference_dpo:
        preference_execution = _resolve_preference_execution(
            plan_id=plan_id,
            shared_fields=shared_fields,
            constraints=constraints,
            resolved_physical=resolved_physical,
            chat_template_sha256=constraints.chat_template_sha256,
            project_dir=project_dir,
        )
        resolved_preference_field = preference_execution.model_dump(mode="json")
    elif is_full_finetune:
        # The full-model sibling seal: the shared execution sub-specs already carry the full-parameter
        # shape (adapter.method=full_finetune, unquantized precision, full_state checkpoint, merged export);
        # only the SFT objective + loss + data policy are added, exactly like the dense SFT branch below.
        assert data_policy is not None  # built for every non-preference (SFT) path above
        try:
            full_finetune_draft = ResolvedFullFinetuneExecutionConfiguration.model_validate(
                {
                    **shared_fields,
                    "configuration_hash": "0" * 64,
                    "objective_ref": objective_ref.model_dump(mode="json"),
                    "loss_impl": loss_impl,
                    "data": data_policy.model_dump(mode="json"),
                }
            )
        except ValidationError as exc:
            raise PlannerError(
                f"the resolved full-finetune configuration is invalid: {exc}"
            ) from exc
        full_finetune_execution = full_finetune_draft.model_copy(
            update={
                "configuration_hash": full_finetune_execution_configuration_hash_for(
                    full_finetune_draft
                )
            }
        )
        resolved_full_finetune_field = full_finetune_execution.model_dump(mode="json")
    elif is_reward_model:
        # The reward sibling seal (RL slice S5a-1): the shared execution sub-specs carry the QLoRA shape;
        # the resolver reuses the preference chosen/rejected data policy + adds the RewardModelingSpec and
        # overrides the head (SEQ_CLS) + export family (reward_model).
        reward_execution = _resolve_reward_execution(
            plan_id=plan_id,
            shared_fields=shared_fields,
            constraints=constraints,
            resolved_physical=resolved_physical,
            chat_template_sha256=constraints.chat_template_sha256,
            project_dir=project_dir,
        )
        resolved_reward_field = reward_execution.model_dump(mode="json")
    elif is_rollout:
        # The on-policy RL sibling seal (RL slice S5b): the shared execution sub-specs carry the QLoRA
        # CAUSAL_LM policy shape; the resolver adds the rollout/experience/stability/policy specs + binds the
        # provenance-verified served reward source, and keeps the adapter_peft policy export.
        rollout_execution = _resolve_rollout_execution(
            plan_id=plan_id,
            shared_fields=shared_fields,
            constraints=constraints,
            resolved_physical=resolved_physical,
            chat_template_sha256=constraints.chat_template_sha256,
            project_dir=project_dir,
        )
        resolved_rollout_field = rollout_execution.model_dump(mode="json")
    else:
        assert data_policy is not None  # built for every non-preference (SFT) path above
        try:
            execution_draft = ResolvedExecutionConfiguration.model_validate(
                {
                    **shared_fields,
                    "configuration_hash": "0" * 64,
                    "objective_ref": objective_ref.model_dump(mode="json"),
                    "loss_impl": loss_impl,
                    "data": data_policy.model_dump(mode="json"),
                }
            )
        except ValidationError as exc:
            raise PlannerError(f"the resolved execution configuration is invalid: {exc}") from exc
        execution = execution_draft.model_copy(
            update={"configuration_hash": execution_configuration_hash_for(execution_draft)}
        )
        resolved_execution_field = execution.model_dump(mode="json")

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
        # A preference plan's loss summary IS the DPO loss (the supervised loss_impl would contradict the
        # preference seal the runner cross-checks); the SFT loss stays on the dense path.
        "loss_impl": (
            "grpo"
            if is_rollout
            else ("reward_bt" if is_reward_model else ("dpo" if is_preference_dpo else loss_impl))
        ),
        "attention_backend": attention_backend,
        "sequence": sequence,
        "batching": batching,
        "checkpoint_policy": checkpoint_policy,
        "offload_strategy": offload_strategy.value,
        "allocator_policy": constraints.allocator_policy,
        "allocator_max_split_size_mb": constraints.allocator_max_split_size_mb,
        "allocator_gc_threshold": constraints.allocator_gc_threshold,
        "gradient_checkpointing": True,
        "export": {"format": constraints.export_format, "output_dir": constraints.output_dir},
        "seed": constraints.seed,
        "training_config_snapshot": {},
        "resolved_execution": resolved_execution_field,
        "resolved_preference_execution": resolved_preference_field,
        "resolved_full_finetune_execution": resolved_full_finetune_field,
        "resolved_reward_execution": resolved_reward_field,
        "resolved_rollout_execution": resolved_rollout_field,
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
