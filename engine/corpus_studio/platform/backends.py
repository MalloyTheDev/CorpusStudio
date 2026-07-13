"""The training-backend registry — the "pick your framework" substrate.

A :class:`BackendManifest` is a backend's STATIC self-declaration of everything it can do. The core
reads the registry to decide which backends can even ATTEMPT a resolved plan on a given host, BEFORE
dispatch — resolving the plan's requirements against each backend's *declared* surface intersected
with what actually PROVED to work on the host (``effective_capabilities``). "No backend supported
until the probes pass."

Built-in backends today: the first-party ``corpus_studio`` trainer (HF + TRL + PEFT, LoRA/QLoRA) and
``unsloth`` (the accelerated QLoRA path). Both are declared honestly — e.g. Unsloth's fused kernels
are flash/sdpa, so on Blackwell (sm_120, which needs the ``math`` path) Unsloth is correctly filtered
OUT and the plan routes to the first-party math-path trainer. Adding a backend is adding a manifest.

Dependency-light: pure contracts + stdlib, no torch.
"""

from __future__ import annotations

import hashlib
import json

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    BackendManifest,
    EffectiveCapabilities,
    PhysicalExecutionSpec,
)
from corpus_studio.platform.enums import OffloadStrategy, PlacementMode, PlacementRole

# ------------------------------------------------------------------------------------------------
# Built-in backend manifests. Constructed via model_validate so the enum-list fields coerce from
# plain strings (mirrors the demo plans; keeps mypy happy without importing every enum).
# ------------------------------------------------------------------------------------------------
_CORPUS_STUDIO = {
    "backend_id": "corpus_studio",
    "display_name": "Corpus Studio (first-party · HF + TRL + PEFT)",
    "backend_version": "1.0.0",
    "trainer_target": "corpus_studio",
    "supported_os": ["windows", "wsl", "linux", "macos"],
    "supported_devices": ["cuda", "cpu"],
    "task_types": ["sft"],
    "precision_modes": ["bf16", "fp16", "fp32"],
    "quantization_modes": ["none", "nf4", "int8"],
    "adapter_methods": ["lora", "qlora"],
    # math/eager cover the known WDDM-safe path; sdpa/flash still require host capability evidence.
    "attention_impls": ["math", "eager", "sdpa", "flash_attention_2"],
    "loss_impls": ["cross_entropy", "liger_fused_ce"],
    # Declarations only. CapabilityReport intentionally leaves these empty until an end-to-end
    # objective probe proves them in the selected environment.
    "objective_capabilities": ["adapter_lora", "adapter_qlora", "causal_lm_sft"],
    "checkpoint_impls": ["adapter_only", "safetensors"],
    "optimizers": ["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "adamw_bnb_8bit"],
    "placement_modes": ["single_resource"],
    "placement_tiers": ["gpu"],
    "export_formats": ["adapter_peft"],
    "dependency_requirements": [
        {"name": "torch"},
        {"name": "transformers"},
        {"name": "peft"},
        {"name": "trl"},
        {"name": "datasets"},
        {"name": "accelerate"},
        {"name": "bitsandbytes", "optional": True, "reason": "only for 4-bit/8-bit quantization"},
    ],
    "known_failure_modes": [
        {
            "taxonomy": "KERNEL_STALL",
            "condition": "native Windows (WDDM) + compute_capability_major>=12 with fused flash attention",
            "description": "The fused flash SDPA backward deadlocks on Blackwell (sm_120) under the "
            "Windows WDDM driver. WSL has separate passing evidence; bare-Linux behavior remains "
            "unverified until probed on the final host.",
            "mitigation": "On native Windows the planner forces the math attention path on sm_120; "
            "outside WDDM, require a passing flash-attention capability probe before selecting flash.",
        }
    ],
    "capability_probes": ["cuda_available", "bf16_matmul", "bnb_4bit_load", "flash_attn_backward"],
}

_UNSLOTH = {
    "backend_id": "unsloth",
    "display_name": "Unsloth (accelerated QLoRA · ~2× faster, less VRAM)",
    "backend_version": "1.0.0",
    "trainer_target": "unsloth_script",
    # Unsloth is CUDA/triton-focused — no CPU training path.
    "supported_os": ["linux", "wsl", "windows"],
    "supported_devices": ["cuda"],
    "required_compute_capability": ">=7.5",
    "task_types": ["sft"],
    "precision_modes": ["bf16", "fp16"],
    "quantization_modes": ["nf4", "int4", "none"],
    "adapter_methods": ["lora", "qlora"],
    # Unsloth's kernels are flash/sdpa — it does NOT provide the math path, so a Blackwell plan (which
    # requires math) will not select Unsloth. This is declared honestly, not hidden.
    "attention_impls": ["flash_attention_2", "sdpa"],
    "loss_impls": ["cross_entropy"],
    "objective_capabilities": ["adapter_lora", "adapter_qlora", "causal_lm_sft"],
    "checkpoint_impls": ["adapter_only", "safetensors"],
    "optimizers": ["adamw_8bit", "paged_adamw_8bit"],
    "placement_modes": ["single_resource"],
    "placement_tiers": ["gpu"],
    "export_formats": ["adapter_peft", "merged_safetensors", "merged_fp16", "gguf"],
    "dependency_requirements": [
        {"name": "unsloth"},
        {"name": "torch"},
        {"name": "transformers"},
        {"name": "trl"},
        {"name": "peft"},
        {"name": "datasets"},
        {"name": "bitsandbytes"},
    ],
    "known_failure_modes": [
        {
            "taxonomy": "UNSUPPORTED_CONFIGURATION",
            "condition": "native Windows (WDDM) + compute_capability_major>=12 (Blackwell / sm_120)",
            "description": "Unsloth's fused kernels are flash/sdpa; native-Windows Blackwell needs the "
            "math path (WDDM flash deadlock), which Unsloth lacks. WSL has separate SDPA evidence; "
            "bare-Linux capability remains unverified.",
            "mitigation": "Use the corpus_studio backend on native-Windows Blackwell. On another "
            "platform, select Unsloth only after its environment capability probes pass.",
        }
    ],
    "capability_probes": ["cuda_available", "bf16_matmul", "bnb_4bit_load"],
}

_ECHO = {
    "backend_id": "echo",
    "display_name": "CorpusStudio protocol echo worker",
    "backend_version": "1.0.0",
    "supported_os": ["windows", "wsl", "linux", "macos"],
    "supported_devices": ["cpu"],
    "task_types": ["evaluation"],
    "attention_impls": ["math"],
    "loss_impls": ["cross_entropy"],
    "checkpoint_impls": ["adapter_only"],
    "optimizers": ["adamw_torch"],
    "placement_modes": ["single_resource"],
    "placement_tiers": ["pageable_ram"],
    "export_formats": ["adapter_peft"],
}

_BUILTIN = tuple(BackendManifest.model_validate(m) for m in (_CORPUS_STUDIO, _UNSLOTH))
_ECHO_BACKEND = BackendManifest.model_validate(_ECHO)


def builtin_backends() -> list[BackendManifest]:
    """Every registered training backend."""
    return list(_BUILTIN)


def get_backend(backend_id: str) -> BackendManifest | None:
    """The manifest for ``backend_id``, or ``None`` if unknown."""
    return next((b for b in _BUILTIN if b.backend_id == backend_id), None)


def get_worker_backend(backend_id: str) -> BackendManifest | None:
    """A backend manifest a worker can present during the protocol handshake.

    ``echo`` is deliberately protocol-only and is not returned by :func:`builtin_backends`, so it
    can exercise the supervisor without appearing as a selectable training backend.
    """

    if backend_id == _ECHO_BACKEND.backend_id:
        return _ECHO_BACKEND
    return get_backend(backend_id)


def backend_manifest_digest(manifest: BackendManifest) -> str:
    """Stable content identity for a backend's static declaration."""

    payload = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def backend_manifest_ref(manifest: BackendManifest) -> Ref:
    """Hash-pin a RunPlan to the exact static backend declaration it was planned against."""

    return Ref(id=manifest.backend_id, hash=HashRef(value=backend_manifest_digest(manifest)))


def unmet_requirements(
    manifest: BackendManifest,
    *,
    os: str,
    device: str,
    task_type: str,
    precision: str,
    quantization: str,
    adapter_method: str,
    attention: str,
) -> list[str]:
    """The reasons ``manifest`` cannot run the given plan requirements — empty when it can. Each check
    is against what the backend DECLARES it supports (the caller resolves those against the proven
    host capabilities separately)."""
    reasons: list[str] = []
    values = {m.value for m in manifest.supported_os}
    if os not in values:
        reasons.append(f"OS '{os}' not in {sorted(values)}")
    if device not in {d.value for d in manifest.supported_devices}:
        reasons.append(f"device '{device}' not supported")
    if task_type not in {t.value for t in manifest.task_types}:
        reasons.append(f"task '{task_type}' not supported")
    if precision not in {p.value for p in manifest.precision_modes}:
        reasons.append(f"precision '{precision}' not supported")
    if quantization not in {q.value for q in manifest.quantization_modes}:
        reasons.append(f"quantization '{quantization}' not supported")
    if adapter_method not in {a.value for a in manifest.adapter_methods}:
        reasons.append(f"adapter '{adapter_method}' not supported")
    if attention not in {a.value for a in manifest.attention_impls}:
        reasons.append(f"attention '{attention}' not supported")
    return reasons


def compatible_backends(
    *,
    os: str,
    device: str,
    task_type: str,
    precision: str,
    quantization: str,
    adapter_method: str,
    attention: str,
) -> list[BackendManifest]:
    """Every registered backend that DECLARES support for the given plan requirements."""
    return [
        b
        for b in _BUILTIN
        if not unmet_requirements(
            b,
            os=os,
            device=device,
            task_type=task_type,
            precision=precision,
            quantization=quantization,
            adapter_method=adapter_method,
            attention=attention,
        )
    ]


def unmet_physical_requirements(
    manifest: BackendManifest,
    effective: EffectiveCapabilities | None,
    spec: PhysicalExecutionSpec,
    *,
    offload_strategy: OffloadStrategy,
) -> list[str]:
    """Return every declared/proven capability missing for a non-trivial physical plan.

    The ordinary singleton plan is checked by :func:`unmet_requirements`; callers should invoke this
    only for a custom/offloaded/distributed spec. Static declaration and functional proof are both
    required, so an installed framework never becomes "supported" merely by naming a feature.
    """

    requested_tiers = {item.tier for item in spec.resources}
    requested_modes: set[PlacementMode] = set()
    if len(spec.resources) == 1:
        requested_modes.add(PlacementMode.single_resource)
    if len({item.tier for item in spec.resources}) > 1:
        requested_modes.add(PlacementMode.tiered)
    if spec.requires_parameter_accounting():
        requested_modes.add(PlacementMode.identity_scoped)
    if any(item.role == PlacementRole.replica for item in spec.placements):
        requested_modes.add(PlacementMode.replicated)
    if any(item.role == PlacementRole.shard for item in spec.placements):
        requested_modes.add(PlacementMode.sharded)
    if any(item.selector.expert_ids for item in spec.placements) or any(
        item.selector.expert_ids for item in spec.offload_rules
    ):
        requested_modes.add(PlacementMode.expert_scoped)
    requested_parallelism = {item.kind for item in spec.parallelism.groups}
    requested_communication = {
        item.communication_backend for item in spec.parallelism.groups
    }
    requested_offload = (
        {offload_strategy} if offload_strategy != OffloadStrategy.none else set()
    )

    declared: dict[str, set[str]] = {
        "placement tier": {item.value for item in manifest.placement_tiers},
        "placement mode": {item.value for item in manifest.placement_modes},
        "parallelism kind": {item.value for item in manifest.parallelism_kinds},
        "communication backend": {item.value for item in manifest.communication_backends},
        "offload strategy": {item.value for item in manifest.offload_strategies},
    }
    proven: dict[str, set[str]] = {
        "placement tier": {item.value for item in effective.placement_tiers}
        if effective
        else set(),
        "placement mode": {item.value for item in effective.placement_modes}
        if effective
        else set(),
        "parallelism kind": {item.value for item in effective.parallelism_kinds}
        if effective
        else set(),
        "communication backend": {item.value for item in effective.communication_backends}
        if effective
        else set(),
        "offload strategy": {item.value for item in effective.offload_strategies}
        if effective
        else set(),
    }
    requested: dict[str, set[str]] = {
        "placement tier": {item.value for item in requested_tiers},
        "placement mode": {item.value for item in requested_modes},
        "parallelism kind": {item.value for item in requested_parallelism},
        "communication backend": {item.value for item in requested_communication},
        "offload strategy": {item.value for item in requested_offload},
    }
    reasons: list[str] = []
    for label, tokens in requested.items():
        for token in sorted(tokens):
            if token not in declared[label]:
                reasons.append(f"{label} '{token}' is not declared by the backend")
            elif token not in proven[label]:
                reasons.append(f"{label} '{token}' is not functionally verified")
    return reasons
