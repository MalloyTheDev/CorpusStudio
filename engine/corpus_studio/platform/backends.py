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

from corpus_studio.platform.contracts import BackendManifest

# ------------------------------------------------------------------------------------------------
# Built-in backend manifests. Constructed via model_validate so the enum-list fields coerce from
# plain strings (mirrors the demo plans; keeps mypy happy without importing every enum).
# ------------------------------------------------------------------------------------------------
_CORPUS_STUDIO = {
    "backend_id": "corpus_studio",
    "display_name": "Corpus Studio (first-party · HF + TRL + PEFT)",
    "backend_version": "1.0.0",
    "trainer_target": "corpus_studio",
    "supported_os": ["windows", "linux", "macos"],
    "supported_devices": ["cuda", "cpu"],
    "task_types": ["sft"],
    "precision_modes": ["bf16", "fp16", "fp32"],
    "quantization_modes": ["none", "nf4", "int8"],
    "adapter_methods": ["lora", "qlora"],
    # math/eager/sdpa cover the Blackwell-safe path; flash_attention_2 only where the GPU allows it.
    "attention_impls": ["math", "eager", "sdpa", "flash_attention_2"],
    "loss_impls": ["cross_entropy", "liger_fused_ce"],
    "checkpoint_impls": ["adapter_only", "safetensors"],
    "optimizers": ["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "adamw_bnb_8bit"],
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
            "condition": "compute_capability_major>=12 with fused flash attention",
            "description": "The fused flash/mem-efficient SDPA kernels deadlock on Blackwell (sm_120).",
            "mitigation": "Use the math attention path (the planner forces it on sm_120).",
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
    "supported_os": ["linux", "windows"],
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
    "checkpoint_impls": ["adapter_only", "safetensors"],
    "optimizers": ["adamw_8bit", "paged_adamw_8bit"],
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
            "condition": "compute_capability_major>=12 (Blackwell / sm_120)",
            "description": "Unsloth's fused kernels are flash/sdpa; Blackwell needs the math path.",
            "mitigation": "Use the corpus_studio backend on Blackwell.",
        }
    ],
    "capability_probes": ["cuda_available", "bf16_matmul", "bnb_4bit_load"],
}

_BUILTIN = tuple(BackendManifest.model_validate(m) for m in (_CORPUS_STUDIO, _UNSLOTH))


def builtin_backends() -> list[BackendManifest]:
    """Every registered training backend."""
    return list(_BUILTIN)


def get_backend(backend_id: str) -> BackendManifest | None:
    """The manifest for ``backend_id``, or ``None`` if unknown."""
    return next((b for b in _BUILTIN if b.backend_id == backend_id), None)


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
