"""The fit calibrator — platform slice 8.

The planner (slice 6) resolves a VALID plan but deliberately does not assert it FITS. This estimates
a :class:`RunPlan`'s peak VRAM against the target device and returns a PREDICTED
:class:`FitClassification` — the honest prediction, never a proven fit. Only a measured run earns
``NATIVE_SAFE``; an estimate that fits is ``NATIVE_UNPROVEN``.

Crucially it distinguishes a Windows/WDDM **silent spill** (pages to shared memory, 10-25× slowdown)
from a hard **OOM** via the host's memory-residency model — the platform's key insight, so a run is
never launched into a paging thrash mislabeled "it fits".

Reuses the VRAM arithmetic in ``training.estimators`` (pure, torch-free; calibrated to a real
Qwen2.5-7B 4-bit QLoRA memory sweep) via a lazy import, so this module stays dependency-light.
"""

from __future__ import annotations

from corpus_studio.platform.contracts import EnvironmentProfile, FitClassification, RunPlan
from corpus_studio.platform.enums import FitClass, MemoryResidencyModel

# The engine's VRAM headroom margin (preflight._VRAM_SAFETY_MARGIN_GB) — within it, a run is "close to
# the edge" and treated as MARGINAL rather than a predicted-safe fit.
_SAFETY_MARGIN_BYTES = 1_500_000_000
_GB = 1_000_000_000
# Attention paths that materialize the seq-scaling attention-scores memory (Blackwell is forced here).
_MATH_ATTENTION = frozenset({"math", "eager"})
_INT4_QUANT = frozenset({"nf4", "int4", "fp4"})


def classify_fit(plan: RunPlan, profile: EnvironmentProfile) -> FitClassification:
    """Predict whether ``plan`` fits ``profile``'s GPU. Returns a :class:`FitClassification` whose
    ``classification`` is at best ``NATIVE_UNPROVEN`` (predicted to fit, not measured), ``MARGINAL``
    (within the safety margin), or a predicted spill/OOM keyed to the residency model. Never
    ``NATIVE_SAFE`` — that requires a measured run."""
    from corpus_studio.training.estimators import build_vram_estimate  # noqa: PLC0415 - torch-free

    # A cpu-toy plan runs on CPU — GPU VRAM fit is not applicable.
    if plan.training_config_snapshot.get("cpu_toy"):
        return FitClassification(
            classification=FitClass.NATIVE_UNPROVEN,
            rationale="cpu-toy plan runs on CPU; GPU VRAM fit is not applicable.",
        )

    capacity = max(
        (g.vram_total_bytes for g in profile.gpus if g.vram_total_bytes is not None), default=None
    )
    if capacity is None:
        return FitClassification(
            classification=FitClass.NATIVE_UNPROVEN,
            rationale="no GPU VRAM capacity to assess against (CPU host, or the probe couldn't read "
            "VRAM) — fit not estimated.",
        )

    math_attention = plan.attention_backend.value in _MATH_ATTENTION
    estimate = build_vram_estimate(
        base_model=plan.base_model,
        lora_r=plan.adapter.lora_r or 16,
        sequence_len=plan.sequence.max_sequence_len,
        micro_batch_size=plan.batching.micro_batch_size,
        adapter=plan.adapter.method.value,
        math_attention=math_attention,
    )
    total_gb = _select_total_gb(estimate, plan.quantization.value)
    if total_gb is None:
        return FitClassification(
            classification=FitClass.NATIVE_UNPROVEN,
            device_capacity_bytes=capacity,
            attention_path=plan.attention_backend,
            rationale=estimate.note or "could not estimate peak memory from the model name — fit unproven.",
        )

    peak = int(total_gb * _GB)
    headroom = capacity - peak
    classification, rationale = _band(headroom, peak, capacity, profile.host.memory_residency_model)
    return FitClassification(
        classification=classification,
        estimated_peak_bytes=peak,
        device_capacity_bytes=capacity,
        headroom_bytes=headroom,
        attention_path=plan.attention_backend,
        rationale=rationale,
    )


def _select_total_gb(estimate: object, quantization: str) -> float | None:
    if quantization in _INT4_QUANT:
        return getattr(estimate, "total_gb_int4", None)
    if quantization == "int8":
        return getattr(estimate, "total_gb_int8", None)
    return getattr(estimate, "total_gb_fp16", None)


def _band(
    headroom: int, peak: int, capacity: int, residency: MemoryResidencyModel
) -> tuple[FitClass, str]:
    peak_gb = peak / _GB
    cap_gb = capacity / _GB
    if headroom >= _SAFETY_MARGIN_BYTES:
        return (
            FitClass.NATIVE_UNPROVEN,
            f"estimated peak ~{peak_gb:.1f} GB fits within {cap_gb:.1f} GB with ~{headroom / _GB:.1f} "
            "GB headroom — predicted to fit, NOT measured.",
        )
    if headroom >= 0:
        return (
            FitClass.MARGINAL,
            f"estimated peak ~{peak_gb:.1f} GB is within the 1.5 GB safety margin of {cap_gb:.1f} GB — "
            "likely to spill/OOM; measure before trusting.",
        )

    over_gb = (peak - capacity) / _GB
    head = f"estimated peak ~{peak_gb:.1f} GB exceeds {cap_gb:.1f} GB by ~{over_gb:.1f} GB"
    if residency == MemoryResidencyModel.wddm:
        return (
            FitClass.ACCIDENTAL_WDDM_SPILL,
            head + " — on Windows/WDDM this silently spills to shared memory (10-25× slowdown), not a "
            "clean OOM. Reduce sequence_len / micro_batch_size, or offload.",
        )
    if residency == MemoryResidencyModel.unified_memory:
        return (
            FitClass.ACCIDENTAL_UNIFIED_MEMORY_PAGING,
            head + " — on unified memory this pages to system RAM (severe slowdown). Reduce "
            "sequence_len / micro_batch_size.",
        )
    return (
        FitClass.FAIL,
        head + " — this will hard-OOM. Reduce sequence_len / micro_batch_size, quantize, or use a "
        "smaller base model.",
    )
