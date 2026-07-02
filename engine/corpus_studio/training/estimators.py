"""Lightweight training estimate models.

Token budgets use the shared Unicode-aware token estimator (tiktoken when
installed, else a heuristic) instead of a flat characters/4 rule. Nothing here
inspects hardware or imports ML frameworks.
"""

import re
from typing import Any

from pydantic import BaseModel

from corpus_studio.tokenization.estimate import estimate_tokens, estimator_name


class TokenBudgetEstimate(BaseModel):
    """Approximate token budget for a dataset."""

    example_count: int = 0
    estimated_tokens: int = 0
    method: str = "heuristic"
    sequence_len: int = 0
    mean_tokens_per_example: float = 0.0
    max_tokens_in_example: int = 0
    examples_over_sequence_len: int = 0
    tokens_per_epoch: int = 0


class VramEstimate(BaseModel):
    """Rough, arithmetic VRAM planning estimate (never hardware-inspected)."""

    base_model: str
    adapter: str = "lora"
    parameter_count_billions: float | None = None
    weights_gb_fp16: float | None = None
    weights_gb_int8: float | None = None
    weights_gb_int4: float | None = None
    lora_overhead_gb: float | None = None
    activation_overhead_gb: float | None = None
    total_gb_fp16: float | None = None
    total_gb_int8: float | None = None
    total_gb_int4: float | None = None
    assumptions: list[str] = []
    note: str = ""


class LoraRecommendation(BaseModel):
    """Suggested LoRA rank/alpha for a model size, with sanity warnings."""

    recommended_r: int
    recommended_alpha: int
    warnings: list[str] = []


def _row_text(value: Any) -> str:
    parts: list[str] = []

    def _walk(item: Any) -> None:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                _walk(child)
        elif isinstance(item, list):
            for child in item:
                _walk(child)
        elif item is not None:
            parts.append(str(item))

    _walk(value)
    return " ".join(parts)


def estimate_token_budget(text_samples: list[str]) -> TokenBudgetEstimate:
    """Estimate tokens across text samples using the shared token estimator."""

    counts = [estimate_tokens(sample) for sample in text_samples]
    total = sum(counts)
    return TokenBudgetEstimate(
        example_count=len(text_samples),
        estimated_tokens=total,
        method=estimator_name(),
        mean_tokens_per_example=round(total / len(counts), 1) if counts else 0.0,
        max_tokens_in_example=max(counts) if counts else 0,
    )


def build_training_token_budget(rows: list[dict], sequence_len: int) -> TokenBudgetEstimate:
    """Full per-row token budget, including truncation against ``sequence_len``.

    ``tokens_per_epoch`` caps each row at ``sequence_len`` (what a trainer would
    actually process after truncation), and ``examples_over_sequence_len`` counts
    the rows that would be truncated.
    """

    counts = [estimate_tokens(_row_text(row)) for row in rows]
    if not counts:
        return TokenBudgetEstimate(sequence_len=sequence_len, method=estimator_name())

    total = sum(counts)
    return TokenBudgetEstimate(
        example_count=len(counts),
        estimated_tokens=total,
        method=estimator_name(),
        sequence_len=sequence_len,
        mean_tokens_per_example=round(total / len(counts), 1),
        max_tokens_in_example=max(counts),
        examples_over_sequence_len=sum(1 for count in counts if count > sequence_len),
        tokens_per_epoch=sum(min(count, sequence_len) for count in counts),
    )


# --- VRAM planning estimate --------------------------------------------------
# Pure arithmetic from the model name. Assumptions are listed on the estimate;
# nothing here inspects hardware or imports ML frameworks.

_PARAM_MOE_RE = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)
_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

# LoRA trainable fraction of base params at r=16 (all-linear targets, rough).
_LORA_FRACTION_AT_R16 = 0.006
# Bytes per trainable LoRA param: fp16 weight+grad plus fp32 AdamW states.
_LORA_BYTES_PER_PARAM = 16
# Rough activation memory for a 7B model at seq 4096, micro-batch 1, with
# gradient checkpointing enabled.
_ACTIVATION_BASE_GB = 1.5
# Fixed CUDA context / framework overhead.
_RUNTIME_OVERHEAD_GB = 1.0


def parse_parameter_count(base_model: str) -> float | None:
    """Parse a parameter count in billions from a model name, or None.

    Handles ``7B``, ``0.5b``, ``Qwen2.5-Coder-7B-Instruct``, ``llama-3-8b``,
    and MoE names like ``mixtral-8x7b`` (total parameters).
    """

    moe = _PARAM_MOE_RE.search(base_model)
    if moe:
        return round(float(moe.group(1)) * float(moe.group(2)), 3)

    # Take the last size-looking token so version fragments like "Qwen2.5" or
    # "llama-3" do not win over the real "7B"/"8b" suffix.
    matches = _PARAM_COUNT_RE.findall(base_model)
    if matches:
        return float(matches[-1])

    return None


def build_vram_estimate(
    base_model: str,
    lora_r: int = 16,
    sequence_len: int = 4096,
    micro_batch_size: int = 1,
    adapter: str = "lora",
) -> VramEstimate:
    """Rough VRAM planning estimate from the model name (no hardware access)."""

    params_b = parse_parameter_count(base_model)
    if params_b is None:
        return VramEstimate(
            base_model=base_model,
            adapter=adapter,
            note=(
                "Could not parse a parameter count from the model name, so no VRAM "
                "estimate is possible. Name the model with its size (e.g. '7B') or "
                "check the model card."
            ),
        )

    params = params_b * 1e9
    weights_fp16 = params * 2 / 1e9
    weights_int8 = params * 1 / 1e9
    weights_int4 = params * 0.5 / 1e9

    lora_params = params * _LORA_FRACTION_AT_R16 * (lora_r / 16)
    lora_overhead = lora_params * _LORA_BYTES_PER_PARAM / 1e9

    activation_overhead = (
        _ACTIVATION_BASE_GB
        * (params_b / 7)
        * (sequence_len / 4096)
        * max(1, micro_batch_size)
    )

    def _total(weights: float) -> float:
        return round(weights + lora_overhead + activation_overhead + _RUNTIME_OVERHEAD_GB, 1)

    return VramEstimate(
        base_model=base_model,
        adapter=adapter,
        parameter_count_billions=params_b,
        weights_gb_fp16=round(weights_fp16, 1),
        weights_gb_int8=round(weights_int8, 1),
        weights_gb_int4=round(weights_int4, 1),
        lora_overhead_gb=round(lora_overhead, 1),
        activation_overhead_gb=round(activation_overhead, 1),
        total_gb_fp16=_total(weights_fp16),
        total_gb_int8=_total(weights_int8),
        total_gb_int4=_total(weights_int4),
        assumptions=[
            f"Parameter count {params_b}B parsed from the model name.",
            "Weights: fp16 2 bytes/param, 8-bit 1, 4-bit 0.5.",
            f"LoRA overhead: ~{_LORA_FRACTION_AT_R16:.1%} of params at r=16, scaled by r={lora_r}, "
            f"{_LORA_BYTES_PER_PARAM} bytes/trainable param (weight+grad+AdamW states).",
            f"Activations assume gradient checkpointing, seq_len {sequence_len}, "
            f"micro-batch {micro_batch_size}.",
            f"+{_RUNTIME_OVERHEAD_GB:.0f} GB fixed runtime overhead.",
        ],
        note="Rough planning estimate only; real usage varies by architecture and trainer.",
    )


# LoRA rank conventions by model size (rough community practice).
_LORA_R_BY_SIZE = (
    (3.0, 8),
    (13.0, 16),
    (34.0, 32),
    (float("inf"), 64),
)


def recommend_lora(
    parameter_count_billions: float | None,
    lora_r: int,
    lora_alpha: int,
) -> LoraRecommendation:
    """Recommend LoRA rank/alpha for a model size and sanity-check the choice."""

    if parameter_count_billions is None:
        recommended_r = 16
    else:
        recommended_r = next(
            rank for limit, rank in _LORA_R_BY_SIZE if parameter_count_billions <= limit
        )

    recommended_alpha = recommended_r * 2
    warnings: list[str] = []

    if lora_r > recommended_r * 4:
        warnings.append(
            f"lora_r={lora_r} is unusually high for this model size "
            f"(typical: {recommended_r}); it increases memory and overfitting risk."
        )
    elif lora_r * 4 < recommended_r:
        warnings.append(
            f"lora_r={lora_r} is unusually low for this model size "
            f"(typical: {recommended_r}); the adapter may lack capacity."
        )

    if lora_alpha != lora_r * 2:
        warnings.append(
            f"lora_alpha={lora_alpha} deviates from the common alpha=2*r convention "
            f"({lora_r * 2} for r={lora_r})."
        )

    return LoraRecommendation(
        recommended_r=recommended_r,
        recommended_alpha=recommended_alpha,
        warnings=warnings,
    )
