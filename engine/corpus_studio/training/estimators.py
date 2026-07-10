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


# Chat-template overhead. Real chat templates wrap each turn (e.g.
# ``<|im_start|>role\n…<|im_end|>\n``) — a handful of special tokens per message —
# plus BOS/EOS bookends per conversation. Counting only the raw message *content*
# under-estimates chat/instruction rows and under-predicts truncation, so a chat
# row adds this conservative, model-agnostic overhead. It is a heuristic; exact
# per-model template rendering would need the model's ``tokenizer_config`` +
# ``transformers`` (a heavier optional follow-up).
_CHAT_TOKENS_PER_MESSAGE = 4
_CHAT_TOKENS_PER_CONVERSATION = 3


def _is_chat_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    messages = row.get("messages")
    return (
        isinstance(messages, list)
        and len(messages) > 0
        and all(isinstance(message, dict) and "content" in message for message in messages)
    )


def estimate_row_tokens(row: Any, model_id: str | None = None) -> int:
    """Token estimate for one training row, aware of chat structure.

    A chat row (a ``messages`` list of role/content turns) is counted as the sum of
    each turn's content tokens PLUS the per-message role/turn markers and the
    per-conversation BOS/EOS a chat template adds — so a chat budget doesn't
    under-count and under-predict truncation. Any other row falls back to the flat
    text extraction. ``model_id`` selects the model's tokenizer when available.
    """
    if _is_chat_row(row):
        messages = row["messages"]
        content_tokens = sum(
            estimate_tokens(str(message.get("content", "")), model_id) for message in messages
        )
        overhead = _CHAT_TOKENS_PER_MESSAGE * len(messages) + _CHAT_TOKENS_PER_CONVERSATION
        return content_tokens + overhead
    return estimate_tokens(_row_text(row), model_id)


def estimate_token_budget(
    text_samples: list[str], model_id: str | None = None
) -> TokenBudgetEstimate:
    """Estimate tokens across text samples using the shared token estimator. When
    ``model_id`` (a Hub id) is given and its tokenizer is available, the counts are
    exact for that model; otherwise it falls back to tiktoken / the heuristic."""

    counts = [estimate_tokens(sample, model_id) for sample in text_samples]
    total = sum(counts)
    return TokenBudgetEstimate(
        example_count=len(text_samples),
        estimated_tokens=total,
        method=estimator_name(model_id),
        mean_tokens_per_example=round(total / len(counts), 1) if counts else 0.0,
        max_tokens_in_example=max(counts) if counts else 0,
    )


def build_training_token_budget(
    rows: list[dict], sequence_len: int, model_id: str | None = None
) -> TokenBudgetEstimate:
    """Full per-row token budget, including truncation against ``sequence_len``.

    ``tokens_per_epoch`` caps each row at ``sequence_len`` (what a trainer would
    actually process after truncation), and ``examples_over_sequence_len`` counts
    the rows that would be truncated. When ``model_id`` is given and its tokenizer
    is available, the counts are exact for that target model.
    """

    counts = [estimate_row_tokens(row, model_id) for row in rows]
    if not counts:
        return TokenBudgetEstimate(sequence_len=sequence_len, method=estimator_name(model_id))

    total = sum(counts)
    return TokenBudgetEstimate(
        example_count=len(counts),
        estimated_tokens=total,
        method=estimator_name(model_id),
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
# Size token: digits (optional decimal) + 'b', with an optional trailing digit
# for names like BLOOM 'b7b1' (= 7.1B). The negative lookahead for a letter
# stops '8bit' (quantization) parsing as 8B.
_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(\d+)?(?![a-z])", re.IGNORECASE)

# LoRA trainable fraction of base params at r=16 (all-linear targets, rough).
_LORA_FRACTION_AT_R16 = 0.006
# Bytes per trainable LoRA param: fp16 weight+grad plus fp32 AdamW states.
_LORA_BYTES_PER_PARAM = 16
# Fixed CUDA context / framework overhead.
_RUNTIME_OVERHEAD_GB = 1.0
# Extra GB per billion params for a QUANTIZED (4-bit/8-bit) base: bitsandbytes keeps dequant/compute
# buffers + quant state beyond the raw packed weights, so the real footprint exceeds bytes×params.
# Calibrated so a 7B 4-bit QLoRA BASE (weights + LoRA + AdamW, no activations) lands at ~7.9 GB.
_QUANT_RUNTIME_GB_PER_B = 0.39
# Activation memory for a 7B at seq 4096, micro-batch 1, WITH gradient checkpointing — LINEAR in
# sequence_len (checkpointing makes the peak scale linearly, not seq²), scaled by params ÷7 and batch.
# CALIBRATED to a real Qwen2.5-7B 4-bit QLoRA memory sweep on an RTX 5070 (torch max_memory_allocated,
# which counts the WDDM shared-memory spill): peak = 7.91 GB base + ~2.85 GB per 1024 tokens →
# 10.83 GB @ seq1024, 13.76 @ seq2048. Two coefficients: the math/eager attention path (what Blackwell
# is forced onto) materializes far more than flash/mem-efficient attention (~5×).
_ACTIVATION_GB_MATH = 11.4  # math/eager SDPA, seq 4096, 7B (the measured path)
_ACTIVATION_GB_FLASH = 2.2  # flash/mem-efficient SDPA, seq 4096, 7B (memory-efficient; ~9 GB @ seq2048)


def parse_parameter_count(base_model: str) -> float | None:
    """Parse a parameter count in billions from a model name, or None.

    Handles ``7B``, ``0.5b``, ``Qwen2.5-Coder-7B-Instruct``, ``llama-3-8b``,
    MoE names like ``mixtral-8x7b`` (total parameters), ``A##B`` active-expert
    suffixes like ``Qwen3-30B-A3B`` (returns the 30B total), and BLOOM-style
    ``7b1`` (= 7.1B). Known limitation: underscore-decimal names such as
    ``stablelm-2-1_6b`` are not recognized (the ``_`` is ambiguous with a
    separator, e.g. ``llama_2_7b``).
    """

    moe = _PARAM_MOE_RE.search(base_model)
    if moe:
        return round(float(moe.group(1)) * float(moe.group(2)), 3)

    # Take the LARGEST size token. For 'Qwen3-30B-A3B' the total (30) must win
    # over the active-expert suffix (3); version fragments like 'Qwen2.5' never
    # match because they are not followed by 'b'.
    sizes: list[float] = []
    for whole, fraction in _PARAM_COUNT_RE.findall(base_model):
        value = float(whole)
        if fraction:  # e.g. '7' + 'b' + '1' -> 7.1
            value += int(fraction) / (10 ** len(fraction))
        sizes.append(round(value, 3))
    if sizes:
        return max(sizes)

    return None


def build_vram_estimate(
    base_model: str,
    lora_r: int = 16,
    sequence_len: int = 4096,
    micro_batch_size: int = 1,
    adapter: str = "lora",
    math_attention: bool = False,
) -> VramEstimate:
    """Rough VRAM planning estimate from the model name (no hardware access).

    ``math_attention=True`` adds the seq²-scaling attention-scores memory that the *math* (eager /
    math-SDPA) attention path materializes and flash/mem-efficient attention avoids. Blackwell GPUs
    (sm_120) MUST use the math path (the fused kernels deadlock), so on that arch the estimate is
    meaningfully higher — pass True there."""

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

    # Linear in sequence_len (gradient checkpointing makes the peak scale linearly, not seq²). The math
    # path (Blackwell is forced onto it) materializes far more than flash/mem-efficient attention.
    activation_base = _ACTIVATION_GB_MATH if math_attention else _ACTIVATION_GB_FLASH
    activation_overhead = (
        activation_base
        * (params_b / 7)
        * (sequence_len / 4096)
        * max(1, micro_batch_size)
    )

    quant_overhead = params_b * _QUANT_RUNTIME_GB_PER_B

    def _total(weights: float, quantized: bool) -> float:
        base = weights + lora_overhead + activation_overhead + _RUNTIME_OVERHEAD_GB
        return round(base + (quant_overhead if quantized else 0.0), 1)

    return VramEstimate(
        base_model=base_model,
        adapter=adapter,
        parameter_count_billions=params_b,
        weights_gb_fp16=round(weights_fp16, 1),
        weights_gb_int8=round(weights_int8, 1),
        weights_gb_int4=round(weights_int4, 1),
        lora_overhead_gb=round(lora_overhead, 1),
        activation_overhead_gb=round(activation_overhead, 1),
        total_gb_fp16=_total(weights_fp16, quantized=False),
        total_gb_int8=_total(weights_int8, quantized=True),
        total_gb_int4=_total(weights_int4, quantized=True),
        assumptions=[
            f"Parameter count {params_b}B parsed from the model name.",
            "Weights: fp16 2 bytes/param, 8-bit 1, 4-bit 0.5.",
            f"LoRA overhead: ~{_LORA_FRACTION_AT_R16:.1%} of params at r=16, scaled by r={lora_r}, "
            f"{_LORA_BYTES_PER_PARAM} bytes/trainable param (weight+grad+AdamW states).",
            f"Activations (~{activation_overhead:.1f} GB, {'math/eager' if math_attention else 'flash/mem-efficient'} "
            f"attention): gradient checkpointing, seq_len {sequence_len}, micro-batch {micro_batch_size} — LINEAR "
            "in seq_len; the math path (forced on Blackwell) uses ~5× more than flash.",
            f"Quantized paths add ~{quant_overhead:.1f} GB for bitsandbytes dequant/compute buffers.",
            f"+{_RUNTIME_OVERHEAD_GB:.0f} GB fixed runtime overhead.",
            "Calibrated to a real Qwen2.5-7B 4-bit QLoRA memory sweep (base ~7.9 GB + ~2.85 GB/1024 tokens).",
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
