"""Lightweight training estimate models.

Token budgets use the shared Unicode-aware token estimator (tiktoken when
installed, else a heuristic) instead of a flat characters/4 rule. Nothing here
inspects hardware or imports ML frameworks.
"""

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
    """Human-readable VRAM estimate placeholder."""

    base_model: str
    adapter: str
    note: str


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


def describe_vram_estimate(base_model: str, adapter: str = "lora") -> VramEstimate:
    """Return a placeholder VRAM note without claiming hardware precision."""

    return VramEstimate(
        base_model=base_model,
        adapter=adapter,
        note="Estimate requires model size, quantization, sequence length, and batch settings.",
    )
