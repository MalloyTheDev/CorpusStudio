"""Lightweight training estimate models.

These helpers provide safe placeholders for future UI planning. They do not
inspect hardware or import ML frameworks.
"""

from pydantic import BaseModel


class TokenBudgetEstimate(BaseModel):
    """Approximate token estimate for a dataset."""

    example_count: int
    estimated_tokens: int
    method: str = "chars_div_4_placeholder"


class VramEstimate(BaseModel):
    """Human-readable VRAM estimate placeholder."""

    base_model: str
    adapter: str
    note: str


def estimate_token_budget(text_samples: list[str]) -> TokenBudgetEstimate:
    """Estimate tokens using a conservative characters-divided-by-four rule."""

    char_count = sum(len(sample) for sample in text_samples)
    return TokenBudgetEstimate(
        example_count=len(text_samples),
        estimated_tokens=max(1, round(char_count / 4)) if text_samples else 0,
    )


def describe_vram_estimate(base_model: str, adapter: str = "lora") -> VramEstimate:
    """Return a placeholder VRAM note without claiming hardware precision."""

    return VramEstimate(
        base_model=base_model,
        adapter=adapter,
        note="Estimate requires model size, quantization, sequence length, and batch settings.",
    )

