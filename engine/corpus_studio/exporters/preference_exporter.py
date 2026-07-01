"""Export preference rows into trainer-ready formats.

Preference rows use the built-in ``preference`` schema (prompt, chosen,
rejected, optional reason). These helpers reshape them for common preference
trainers. Nothing here launches training; it only produces inspectable rows.
"""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel

from corpus_studio.quality.basic_quality import normalized_text_signature

PREFERENCE_EXPORT_FORMATS = ("dpo", "kto", "reward")
PREFERENCE_LOW_CONTRAST_THRESHOLD = 0.9


def _pair_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("prompt", "")),
        str(row.get("chosen", "")),
        str(row.get("rejected", "")),
    )


def to_dpo(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """TRL DPO / ORPO format: ``{prompt, chosen, rejected}``."""
    result = []
    for row in rows:
        prompt, chosen, rejected = _pair_fields(row)
        result.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return result


def to_kto(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """TRL KTO format: one row per completion with a binary preference label."""
    result = []
    for row in rows:
        prompt, chosen, rejected = _pair_fields(row)
        result.append({"prompt": prompt, "completion": chosen, "label": True})
        result.append({"prompt": prompt, "completion": rejected, "label": False})
    return result


def to_reward(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conversational pairwise reward-modeling format: ``{chosen: [...], rejected: [...]}``."""
    result = []
    for row in rows:
        prompt, chosen, rejected = _pair_fields(row)
        result.append(
            {
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected},
                ],
            }
        )
    return result


_EXPORTERS = {"dpo": to_dpo, "kto": to_kto, "reward": to_reward}


def export_preference(
    rows: Iterable[dict[str, Any]], target_format: str
) -> list[dict[str, Any]]:
    """Reshape preference rows for the requested trainer format."""
    normalized = target_format.strip().lower()
    exporter = _EXPORTERS.get(normalized)
    if exporter is None:
        supported = ", ".join(PREFERENCE_EXPORT_FORMATS)
        raise ValueError(
            f"Unsupported preference export format '{target_format}'. Use one of: {supported}."
        )
    return exporter(list(rows))


class PreferencePairIssues(BaseModel):
    """Integrity summary for preference (chosen/rejected) rows."""

    total: int = 0
    empty_chosen: int = 0
    empty_rejected: int = 0
    identical: int = 0
    low_contrast: int = 0
    degenerate: int = 0


def _pair_status(row: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    """Return (empty_chosen, empty_rejected, identical, low_contrast) for a pair."""
    _, chosen, rejected = _pair_fields(row)
    empty_chosen = not chosen.strip()
    empty_rejected = not rejected.strip()
    identical = False
    low_contrast = False
    if not empty_chosen and not empty_rejected:
        chosen_sig = normalized_text_signature(chosen)
        rejected_sig = normalized_text_signature(rejected)
        if chosen_sig and chosen_sig == rejected_sig:
            identical = True
        else:
            chosen_tokens = set(chosen_sig.split())
            rejected_tokens = set(rejected_sig.split())
            if chosen_tokens and rejected_tokens:
                overlap = len(chosen_tokens & rejected_tokens) / len(
                    chosen_tokens | rejected_tokens
                )
                low_contrast = overlap >= PREFERENCE_LOW_CONTRAST_THRESHOLD
    return empty_chosen, empty_rejected, identical, low_contrast


def _is_degenerate(row: dict[str, Any]) -> bool:
    empty_chosen, empty_rejected, identical, _ = _pair_status(row)
    return empty_chosen or empty_rejected or identical


def analyze_preference_pairs(rows: Iterable[dict[str, Any]]) -> PreferencePairIssues:
    """Count empty, identical, and low-contrast (weak) preference pairs.

    Identical or empty pairs are a zero-margin / contradictory training signal
    for DPO/KTO/reward models; low-contrast pairs are weak. Nothing is mutated.
    """
    issues = PreferencePairIssues()
    for row in rows:
        empty_chosen, empty_rejected, identical, low_contrast = _pair_status(row)
        issues.total += 1
        if empty_chosen:
            issues.empty_chosen += 1
        if empty_rejected:
            issues.empty_rejected += 1
        if identical:
            issues.identical += 1
        if low_contrast:
            issues.low_contrast += 1
        if empty_chosen or empty_rejected or identical:
            issues.degenerate += 1
    return issues


def drop_degenerate_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only usable preference pairs (drop empty or identical chosen/rejected)."""
    return [row for row in rows if not _is_degenerate(row)]
