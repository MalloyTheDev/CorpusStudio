"""Export preference rows into trainer-ready formats.

Preference rows use the built-in ``preference`` schema (prompt, chosen,
rejected, optional reason). These helpers reshape them for common preference
trainers. Nothing here launches training; it only produces inspectable rows.
"""

from __future__ import annotations

from typing import Any, Iterable

PREFERENCE_EXPORT_FORMATS = ("dpo", "kto", "reward")


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
