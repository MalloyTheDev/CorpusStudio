"""Read-only diff between two dataset versions (v1.0.2).

Compares the two versions' ordered row-id manifests as **multisets** (duplicate
rows count): a row present N times in one version and M in the other contributes
``|N - M|`` to added/removed. Because identity is the canonical exact signature,
a pure reordering or a key-order/whitespace-only change is NOT a content diff.

Reorder detection (#196): when the two versions hold the *same* multiset of rows
(nothing added, nothing removed) but the *ordered* manifests differ, the rows were
**reordered without any content change** — reported explicitly (``reordered`` +
``moved_count``) instead of showing a misleading "no changes", so a re-shuffle that a
sequence-sensitive trainer would notice isn't silently invisible.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field


class DatasetVersionDiff(BaseModel):
    base_version_id: str = ""
    other_version_id: str = ""
    base_row_count: int = 0
    other_row_count: int = 0
    added_count: int = 0
    removed_count: int = 0
    common_count: int = 0
    added_row_ids: list[str] = Field(default_factory=list)
    removed_row_ids: list[str] = Field(default_factory=list)
    # #196: the same rows in a different order (no content change).
    reordered: bool = False
    # Positions whose row-id differs between the two ordered manifests (0 unless reordered).
    moved_count: int = 0


def diff_manifests(
    base_ids: list[str],
    other_ids: list[str],
    base_version_id: str = "",
    other_version_id: str = "",
) -> DatasetVersionDiff:
    """Multiset diff of two ordered row-id manifests (base → other), flagging a pure reorder."""

    base = Counter(base_ids)
    other = Counter(other_ids)
    added = other - base  # multiset: extra copies present in other
    removed = base - other  # multiset: copies dropped from base
    common = base & other  # multiset intersection

    added_count = sum(added.values())
    removed_count = sum(removed.values())
    # A pure reorder: identical multisets (nothing added/removed) but a different order.
    reordered = added_count == 0 and removed_count == 0 and base_ids != other_ids
    moved_count = (
        sum(1 for base_id, other_id in zip(base_ids, other_ids) if base_id != other_id)
        if reordered
        else 0
    )

    return DatasetVersionDiff(
        base_version_id=base_version_id,
        other_version_id=other_version_id,
        base_row_count=len(base_ids),
        other_row_count=len(other_ids),
        added_count=added_count,
        removed_count=removed_count,
        common_count=sum(common.values()),
        added_row_ids=list(added.elements()),
        removed_row_ids=list(removed.elements()),
        reordered=reordered,
        moved_count=moved_count,
    )


def _safe(text: Any) -> str:
    collapsed = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text))
    return re.sub(r"\s+", " ", collapsed).strip()


def _compact(row: Any) -> str:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return text[:200] + ("…" if len(text) > 200 else "")


def render_dataset_version_diff_markdown(
    diff: DatasetVersionDiff,
    sample_added: list[Any] | None = None,
    sample_removed: list[Any] | None = None,
) -> str:
    lines = [
        f"# Dataset Version Diff — {_safe(diff.base_version_id)} → {_safe(diff.other_version_id)}",
        "",
        f"- **Base**: {diff.base_row_count} rows ({_safe(diff.base_version_id)})",
        f"- **Other**: {diff.other_row_count} rows ({_safe(diff.other_version_id)})",
        f"- **Added**: {diff.added_count}",
        f"- **Removed**: {diff.removed_count}",
        f"- **Common**: {diff.common_count}",
    ]
    if diff.reordered:
        lines += [
            "",
            f"> ⚠ **Reordered**: the same rows in a different order — no content changed, but "
            f"{diff.moved_count} position(s) moved. A sequence-sensitive trainer would see a "
            "different dataset.",
        ]
    lines += [
        "",
        "_Identity is the canonical row signature: a key-order/whitespace-only change is not "
        "counted. A pure reordering shows Added/Removed = 0 and is flagged as **Reordered** above._",
    ]
    if sample_added:
        lines += ["", "## Added (sample)", ""]
        lines += [f"- {_safe(_compact(row))}" for row in sample_added]
    if sample_removed:
        lines += ["", "## Removed (sample)", ""]
        lines += [f"- {_safe(_compact(row))}" for row in sample_removed]
    return "\n".join(lines)
