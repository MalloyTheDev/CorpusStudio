"""Read-only diff between two dataset versions (v1.0.2).

Compares the two versions' ordered row-id manifests as **multisets** (duplicate
rows count): a row present N times in one version and M in the other contributes
``|N - M|`` to added/removed. Because identity is the canonical exact signature,
a pure reordering or a key-order/whitespace-only change is NOT a diff.
Reorder/"moved" detection is deferred.
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


def diff_manifests(
    base_ids: list[str],
    other_ids: list[str],
    base_version_id: str = "",
    other_version_id: str = "",
) -> DatasetVersionDiff:
    """Multiset diff of two ordered row-id manifests (base → other)."""

    base = Counter(base_ids)
    other = Counter(other_ids)
    added = other - base  # multiset: extra copies present in other
    removed = base - other  # multiset: copies dropped from base
    common = base & other  # multiset intersection
    return DatasetVersionDiff(
        base_version_id=base_version_id,
        other_version_id=other_version_id,
        base_row_count=len(base_ids),
        other_row_count=len(other_ids),
        added_count=sum(added.values()),
        removed_count=sum(removed.values()),
        common_count=sum(common.values()),
        added_row_ids=list(added.elements()),
        removed_row_ids=list(removed.elements()),
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
        "",
        "_Identity is the canonical row signature: a pure reordering or a "
        "key-order/whitespace-only change is not counted, and moved-row detection "
        "is not included in this view._",
    ]
    if sample_added:
        lines += ["", "## Added (sample)", ""]
        lines += [f"- {_safe(_compact(row))}" for row in sample_added]
    if sample_removed:
        lines += ["", "## Removed (sample)", ""]
        lines += [f"- {_safe(_compact(row))}" for row in sample_removed]
    return "\n".join(lines)
