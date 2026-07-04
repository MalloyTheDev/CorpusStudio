"""Detect train/validation/test leakage after splitting.

`random_split` partitions distinct row objects, but if the input contains exact
or near-duplicate rows, copies can land in different splits — contaminating the
test set and silently inflating evaluation scores. This module reports (but does
not mutate) such cross-split collisions using the same NFKC/Unicode-aware
normalization the quality report uses, so a whitespace/case/punctuation variant
is caught as leakage, not just byte-identical rows.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from corpus_studio.quality.basic_quality import normalized_text_signature

SPLIT_LEAKAGE_SAMPLE_LIMIT = 20


class SplitLeakage(BaseModel):
    """One group of duplicate/near-duplicate rows shared across ≥2 splits."""

    splits: list[str]
    row_count: int
    exact: bool
    sample: str


class SplitLeakageReport(BaseModel):
    leaked_group_count: int = 0
    rows_shared_across_splits: int = 0
    leaks: list[SplitLeakage] = Field(default_factory=list)


def _exact_signature(row: Any) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def _readable_sample(row: Any) -> str:
    """A readable one-line rendering of an original leaked row (compact JSON, Unicode
    preserved) so a leak report shows the real text, not its normalized signature."""

    try:
        return json.dumps(row, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(row)


def detect_split_leakage(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> SplitLeakageReport:
    named_splits = (("train", train), ("validation", validation), ("test", test))
    groups: dict[str, dict[str, Any]] = {}

    for split_name, rows in named_splits:
        for row in rows:
            signature = normalized_text_signature(row)
            if not signature:
                continue
            # Keep a reference to the first original row for a readable sample; render it
            # only for groups that actually leak (below), not once per row.
            group = groups.setdefault(
                signature, {"splits": {}, "exact_signatures": set(), "sample_row": row}
            )
            group["splits"][split_name] = group["splits"].get(split_name, 0) + 1
            group["exact_signatures"].add(_exact_signature(row))

    leaks: list[SplitLeakage] = []
    rows_shared = 0
    for group in groups.values():
        if len(group["splits"]) < 2:
            continue
        total = sum(group["splits"].values())
        # Count only the copies that cross a split boundary — everything beyond the split
        # that holds the most copies. Summing `total` double-counts within-split duplicates,
        # which are not leakage (the largest split is the group's "home").
        rows_shared += total - max(group["splits"].values())
        leaks.append(
            SplitLeakage(
                splits=sorted(group["splits"].keys()),
                row_count=total,
                exact=len(group["exact_signatures"]) == 1,
                sample=_readable_sample(group["sample_row"])[:120],
            )
        )

    leaks.sort(key=lambda leak: (-leak.row_count, leak.sample))
    return SplitLeakageReport(
        leaked_group_count=len(leaks),
        rows_shared_across_splits=rows_shared,
        leaks=leaks[:SPLIT_LEAKAGE_SAMPLE_LIMIT],
    )
