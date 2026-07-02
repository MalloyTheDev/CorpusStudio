"""Optional cleaning pass for JSONL export.

Plain export is a byte-for-byte copy. When cleaning is requested this module
drops exact and normalized-duplicate rows and (optionally) low-information rows,
and records a manifest of exactly what was removed so the cleaned export stays
reproducible and inspectable. It never edits row content — it only drops whole
rows, using the same NFKC/Unicode-aware normalization as the quality report.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from corpus_studio.quality.basic_quality import (
    LOW_INFORMATION_TOKEN_THRESHOLD,
    normalized_text_signature,
)

CLEAN_MANIFEST_SAMPLE_LIMIT = 100


class RemovedRow(BaseModel):
    row_number: int
    reason: str  # exact_duplicate | normalized_duplicate | low_information
    sample: str


class CleanResult(BaseModel):
    input_rows: int = 0
    kept_rows: int = 0
    removed_rows: int = 0
    removed_exact_duplicates: int = 0
    removed_normalized_duplicates: int = 0
    removed_low_information: int = 0
    removed: list[RemovedRow] = Field(default_factory=list)


def _exact_signature(row: Any) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


# Public alias for the canonical per-row exact signature. Reused verbatim by the
# dataset version fingerprint (versions/version_registry.py) so row identity is
# computed exactly one way across cleaning, quality, leakage, and versioning.
exact_row_signature = _exact_signature


def clean_rows(
    rows: list[dict[str, Any]],
    *,
    dedupe: bool = False,
    drop_low_information: bool = False,
    low_information_threshold: int = LOW_INFORMATION_TOKEN_THRESHOLD,
) -> tuple[list[dict[str, Any]], CleanResult]:
    kept: list[dict[str, Any]] = []
    removed: list[RemovedRow] = []
    exact_seen: set[str] = set()
    normalized_seen: set[str] = set()
    exact_count = normalized_count = low_information_count = 0

    for row_number, row in enumerate(rows, start=1):
        reason: str | None = None

        if dedupe:
            exact_signature = _exact_signature(row)
            if exact_signature in exact_seen:
                reason = "exact_duplicate"
                exact_count += 1
            else:
                exact_seen.add(exact_signature)
                normalized_signature = normalized_text_signature(row)
                if normalized_signature and normalized_signature in normalized_seen:
                    reason = "normalized_duplicate"
                    normalized_count += 1
                elif normalized_signature:
                    normalized_seen.add(normalized_signature)

        if reason is None and drop_low_information:
            token_count = len(normalized_text_signature(row).split())
            if 0 < token_count < low_information_threshold:
                reason = "low_information"
                low_information_count += 1

        if reason is None:
            kept.append(row)
        else:
            removed.append(
                RemovedRow(
                    row_number=row_number,
                    reason=reason,
                    sample=normalized_text_signature(row)[:120],
                )
            )

    result = CleanResult(
        input_rows=len(rows),
        kept_rows=len(kept),
        removed_rows=len(removed),
        removed_exact_duplicates=exact_count,
        removed_normalized_duplicates=normalized_count,
        removed_low_information=low_information_count,
        removed=removed[:CLEAN_MANIFEST_SAMPLE_LIMIT],
    )
    return kept, result
