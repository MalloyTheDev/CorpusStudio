"""Opt-in PII / secret **redaction** for exports (v1.x).

Masks the *same* high-precision PII/secret patterns the quality reporter
DETECTS (emails, SSNs, private keys, AWS/API keys, JWTs, Luhn-valid payment
cards) by replacing each matched span with a typed placeholder such as
``[REDACTED:email]``. Reusing ``basic_quality``'s patterns is deliberate: the
thing that gets redacted is exactly the thing that gets flagged, so the export
gate that BLOCKS on PII/secrets passes once redaction has masked them.

**Honesty boundary (read before trusting this):** redaction is a safety net for
*known, high-precision* patterns — it is **not** a guarantee of de-identification.
Novel or obfuscated secret formats, personal names, postal addresses,
free-text identifiers, and anything outside the detector's patterns are **not**
caught. Treat a redacted export as "known patterns masked", never as "safe to
publish".

Redaction runs only when producing an export — the engine never rewrites
``examples.jsonl`` in place — and records a manifest of *what* was masked (kind +
counts + affected rows). The manifest never stores the raw secret values.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from corpus_studio.quality.basic_quality import (
    _PII_CC_CANDIDATE_RE,
    _PII_PATTERNS,
    _looks_like_payment_card,
    _luhn_valid,
)

REDACTION_MANIFEST_ROW_LIMIT = 50


class RedactionHit(BaseModel):
    kind: str  # email | ssn | api_key | aws_access_key | private_key | jwt | credit_card
    count: int


class RedactionReport(BaseModel):
    """What redaction masked — counts only, never the raw values."""

    redacted_spans: int = 0  # total individual matches masked
    redacted_rows: int = 0  # rows that had at least one match
    by_kind: list[RedactionHit] = Field(default_factory=list)
    # First N affected row numbers (1-based), for a spot-check; capped so the manifest stays small.
    affected_row_numbers: list[int] = Field(default_factory=list)


def _placeholder(kind: str) -> str:
    return f"[REDACTED:{kind}]"


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Mask known PII/secret spans in one string.

    Returns the redacted text and a ``{kind: count}`` tally. Patterns are applied
    in the detector's order; a placeholder never matches a later pattern, so
    masks don't cascade.
    """
    hits: dict[str, int] = {}

    def _make_sub(kind: str):
        def _sub(_match: re.Match[str]) -> str:
            hits[kind] = hits.get(kind, 0) + 1
            return _placeholder(kind)

        return _sub

    for kind, _severity, pattern, _suggestion in _PII_PATTERNS:
        text = pattern.sub(_make_sub(kind), text)

    # Payment cards: only mask a candidate digit run that is Luhn-valid AND card-shaped, so we
    # don't clobber arbitrary long numbers (order/phone/ID). SSNs (9 digits) never reach here —
    # they are shorter than the 13-digit candidate floor and are already masked above.
    def _cc_sub(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits) and _looks_like_payment_card(digits):
            hits["credit_card"] = hits.get("credit_card", 0) + 1
            return _placeholder("credit_card")
        return match.group(0)

    text = _PII_CC_CANDIDATE_RE.sub(_cc_sub, text)
    return text, hits


def _redact_value(value: Any, tally: dict[str, int]) -> Any:
    """Recursively redact string leaves in a row value; non-strings pass through."""
    if isinstance(value, str):
        redacted, hits = redact_text(value)
        for kind, count in hits.items():
            tally[kind] = tally.get(kind, 0) + count
        return redacted
    if isinstance(value, list):
        return [_redact_value(item, tally) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item, tally) for key, item in value.items()}
    return value


def redact_rows(rows: list[dict]) -> tuple[list[dict], RedactionReport]:
    """Redact known PII/secrets across every row (pure). Returns the redacted rows plus a
    manifest of what was masked — counts and affected row numbers only, no raw values."""
    redacted_rows: list[dict] = []
    by_kind_total: dict[str, int] = {}
    affected: list[int] = []
    total_spans = 0

    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            redacted_rows.append(row)
            continue
        row_tally: dict[str, int] = {}
        redacted_rows.append({key: _redact_value(value, row_tally) for key, value in row.items()})
        if row_tally:
            affected.append(row_number)
            for kind, count in row_tally.items():
                by_kind_total[kind] = by_kind_total.get(kind, 0) + count
                total_spans += count

    report = RedactionReport(
        redacted_spans=total_spans,
        redacted_rows=len(affected),
        by_kind=[
            RedactionHit(kind=kind, count=count) for kind, count in sorted(by_kind_total.items())
        ],
        affected_row_numbers=affected[:REDACTION_MANIFEST_ROW_LIMIT],
    )
    return redacted_rows, report
