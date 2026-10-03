"""Opt-in PII / secret **redaction** for exports (v1.x).

Masks the *same* high-precision PII/secret patterns the quality reporter
DETECTS (emails, SSNs, private keys, AWS/API keys, JWTs, Luhn-valid payment
cards) by replacing each matched span with a typed placeholder such as
``[REDACTED:email]``. Reusing ``basic_quality``'s patterns is deliberate: the
thing that gets redacted is the thing that gets flagged, so the export gate
that BLOCKS on PII/secrets passes once redaction has masked them.

Private keys are redacted over the same unit the detector scans: the text of
the row's leaves joined in order (``_collect_text_values``). A key is masked as a
whole PEM block - BEGIN marker, body and END marker - never just its header,
even when the block spans several fields or list items (a key stored as a list
of lines, or split across ``instruction`` / ``output``). An unterminated block
is masked to the end of the row, and an END marker with no BEGIN marker before
it masks back to the previous key, an existing placeholder, or the start of
the row (see ``_private_key_spans``), so no key material survives next to a
placeholder. Every string leaf inside a block is masked except a chat ``role``
word, which carries no key material and which the chat schema requires; JSON
numbers and booleans inside a block keep their value and type.

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
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from corpus_studio.quality.basic_quality import (
    _PII_CC_CANDIDATE_RE,
    _PII_PATTERNS,
    _PII_PRIVATE_KEY_RE,
    _collect_text_values,
    _looks_like_payment_card,
    _luhn_valid,
)
from corpus_studio.validators.basic_validator import VALID_MESSAGE_ROLES

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


def _private_key_spans(text: str) -> list[tuple[int, int]]:
    """The ``[start, end)`` spans of ``text`` to mask as private-key material, one per match of
    the detector's pattern, so the redaction count equals the detector's match count.

    A BEGIN match already spans the whole block (through its END marker, or to the end of the
    text when unterminated). A bare END marker means the block's body sits before it while its
    BEGIN marker does not: a key cut off before its header, or a header an earlier header-only
    redaction already replaced. The span then extends back to the nearest earlier boundary - the
    end of the previous private-key match, or a pre-existing private-key placeholder after it,
    else the start of the text - so the body cannot survive beside the mask. Over-masking the
    prose before a stray END marker is the accepted cost: leaking key material is not. Each
    backward window starts where the previous match ended, so windows never overlap and the pass
    stays linear in the length of the text.
    """
    placeholder = _placeholder("private_key")
    spans: list[tuple[int, int]] = []
    cursor = 0  # end of the text already assigned to a span or left unmasked
    for match in _PII_PRIVATE_KEY_RE.finditer(text):
        start = match.start()
        if text.startswith("-----END", start):
            earlier_mask = text.rfind(placeholder, cursor, start)
            start = earlier_mask if earlier_mask != -1 else cursor
        spans.append((start, match.end()))
        cursor = match.end()
    return spans


def _mask_private_key_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace each sorted, non-overlapping ``[start, end)`` span of ``text`` with the
    private-key placeholder."""
    placeholder = _placeholder("private_key")
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        pieces.append(placeholder)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _leaf_segments(leaves: list[str], spans: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Map spans of ``" ".join(leaves)`` onto each leaf as leaf-local ``[start, end)`` segments.

    The single-space separators belong to no leaf, so a span that only touches a separator adds
    nothing. Both the spans and the leaves are walked in order once, which keeps the mapping
    linear in the number of leaves plus spans.
    """
    segments: list[list[tuple[int, int]]] = [[] for _ in leaves]
    first = 0  # first leaf that does not end before the current span starts
    first_offset = 0  # offset of that leaf in the joined text
    for span_start, span_end in spans:
        while first < len(leaves) and first_offset + len(leaves[first]) <= span_start:
            first_offset += len(leaves[first]) + 1
            first += 1
        index, offset = first, first_offset
        while index < len(leaves) and offset < span_end:
            low = max(span_start, offset) - offset
            high = min(span_end, offset + len(leaves[index])) - offset
            if high > low:
                segments[index].append((low, high))
            offset += len(leaves[index]) + 1
            index += 1
    return segments


def _replace_leaves(value: Any, masked: Iterator[str | None], key: Any = None) -> Any:
    """Rebuild ``value`` with its masked string leaves, visiting leaves in exactly the order
    ``_collect_text_values`` yields them (``None`` yields no leaf; any other scalar yields one).

    A leaf keeps its value when it has no mask, when it is a chat ``role`` word (a fixed
    vocabulary word the chat schema requires, never key material, so masking it would only make
    the schema gate refuse the row), or when it is not a string: a JSON number or boolean inside a
    block is not PEM text, and keeping it keeps its type.
    """
    if isinstance(value, dict):
        return {item_key: _replace_leaves(item, masked, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_replace_leaves(item, masked) for item in value]
    if value is None:
        return value
    replacement = next(masked)
    if replacement is None or not isinstance(value, str):
        return value
    if key == "role" and value in VALID_MESSAGE_ROLES:
        return value
    return replacement


def _redact_row_private_keys(row: dict, tally: dict[str, int]) -> dict:
    """Mask private-key blocks across a whole row, over the text the detector scans.

    The detector matches private keys in the row's leaves joined with single spaces, so a
    block split across fields or list items is one match there. Masking each leaf on its own would
    leave a leaf that holds only key body (no marker) untouched, and once the markers were masked
    the gate would pass with that body still in the deliverable. The spans are therefore found in
    the joined text and mapped back onto every leaf they cover.
    """
    leaves = _collect_text_values(row)
    spans = _private_key_spans(" ".join(leaves))
    if not spans:
        return row
    tally["private_key"] = tally.get("private_key", 0) + len(spans)
    masked = [
        _mask_private_key_spans(leaf, leaf_spans) if leaf_spans else None
        for leaf, leaf_spans in zip(leaves, _leaf_segments(leaves, spans), strict=True)
    ]
    return _replace_leaves(row, iter(masked))


def _redact_other_patterns(text: str, hits: dict[str, int]) -> str:
    """Mask every known pattern except private keys in one string, counting each in ``hits``.
    Patterns are applied in the detector's order; a placeholder never matches a later pattern,
    so masks don't cascade."""

    def _make_sub(kind: str):
        def _sub(_match: re.Match[str]) -> str:
            hits[kind] = hits.get(kind, 0) + 1
            return _placeholder(kind)

        return _sub

    for kind, _severity, pattern, _suggestion in _PII_PATTERNS:
        if kind != "private_key":
            text = pattern.sub(_make_sub(kind), text)

    # Payment cards: only mask a candidate digit run that is Luhn-valid AND card-shaped, so we
    # don't clobber arbitrary long numbers (order/phone/ID). SSNs (9 digits) never reach here -
    # they are shorter than the 13-digit candidate floor and are already masked above.
    def _cc_sub(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits) and _looks_like_payment_card(digits):
            hits["credit_card"] = hits.get("credit_card", 0) + 1
            return _placeholder("credit_card")
        return match.group(0)

    return _PII_CC_CANDIDATE_RE.sub(_cc_sub, text)


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Mask known PII/secret spans in one string.

    Returns the redacted text and a ``{kind: count}`` tally. Private-key blocks are masked first
    (the string is treated as a one-field row), then the other patterns in the detector's order.
    """
    hits: dict[str, int] = {}
    spans = _private_key_spans(text)
    if spans:
        hits["private_key"] = len(spans)
        text = _mask_private_key_spans(text, spans)
    return _redact_other_patterns(text, hits), hits


def _redact_value(value: Any, tally: dict[str, int]) -> Any:
    """Recursively redact the non-private-key PII/secret spans in a row value (private keys are
    masked across the whole row first, by ``_redact_row_private_keys``). String leaves are masked
    in place; a NUMERIC leaf whose text form contains PII (e.g. a payment-card number or SSN
    stored as a JSON number) is masked to the placeholder string, so numeric PII the
    quality gate flags can actually be cleared (#505). A value with no match is returned
    unchanged and keeps its type."""
    if isinstance(value, str):
        return _redact_other_patterns(value, tally)
    if isinstance(value, list):
        return [_redact_value(item, tally) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item, tally) for key, item in value.items()}
    # PII detection scans numeric leaves as text (a card number stored as a number is
    # still PII), so redaction must be able to clear them too - but only on a real match.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        hits: dict[str, int] = {}
        redacted = _redact_other_patterns(str(value), hits)
        if hits:
            for kind, count in hits.items():
                tally[kind] = tally.get(kind, 0) + count
            return redacted
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
        row = _redact_row_private_keys(row, row_tally)
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
