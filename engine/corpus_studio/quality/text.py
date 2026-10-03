"""Shared text extraction and tokenization for the quality plane.

Lifted out of ``basic_quality`` unchanged so the quality SIGNALS and the shape/applicability
assessment (:mod:`corpus_studio.quality.applicability`) can share one tokenizer without importing
each other. ``basic_quality`` re-exports these names, so existing callers are unaffected.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# CJK / kana / Hangul scripts have no spaces between words, so each such
# character is treated as its own token; other word characters group into runs.
# This keeps near-duplicate signatures and low-information counts meaningful for
# non-Latin text while preserving ASCII tokenization exactly.
_CJK_RANGES = (
    "\u3040-\u30ff"  # Hiragana + Katakana
    "\u3400-\u4dbf"  # CJK Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\uac00-\ud7af"  # Hangul syllables
    "\uff66-\uff9f"  # Half-width Katakana
)
_TOKEN_RE = re.compile(rf"[{_CJK_RANGES}]|[^\W{_CJK_RANGES}]+", re.UNICODE)


def collect_text_values(value: Any) -> list[str]:
    """Every scalar inside ``value``, flattened to strings, in container order."""
    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        collected: list[str] = []
        for item in value.values():
            collected.extend(collect_text_values(item))
        return collected

    if isinstance(value, list):
        collected = []
        for item in value:
            collected.extend(collect_text_values(item))
        return collected

    if value is None:
        return []

    return [str(value)]


def tokenize_text_values(value: Any) -> list[str]:
    """NFKC-normalized, lowercased tokens of every scalar inside ``value``."""
    text = unicodedata.normalize("NFKC", " ".join(collect_text_values(value))).lower()
    return _TOKEN_RE.findall(text)
