"""Token count estimation.

The old estimate was a flat ``len(text) // 4``, which badly underestimates
non-Latin scripts (CJK/kana/Hangul have no spaces, so char/4 is far too low) and
ignores word/punctuation structure. ``estimate_tokens`` uses a real tokenizer
(tiktoken) when it is installed and otherwise a Unicode-aware heuristic that
counts CJK characters directly and blends word- and character-based estimates
for everything else. No hard dependency is added.
"""

import re
import unicodedata

# CJK / kana / Hangul: roughly one (sometimes more) BPE token per character,
# and no whitespace to split on, so these must be counted directly.
_CJK_RANGES = (
    "぀-ヿ"  # Hiragana + Katakana
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "가-힯"  # Hangul syllables
    "ｦ-ﾟ"  # Half-width Katakana
)
_CJK_RE = re.compile(f"[{_CJK_RANGES}]")
_WORD_OR_PUNCT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Rough tokens-per-CJK-character factor for BPE tokenizers.
_CJK_TOKEN_FACTOR = 1.5


def _heuristic_token_estimate(text: str) -> int:
    """Deterministic, dependency-free token estimate.

    Preserves the historical ``len // 4`` result for long single-token strings
    while counting CJK characters directly and word/punctuation pieces for
    space-delimited text.
    """
    if not text:
        return 0

    normalized = unicodedata.normalize("NFKC", text)
    cjk_count = len(_CJK_RE.findall(normalized))
    non_cjk = _CJK_RE.sub(" ", normalized)

    pieces = len(_WORD_OR_PUNCT_RE.findall(non_cjk))
    char_estimate = len(non_cjk.strip()) / 4
    # Take the larger of the word/punct count and the char estimate so a long
    # single "word" (which BPE splits into many tokens) is not underestimated.
    non_cjk_tokens = max(pieces, char_estimate)

    return max(1, round(non_cjk_tokens + cjk_count * _CJK_TOKEN_FACTOR))


def rough_token_estimate(text: str) -> int:
    """Deterministic heuristic token estimate (see ``_heuristic_token_estimate``).

    Kept as the backwards-compatible name and never routed through tiktoken, so
    its behavior does not depend on the environment.
    """
    return _heuristic_token_estimate(text)


_encoder = None
_encoder_loaded = False


def _tiktoken_encoder():
    global _encoder, _encoder_loaded
    if not _encoder_loaded:
        _encoder_loaded = True
        try:  # tiktoken is an optional accuracy boost, never a hard dependency.
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - any import/load failure falls back.
            _encoder = None
    return _encoder


def estimate_tokens(text: str) -> int:
    """Best-effort token count: exact via tiktoken if available, else heuristic."""
    if not text:
        return 0

    encoder = _tiktoken_encoder()
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text)))
        except Exception:  # noqa: BLE001 - fall back to the heuristic on any error.
            pass

    return _heuristic_token_estimate(text)


def estimator_name() -> str:
    """Which estimator ``estimate_tokens`` will use: 'tiktoken' or 'heuristic'."""
    return "tiktoken" if _tiktoken_encoder() is not None else "heuristic"
