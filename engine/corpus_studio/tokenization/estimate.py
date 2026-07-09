"""Token count estimation.

The old estimate was a flat ``len(text) // 4``, which badly underestimates
non-Latin scripts (CJK/kana/Hangul have no spaces, so char/4 is far too low) and
ignores word/punctuation structure. ``estimate_tokens`` picks the most exact
counter available, in order, and never adds a hard dependency:

1. the **target model's own tokenizer** (via the optional ``tokenizers`` library —
   ``Tokenizer.from_pretrained`` fetches just the model's ``tokenizer.json``) when a
   ``model_id`` is given and the library + model are available — exact for that model;
2. **tiktoken** (optional ``tokenizer`` extra) — exact BPE for the GPT-4 family;
3. a **Unicode-aware heuristic** that counts CJK characters directly and blends
   word- and character-based estimates for everything else.

Every failure (library absent, no network, gated/unknown model, slow-only
tokenizer) falls silently to the next tier, and ``estimator_name`` reports which
tier actually ran so a token budget is never presented as exact when it is not.
"""

import re
import unicodedata
from typing import Any

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


_hf_tokenizer_cache: dict[str, Any] = {}


def _load_hf_tokenizer(model_id: str) -> Any:
    """Load a model's fast tokenizer via the optional ``tokenizers`` library.

    ``Tokenizer.from_pretrained`` fetches only the model's ``tokenizer.json`` from
    the Hub (far lighter than ``transformers``). Returns ``None`` on ANY failure —
    library not installed, no network, gated/unknown model, or a model that ships
    only a slow tokenizer — so estimation always falls back and never raises.
    Kept as a module-level function so tests can inject a fake tokenizer.
    """
    try:  # pragma: no cover - depends on the environment / network
        from tokenizers import Tokenizer

        return Tokenizer.from_pretrained(model_id)
    except Exception:  # noqa: BLE001 - optional accuracy boost, never a hard dependency.
        return None


def _hf_tokenizer(model_id: str) -> Any:
    if model_id not in _hf_tokenizer_cache:
        _hf_tokenizer_cache[model_id] = _load_hf_tokenizer(model_id)
    return _hf_tokenizer_cache[model_id]


def estimate_tokens(text: str, model_id: str | None = None) -> int:
    """Best-effort token count: the target model's own tokenizer when available,
    else tiktoken, else the Unicode-aware heuristic. ``model_id`` is a Hub id such
    as ``"meta-llama/Llama-3-8B"``; omit it to skip the model-specific tier."""
    if not text:
        return 0

    if model_id:
        tokenizer = _hf_tokenizer(model_id)
        if tokenizer is not None:
            try:
                return max(1, len(tokenizer.encode(text).ids))
            except Exception:  # noqa: BLE001 - fall through to tiktoken / heuristic.
                pass

    encoder = _tiktoken_encoder()
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text)))
        except Exception:  # noqa: BLE001 - fall back to the heuristic on any error.
            pass

    return _heuristic_token_estimate(text)


def estimator_name(model_id: str | None = None) -> str:
    """Which estimator ``estimate_tokens`` will use for this model: the model's own
    Hub tokenizer (``'hf:<model_id>'``), ``'tiktoken'``, or ``'heuristic'``."""
    if model_id and _hf_tokenizer(model_id) is not None:
        return f"hf:{model_id}"
    return "tiktoken" if _tiktoken_encoder() is not None else "heuristic"
