from corpus_studio.tokenization import estimate as estimate_mod
from corpus_studio.tokenization.estimate import (
    estimate_tokens,
    estimator_name,
    rough_token_estimate,
)


def test_empty_text_is_zero_tokens():
    assert rough_token_estimate("") == 0
    assert estimate_tokens("") == 0


def test_non_empty_text_is_at_least_one_token():
    assert rough_token_estimate("a") == 1
    assert rough_token_estimate("abcd") == 1


def test_estimate_scales_with_length():
    # A long single "word" is still bounded by the char/4 estimate.
    assert rough_token_estimate("a" * 40) == 10
    assert rough_token_estimate("x" * 400) == 100


def test_multi_word_text_counts_word_pieces():
    # Five words should not collapse to one or two tokens.
    assert rough_token_estimate("the quick brown fox jumps") >= 5


def test_cjk_is_not_underestimated():
    text = "日本語テスト"  # 6 CJK/kana chars -> ~1+ token each
    # Old len//4 gave 1; the Unicode-aware estimate should be far higher.
    assert rough_token_estimate(text) >= 5
    assert rough_token_estimate(text) > len(text) // 4


def test_estimate_tokens_falls_back_to_heuristic_without_tiktoken():
    # tiktoken is optional; when absent, estimate_tokens uses the heuristic.
    assert estimate_tokens("hello world, this is a test") >= 5


# --- model-specific tokenizer (optional `tokenizers` extra) -------------------


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Stand-in for a Hub tokenizer: one id per whitespace-split token."""

    def encode(self, text: str) -> _FakeEncoding:
        return _FakeEncoding(list(range(len(text.split()))))


def _use_fake_hf_tokenizer(monkeypatch, tokenizer=_FakeTokenizer()):
    estimate_mod._hf_tokenizer_cache.clear()
    monkeypatch.setattr(estimate_mod, "_load_hf_tokenizer", lambda model_id: tokenizer)


def _use_no_hf_tokenizer(monkeypatch):
    estimate_mod._hf_tokenizer_cache.clear()
    monkeypatch.setattr(estimate_mod, "_load_hf_tokenizer", lambda model_id: None)


def test_model_tokenizer_is_used_when_available(monkeypatch):
    _use_fake_hf_tokenizer(monkeypatch)
    # "a b c d e" -> the fake tokenizer yields exactly 5 ids for the 5 words.
    assert estimate_tokens("a b c d e", model_id="fake/model") == 5
    assert estimator_name("fake/model") == "hf:fake/model"


def test_falls_back_when_model_tokenizer_unavailable(monkeypatch):
    _use_no_hf_tokenizer(monkeypatch)
    # A model id is given but its tokenizer can't load (not installed / no network /
    # unknown model) -> falls through to the heuristic, never raises.
    assert estimate_tokens("hello world, this is a test", model_id="fake/model") >= 5
    assert estimator_name("fake/model") == "heuristic"


def test_no_model_id_skips_the_model_tokenizer_tier(monkeypatch):
    # Even with a model tokenizer available, omitting model_id keeps the prior
    # (model-agnostic) behavior — so existing callers are unaffected.
    _use_fake_hf_tokenizer(monkeypatch)
    assert estimator_name() == "heuristic"
    assert estimate_tokens("a b c d e") >= 5  # heuristic, not the 5-id fake
