import pytest

from corpus_studio.tokenization import estimate as estimate_mod
from corpus_studio.tokenization.estimate import (
    estimate_tokens,
    estimate_tokens_with_tier,
    estimator_name,
    rough_token_estimate,
    tokenizer_offline,
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


# --- estimate_tokens_with_tier reports the tier that ACTUALLY ran (#568) -------


def test_with_tier_matches_estimate_tokens_and_handles_empty():
    assert estimate_tokens_with_tier("hello world")[0] == estimate_tokens("hello world")
    assert estimate_tokens_with_tier("")[0] == 0


def test_with_tier_reports_hf_when_the_model_tokenizer_runs(monkeypatch):
    _use_fake_hf_tokenizer(monkeypatch)
    count, tier = estimate_tokens_with_tier("a b c d e", model_id="fake/model")
    assert count == 5 and tier == "hf:fake/model"


def test_with_tier_reports_the_actual_tier_when_encode_raises(monkeypatch):
    # estimator_name PREDICTS hf (the tokenizer loads), but a per-text encode() can still raise;
    # estimate_tokens_with_tier reports whatever tier actually ran - the honesty guarantee (#568).
    class _EncodeRaises:
        def encode(self, text: str) -> object:
            raise RuntimeError("boom")

    _use_fake_hf_tokenizer(monkeypatch, tokenizer=_EncodeRaises())
    assert estimator_name("fake/model") == "hf:fake/model"  # prediction (the load succeeded)
    count, tier = estimate_tokens_with_tier("hello world", model_id="fake/model")
    assert count > 0
    assert tier != "hf:fake/model"  # honest: it fell through to a lower tier
    assert tier in ("tiktoken", "heuristic")
    assert estimate_tokens("a b c d e") >= 5  # heuristic, not the 5-id fake


# --- offline safety: the model tier must never touch the network when disabled ----


def _clear_offline_env(monkeypatch):
    for name in estimate_mod._OFFLINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    estimate_mod._hf_tokenizer_cache.clear()


@pytest.mark.parametrize("var", ["CORPUS_STUDIO_TOKENIZER_OFFLINE", "HF_HUB_OFFLINE"])
def test_tokenizer_offline_detects_each_flag(monkeypatch, var):
    _clear_offline_env(monkeypatch)
    assert tokenizer_offline() is False
    monkeypatch.setenv(var, "1")
    assert tokenizer_offline() is True


def test_offline_returns_none_without_any_fetch(monkeypatch):
    _clear_offline_env(monkeypatch)

    def _must_not_fetch(model_id):
        raise AssertionError("offline mode must not attempt the network fetch")

    monkeypatch.setattr(estimate_mod, "_fetch_hf_tokenizer", _must_not_fetch)
    monkeypatch.setenv("CORPUS_STUDIO_TOKENIZER_OFFLINE", "1")

    assert estimate_mod._load_hf_tokenizer("any/model") is None  # no AssertionError raised


def test_online_calls_the_fetch(monkeypatch):
    _clear_offline_env(monkeypatch)
    monkeypatch.setattr(estimate_mod, "_fetch_hf_tokenizer", lambda model_id: _FakeTokenizer())
    assert estimate_mod._load_hf_tokenizer("any/model") is not None


def test_offline_forces_a_deterministic_non_hf_estimator(monkeypatch):
    # Even if a model tokenizer *would* be fetchable, offline mode reports (and uses) a
    # network-free estimator, so the budget is reproducible and can't stall.
    _clear_offline_env(monkeypatch)
    monkeypatch.setattr(estimate_mod, "_fetch_hf_tokenizer", lambda model_id: _FakeTokenizer())
    monkeypatch.setenv("CORPUS_STUDIO_TOKENIZER_OFFLINE", "1")

    assert not estimator_name("fake/model").startswith("hf:")
