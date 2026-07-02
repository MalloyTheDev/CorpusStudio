from corpus_studio.tokenization.estimate import estimate_tokens, rough_token_estimate


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
