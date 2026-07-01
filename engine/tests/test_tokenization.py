from corpus_studio.tokenization.estimate import rough_token_estimate


def test_empty_text_is_zero_tokens():
    assert rough_token_estimate("") == 0


def test_non_empty_text_is_at_least_one_token():
    assert rough_token_estimate("a") == 1
    assert rough_token_estimate("abcd") == 1


def test_estimate_scales_with_length():
    assert rough_token_estimate("a" * 40) == 10
    assert rough_token_estimate("x" * 400) == 100
