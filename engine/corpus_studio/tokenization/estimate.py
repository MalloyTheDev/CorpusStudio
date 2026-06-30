def rough_token_estimate(text: str) -> int:
    """Very rough token estimate.

    Replace later with tokenizer-specific estimation.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
