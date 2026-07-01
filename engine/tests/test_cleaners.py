from corpus_studio.cleaners.basic_cleaner import normalize_text, remove_empty_rows


def test_normalize_text_unifies_line_endings_and_strips():
    assert normalize_text("a\r\nb\rc\n") == "a\nb\nc"
    assert normalize_text("  hello  ") == "hello"
    assert normalize_text("") == ""


def test_remove_empty_rows_keeps_rows_with_any_content():
    rows = [{"a": ""}, {"a": "  "}, {"a": "x"}, {"a": 0}]
    kept = remove_empty_rows(rows)
    assert {"a": "x"} in kept
    assert {"a": 0} in kept  # str(0) is non-empty
    assert {"a": ""} not in kept
    assert {"a": "  "} not in kept
