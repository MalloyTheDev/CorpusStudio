"""Regression tests for two quality-signal correctness fixes (deep-audit cluster):

- item 11: the normalized near-duplicate signature is field-aware, so two rows with the
  same *combined* text but a different field split are no longer counted as near-dups
  (which caused false drops and false split-leakage blocks).
- item 15c: synthetic_pattern_count reports the true total even when the displayed sample
  is capped at SYNTHETIC_WARNING_LIMIT.
"""

from corpus_studio.quality.basic_quality import (
    SYNTHETIC_WARNING_LIMIT,
    _normalized_text_signature,
    build_basic_quality_report,
)


# --- item 11: field-aware near-duplicate signature ---------------------------

def test_same_combined_text_different_fields_is_not_a_near_duplicate():
    # Both rows share the token stream "hello world foo bar", but split across fields
    # differently. A flat signature collided them; the field-aware one must not.
    rows = [
        {"instruction": "hello world", "output": "foo bar"},
        {"instruction": "hello world foo", "output": "bar"},
    ]
    report = build_basic_quality_report(rows)
    assert report.duplicate_normalized_count == 0


def test_genuine_near_duplicate_still_detected():
    # Same fields, whitespace/case-only variation — still a normalized duplicate.
    rows = [
        {"instruction": "Hello  World", "output": "Foo Bar"},
        {"instruction": "hello world", "output": "foo bar"},
    ]
    report = build_basic_quality_report(rows)
    assert report.duplicate_exact_count == 0
    assert report.duplicate_normalized_count >= 1


def test_signature_is_key_scoped_and_order_independent():
    # Field-aware signatures are keyed and key-sorted: identical dicts match regardless of
    # insertion order, but moving a token across fields changes the signature.
    a = _normalized_text_signature({"instruction": "hello world", "output": "foo bar"})
    b = _normalized_text_signature({"output": "foo bar", "instruction": "hello world"})
    c = _normalized_text_signature({"instruction": "hello world foo", "output": "bar"})
    assert a == b
    assert a != c


# --- item 15c: synthetic_pattern_count is the true total, not the capped sample ----

def test_synthetic_pattern_count_reports_true_total_beyond_display_cap():
    # 11 distinct openings + 11 distinct closings = 22 synthetic issues, above the 20 cap.
    rows: list[dict] = []
    for group in range(11):
        text = f"cluster{group} alpha beta gamma delta epsilon zeta tail{group}"
        rows.extend({"instruction": "q", "output": text} for _ in range(3))

    report = build_basic_quality_report(rows)
    assert report.synthetic_pattern_count == 22               # true total
    assert len(report.synthetic_pattern_issues) == SYNTHETIC_WARNING_LIMIT  # displayed sample bounded
    assert report.synthetic_pattern_count > len(report.synthetic_pattern_issues)
