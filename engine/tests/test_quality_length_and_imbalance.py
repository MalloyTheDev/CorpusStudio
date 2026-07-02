from corpus_studio.quality.basic_quality import build_basic_quality_report


# --- token-length outliers ---------------------------------------------------

def test_flags_long_token_outlier():
    rows = [{"instruction": "q", "output": "short answer"} for _ in range(10)]
    rows.append({"instruction": "q", "output": "word " * 300})  # row 11, huge
    report = build_basic_quality_report(rows)
    assert report.token_length_outlier_count >= 1
    assert 11 in [outlier.row_number for outlier in report.token_length_outliers]
    assert report.token_length_threshold > 0


def test_no_token_outliers_for_uniform_lengths():
    rows = [
        {"instruction": f"explain topic {index}", "output": "a normal answer of moderate size"}
        for index in range(12)
    ]
    report = build_basic_quality_report(rows)
    assert report.token_length_outlier_count == 0


def test_token_outliers_skipped_below_min_rows():
    rows = [{"instruction": "q", "output": "x"} for _ in range(3)]
    rows.append({"instruction": "q", "output": "word " * 200})
    report = build_basic_quality_report(rows)  # only 4 rows < 8
    assert report.token_length_outlier_count == 0


# --- category imbalance ------------------------------------------------------

def test_flags_category_imbalance():
    rows = [{"instruction": f"q{index}", "output": f"a{index}", "label": "pos"} for index in range(9)]
    rows.append({"instruction": "q9", "output": "a9", "label": "neg"})
    report = build_basic_quality_report(rows)

    label_findings = [item for item in report.category_imbalances if item.field == "label"]
    assert label_findings
    assert label_findings[0].dominant_value == "pos"
    assert label_findings[0].dominant_count == 9
    assert label_findings[0].share >= 0.8


def test_balanced_field_not_flagged():
    rows = [
        {"instruction": f"q{index}", "output": f"a{index}", "label": "pos" if index % 2 == 0 else "neg"}
        for index in range(10)
    ]
    report = build_basic_quality_report(rows)
    assert not any(item.field == "label" for item in report.category_imbalances)


def test_high_cardinality_field_not_flagged():
    rows = [{"instruction": f"q{index}", "output": f"a{index}", "id": str(index)} for index in range(12)]
    report = build_basic_quality_report(rows)
    assert report.category_imbalances == []


def test_imbalance_skipped_below_min_rows():
    rows = [{"instruction": f"q{index}", "output": "answer", "label": "pos"} for index in range(5)]
    report = build_basic_quality_report(rows)
    assert report.category_imbalances == []


def test_boolean_field_imbalance_is_flagged():
    rows = [{"instruction": f"q{index}", "output": f"a{index}", "passed": True} for index in range(11)]
    rows.append({"instruction": "q11", "output": "a11", "passed": False})
    report = build_basic_quality_report(rows)
    assert any(item.field == "passed" for item in report.category_imbalances)
