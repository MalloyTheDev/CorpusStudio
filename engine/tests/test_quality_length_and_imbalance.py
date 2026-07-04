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


# --- item 15b: imbalance share is dataset-relative, not present-relative ------

def test_imbalance_share_is_dataset_relative_not_present_relative():
    # `label` is present on 16/20 rows; 15 of those are "pos". Present-share = 15/16 = 0.94
    # (the old denominator would flag), but dataset-share = 15/20 = 0.75 is below the 0.8
    # threshold, so the field is NOT dataset-imbalanced.
    rows = [{"instruction": f"q{i}", "output": f"a{i}", "label": "pos"} for i in range(15)]
    rows.append({"instruction": "q15", "output": "a15", "label": "neg"})
    rows.extend({"instruction": f"q{i}", "output": f"a{i}"} for i in range(16, 20))  # 4 rows, no label
    report = build_basic_quality_report(rows)
    assert not any(item.field == "label" for item in report.category_imbalances)


def test_imbalance_still_flags_when_dataset_dominant_despite_partial_presence():
    # `label` present on 18/20 rows, 17 "pos": dataset-share = 17/20 = 0.85 >= 0.8 -> flagged,
    # and the reported total/share are dataset-relative (self-consistent).
    rows = [{"instruction": f"q{i}", "output": f"a{i}", "label": "pos"} for i in range(17)]
    rows.append({"instruction": "q17", "output": "a17", "label": "neg"})
    rows.extend({"instruction": f"q{i}", "output": f"a{i}"} for i in range(18, 20))  # 2 rows, no label
    report = build_basic_quality_report(rows)
    label = [item for item in report.category_imbalances if item.field == "label"]
    assert label
    assert label[0].dominant_count == 17
    assert label[0].total == 20
    assert label[0].share == 0.85
