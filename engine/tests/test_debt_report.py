import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.quality.basic_quality import (
    CategoryImbalance,
    PiiFinding,
    QualityReport,
    SyntheticPatternIssue,
)
from corpus_studio.reporting.debt_report import (
    build_debt_report,
    render_debt_report_markdown,
)

runner = CliRunner()


def _quality(example_count: int = 10, **kw) -> QualityReport:
    base = dict(
        example_count=example_count,
        empty_row_count=0,
        duplicate_exact_count=0,
        duplicate_normalized_count=0,
        low_information_count=0,
    )
    base.update(kw)
    return QualityReport(**base)


# --- grade + emptiness -------------------------------------------------------

def test_clean_dataset_is_grade_a():
    report = build_debt_report(_quality())
    assert report.grade == "A"
    assert report.items == []
    assert report.clean is True


def test_empty_dataset_is_na_not_a():
    report = build_debt_report(_quality(example_count=0))
    assert report.has_data is False
    assert report.grade == "N/A"
    assert report.items == []
    assert report.clean is False  # no data is not "clean"


def test_high_exact_dup_rate_is_high_and_grade_d():
    report = build_debt_report(_quality(example_count=10, duplicate_exact_count=6))
    item = report.items[0]
    assert item.category == "exact_duplicates"
    assert item.severity == "high"
    assert item.rate == 0.6  # normalized: 6/10
    assert report.grade == "D"


def test_low_dup_rate_is_low_and_grade_b():
    # 1/1000 = 0.001 -> below the 0.01 moderate threshold -> low -> grade B.
    report = build_debt_report(_quality(example_count=1000, duplicate_exact_count=1))
    assert report.items[0].severity == "low"
    assert report.grade == "B"


# --- the safety-critical PII rule (presence, NOT rate) -----------------------

def test_single_high_pii_is_critical_even_in_a_huge_dataset():
    finding = PiiFinding(kind="api_key", severity="high", match_count=1, sample="AKIA…", suggestion="Remove")
    report = build_debt_report(
        _quality(example_count=100_000, pii_finding_count=1, pii_findings=[finding])
    )
    assert report.grade == "F"
    secrets = [i for i in report.items if i.category == "secrets"]
    assert len(secrets) == 1
    assert secrets[0].severity == "critical"
    assert secrets[0].rate is None  # presence, never normalized by rate


def test_medium_pii_is_high_severity():
    finding = PiiFinding(kind="email", severity="medium", match_count=3, sample="a@b.com", suggestion="Redact")
    report = build_debt_report(_quality(pii_finding_count=1, pii_findings=[finding]))
    personal = [i for i in report.items if i.category == "personal_data"]
    assert len(personal) == 1 and personal[0].severity == "high"
    assert report.grade == "D"


def test_high_pii_takes_precedence_over_medium():
    findings = [
        PiiFinding(kind="api_key", severity="high", match_count=1, sample="k", suggestion="s"),
        PiiFinding(kind="email", severity="medium", match_count=1, sample="e", suggestion="s"),
    ]
    report = build_debt_report(_quality(pii_finding_count=2, pii_findings=findings))
    # A single 'secrets' critical item; medium is not separately added when high present.
    categories = [i.category for i in report.items]
    assert "secrets" in categories and "personal_data" not in categories


# --- ranking -----------------------------------------------------------------

def test_items_sorted_highest_severity_first():
    finding = PiiFinding(kind="api_key", severity="high", match_count=1, sample="k", suggestion="s")
    report = build_debt_report(
        _quality(example_count=1000, duplicate_exact_count=5, pii_finding_count=1, pii_findings=[finding])
    )
    # critical (secrets) must come before the low duplicate item.
    assert report.items[0].category == "secrets"
    assert report.items[0].severity == "critical"
    assert report.grade == "F"


def test_category_imbalance_severity_by_share():
    imbalance = CategoryImbalance(
        field="label", dominant_value="yes", dominant_count=95, total=100, share=0.95, distinct_values=2
    )
    report = build_debt_report(_quality(example_count=100, category_imbalances=[imbalance]))
    item = [i for i in report.items if i.category == "category_imbalance"][0]
    assert item.severity == "high"  # share 0.95 > 0.90
    assert item.rate is None


def test_synthetic_pattern_severity_maps_from_issue():
    issue = SyntheticPatternIssue(kind="repetition", severity="high", message="m", suggestion="s")
    report = build_debt_report(
        _quality(synthetic_pattern_count=4, synthetic_pattern_issues=[issue])
    )
    item = [i for i in report.items if i.category == "synthetic_patterns"][0]
    assert item.severity == "high" and item.rate is None


# --- purity + render ---------------------------------------------------------

def test_build_is_pure():
    q = _quality(example_count=50, duplicate_exact_count=3, low_information_count=6)
    assert build_debt_report(q).model_dump() == build_debt_report(q).model_dump()


def test_render_leads_with_grade_and_is_injection_safe():
    imbalance = CategoryImbalance(
        field="lab\nel\n> injected", dominant_value="v", dominant_count=95, total=100,
        share=0.95, distinct_values=2,
    )
    report = build_debt_report(_quality(example_count=100, category_imbalances=[imbalance]))
    markdown = render_debt_report_markdown(report)
    assert markdown.startswith("# Dataset Debt — Grade ")
    assert "\n> injected" not in markdown  # sanitized field cannot inject a line


def test_render_clean_and_empty():
    assert "No debt detected" in render_debt_report_markdown(build_debt_report(_quality()))
    assert "No rows to assess" in render_debt_report_markdown(
        build_debt_report(_quality(example_count=0))
    )


# --- CLI end-to-end ----------------------------------------------------------

def test_cli_dataset_debt_end_to_end(tmp_path: Path):
    rows = [{"instruction": "A", "output": "1"}, {"instruction": "A", "output": "1"}]  # a duplicate
    path = tmp_path / "examples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    as_json = runner.invoke(app, ["dataset-debt", str(path), "--json"])
    assert as_json.exit_code == 0, as_json.output
    report = json.loads(as_json.stdout)
    assert report["grade"] in {"A", "B", "C", "D", "F"}
    assert any(item["category"] == "exact_duplicates" for item in report["items"])

    markdown = runner.invoke(app, ["dataset-debt", str(path)])
    assert markdown.exit_code == 0
    assert "Grade" in markdown.stdout


def test_cli_dataset_debt_bad_path_exits_1(tmp_path: Path):
    result = runner.invoke(app, ["dataset-debt", str(tmp_path / "nope.jsonl")])
    assert result.exit_code == 1
