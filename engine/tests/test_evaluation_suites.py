import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import corpus_studio.cli as cli
from corpus_studio.cli import app
from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.gates.models import GateStatus
from corpus_studio.suites.models import SuiteCase, SuiteDefinition
from corpus_studio.suites.runner import load_suite_definition, run_suite, save_suite_report

runner = CliRunner()


def _report(avg: float = 80.0, tested: int = 10, failed: int = 1, metric: str = "keyword_overlap") -> EvaluationReport:
    return EvaluationReport(
        dataset="d", model="m", examples_tested=tested, average_score=avg, failed_examples=failed, metric=metric
    )


def _case(name: str, *, metric: str = "keyword_overlap", min_score: float | None = None,
          dataset_path: str = "ds.jsonl", judge_model: str | None = None) -> SuiteCase:
    return SuiteCase(
        name=name, schema="instruction", dataset_path=dataset_path, model="m",
        metric=metric, min_score=min_score, judge_model=judge_model,
    )


def _definition(*cases: SuiteCase, name: str = "my-suite") -> SuiteDefinition:
    return SuiteDefinition(name=name, cases=list(cases))


# --- pure run_suite (injected evaluate_case) --------------------------------

def test_rollup_and_overall_status():
    definition = _definition(
        _case("kw-pass"),
        _case("kw-block", min_score=90),  # 80 < 90 -> block
        _case("judge", metric="llm_judge", judge_model="j"),
    )
    reports = {"kw-pass": _report(avg=80), "kw-block": _report(avg=80), "judge": _report(avg=95, metric="llm_judge")}

    report = run_suite(definition, lambda case: reports[case.name])

    assert report.overall_status == GateStatus.BLOCK  # a blocked case blocks the suite
    per_metric = {rollup.metric: rollup for rollup in report.per_metric}
    assert per_metric["keyword_overlap"].total == 2
    assert per_metric["keyword_overlap"].passed == 1 and per_metric["keyword_overlap"].blocked == 1
    assert per_metric["llm_judge"].total == 1 and per_metric["llm_judge"].passed == 1


def test_per_metric_never_folds_scores():
    definition = _definition(_case("a"), _case("b", metric="llm_judge", judge_model="j"))
    reports = {"a": _report(avg=80), "b": _report(avg=20, metric="llm_judge")}

    report = run_suite(definition, lambda case: reports[case.name])

    assert {rollup.metric for rollup in report.per_metric} == {"keyword_overlap", "llm_judge"}
    # There is no single blended cross-metric score anywhere on the report.
    assert not hasattr(report, "average_score")


def test_per_case_isolation():
    definition = _definition(_case("good"), _case("bad"), _case("good2"))

    def evaluate(case: SuiteCase) -> EvaluationReport:
        if case.name == "bad":
            raise RuntimeError("backend down")
        return _report(avg=80)

    report = run_suite(definition, evaluate)

    statuses = {result.case: result.status for result in report.cases}
    assert statuses == {"good": "pass", "bad": "error", "good2": "pass"}
    assert report.overall_status == GateStatus.BLOCK  # an errored case blocks the suite
    bad = next(result for result in report.cases if result.case == "bad")
    assert "backend down" in (bad.error or "")


def test_case_min_score_overrides_default():
    strict = run_suite(_definition(_case("strict", min_score=90)), lambda c: _report(avg=80))
    assert strict.cases[0].status == "block"  # 80 < 90
    lenient = run_suite(_definition(_case("lenient")), lambda c: _report(avg=80))
    assert lenient.cases[0].status == "pass"  # 80 >= default 70


def test_all_pass_is_pass():
    report = run_suite(_definition(_case("a"), _case("b")), lambda c: _report(avg=85))
    assert report.overall_status == GateStatus.PASS
    assert "2 case(s)" in report.summary and "2 pass" in report.summary


def test_dataset_fingerprint_recorded(tmp_path: Path):
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    report = run_suite(_definition(_case("c", dataset_path=str(dataset))), lambda c: _report(avg=80))
    assert report.cases[0].dataset_fingerprint is not None


def test_missing_dataset_fingerprint_is_none_but_case_still_runs(tmp_path: Path):
    report = run_suite(
        _definition(_case("c", dataset_path=str(tmp_path / "nope.jsonl"))), lambda c: _report(avg=80)
    )
    assert report.cases[0].dataset_fingerprint is None
    assert report.cases[0].status == "pass"  # the injected eval still ran


# --- definition validation ---------------------------------------------------

def test_definition_rejects_empty_cases():
    with pytest.raises(ValidationError):
        SuiteDefinition(name="ok", cases=[])


def test_definition_rejects_bad_name():
    with pytest.raises(ValidationError):
        SuiteDefinition(name="bad name!", cases=[_case("c")])


def test_llm_judge_case_requires_judge_model():
    with pytest.raises(ValidationError):
        SuiteCase(name="c", schema="instruction", dataset_path="d", model="m", metric="llm_judge")


def test_load_suite_definition_and_bad_file(tmp_path: Path):
    good = tmp_path / "s.json"
    good.write_text(
        json.dumps({"name": "s", "cases": [{"name": "c", "schema": "instruction", "dataset_path": "d", "model": "m"}]}),
        encoding="utf-8",
    )
    definition = load_suite_definition(good)
    assert definition.name == "s" and definition.cases[0].schema_id == "instruction"

    bad = tmp_path / "bad.json"
    bad.write_text('{"name":"s","cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite_definition(bad)


def test_save_suite_report(tmp_path: Path):
    report = run_suite(_definition(_case("c")), lambda c: _report(avg=80))
    path = save_suite_report(tmp_path, report)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["suite"] == "my-suite"


# --- CLI (monkeypatched evaluate_case — no live backend) --------------------

def test_cli_suite_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_evaluate_suite_case", lambda case: _report(avg=80))
    suite = tmp_path / "s.json"
    suite.write_text(
        json.dumps({"name": "s", "cases": [{"name": "c", "schema": "instruction", "dataset_path": "d", "model": "m"}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["suite-run", str(suite), "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["suite"] == "s" and payload["overall_status"] == "pass"
    assert (tmp_path / "suite_reports" / "s.json").exists()


def test_cli_suite_run_strict_exits_2_on_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_evaluate_suite_case", lambda case: _report(avg=20))  # 20 < 70 -> block
    suite = tmp_path / "s.json"
    suite.write_text(
        json.dumps({"name": "s", "cases": [{"name": "c", "schema": "instruction", "dataset_path": "d", "model": "m"}]}),
        encoding="utf-8",
    )

    assert runner.invoke(app, ["suite-run", str(suite)]).exit_code == 0  # advisory by default
    assert runner.invoke(app, ["suite-run", str(suite), "--strict"]).exit_code == 2  # block under --strict


def test_cli_suite_run_bad_definition_exits_1(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert runner.invoke(app, ["suite-run", str(bad)]).exit_code == 1
