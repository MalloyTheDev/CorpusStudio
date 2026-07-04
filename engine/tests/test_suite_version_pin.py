import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import corpus_studio.cli as cli
from corpus_studio.cli import app
from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.suites.models import SuiteCase, SuiteDefinition
from corpus_studio.suites.runner import run_suite
from corpus_studio.versions.version_restore import reconstruct_version_lines

runner = CliRunner()

ROWS = [{"instruction": "A", "output": "1"}, {"instruction": "B", "output": "2"}]


def _make_version(project: Path, *extra: str) -> str:
    (project / "examples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8"
    )
    result = runner.invoke(app, ["dataset-version-create", str(project), *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["version_id"]


# --- reconstruct_version_lines (the reusable helper) -------------------------

def test_reconstruct_version_lines_returns_verified_rows(tmp_path: Path):
    version_id = _make_version(tmp_path, "--store-rows")
    lines = reconstruct_version_lines(tmp_path, version_id)
    parsed = [json.loads(line) for line in lines]
    assert {row["instruction"] for row in parsed} == {"A", "B"}


def test_reconstruct_unknown_version_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        reconstruct_version_lines(tmp_path, "nope-123")


def test_reconstruct_version_without_stored_rows_raises(tmp_path: Path):
    version_id = _make_version(tmp_path, "--no-store-rows")
    with pytest.raises(ValueError):
        reconstruct_version_lines(tmp_path, version_id)


# --- model: exactly one dataset source --------------------------------------

def test_case_requires_exactly_one_source():
    with pytest.raises(ValidationError):  # both
        SuiteCase(name="c", schema="instruction", dataset_path="d", version_id="v", model="m")
    with pytest.raises(ValidationError):  # neither
        SuiteCase(name="c", schema="instruction", model="m")

    case = SuiteCase(name="c", schema="instruction", version_id="v", model="m")
    assert case.version_id == "v" and case.dataset_path is None


# --- runner echoes version_id, no path-fingerprint for a version case -------

def test_run_suite_echoes_version_id_and_skips_fingerprint():
    case = SuiteCase(name="pinned", schema="instruction", version_id="20260101T000000-a", model="m")
    definition = SuiteDefinition(name="s", cases=[case])
    report = EvaluationReport(dataset="d", model="m", examples_tested=10, average_score=80.0, failed_examples=1)

    result = run_suite(definition, lambda case: report).cases[0]
    assert result.version_id == "20260101T000000-a"
    assert result.dataset_fingerprint is None  # version case: version_id is the repro record
    assert result.status == "pass"


# --- CLI wiring: version case reconstructs to temp + isolates failures -------

def test_cli_version_case_reconstructs_to_temp_and_evaluates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    version_id = _make_version(tmp_path, "--store-rows")
    captured: dict = {}

    def fake_eval(case, dataset_path):
        captured["rows"] = Path(dataset_path).read_text(encoding="utf-8")
        return EvaluationReport(dataset="d", model="m", examples_tested=2, average_score=90.0, failed_examples=0)

    monkeypatch.setattr(cli, "_evaluate_suite_dataset", fake_eval)
    suite = tmp_path / "s.json"
    suite.write_text(
        json.dumps({"name": "s", "cases": [{"name": "c", "schema": "instruction", "version_id": version_id, "model": "m"}]}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["suite-run", str(suite), "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["cases"][0]["status"] == "pass"
    assert payload["cases"][0]["version_id"] == version_id
    # the temp file handed to the evaluator held the reconstructed version rows
    assert "A" in captured["rows"] and "B" in captured["rows"]


def test_cli_version_case_without_project_dir_is_isolated_error(tmp_path: Path):
    suite = tmp_path / "s.json"
    suite.write_text(
        json.dumps({"name": "s", "cases": [{"name": "c", "schema": "instruction", "version_id": "v1", "model": "m"}]}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["suite-run", str(suite)])  # path mode, no --project-dir
    assert result.exit_code == 0  # advisory
    payload = json.loads(result.stdout)
    assert payload["cases"][0]["status"] == "error"  # isolated, not a crash
    assert "project-dir" in (payload["cases"][0]["error"] or "")
