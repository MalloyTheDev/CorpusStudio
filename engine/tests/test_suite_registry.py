import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import corpus_studio.cli as cli
from corpus_studio.cli import app
from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.suites.registry import (
    list_suite_definitions,
    load_suite_by_name,
    scaffold_suite,
    suite_definition_path,
)
from corpus_studio.suites.runner import load_suite_definition

runner = CliRunner()


def _report(avg: float = 80.0) -> EvaluationReport:
    return EvaluationReport(
        dataset="d", model="m", examples_tested=10, average_score=avg, failed_examples=1, metric="keyword_overlap"
    )


# --- registry helpers -------------------------------------------------------

def test_scaffold_creates_valid_definition(tmp_path: Path):
    path = scaffold_suite(tmp_path, "my-suite")
    assert path == tmp_path / "evaluation_suites" / "my-suite.json"
    definition = load_suite_definition(path)
    assert definition.name == "my-suite" and len(definition.cases) == 1


def test_scaffold_refuses_overwrite_unless_force(tmp_path: Path):
    scaffold_suite(tmp_path, "s")
    with pytest.raises(FileExistsError):
        scaffold_suite(tmp_path, "s")
    assert scaffold_suite(tmp_path, "s", force=True).exists()  # --force overwrites


@pytest.mark.parametrize("bad", ["../evil", "a/b", "bad name", "", "x/../y"])
def test_bad_names_rejected(tmp_path: Path, bad: str):
    with pytest.raises(ValueError):
        suite_definition_path(tmp_path, bad)
    with pytest.raises(ValueError):
        scaffold_suite(tmp_path, bad)


def test_list_names_counts_sorted(tmp_path: Path):
    scaffold_suite(tmp_path, "bbb")
    scaffold_suite(tmp_path, "aaa")
    summaries = list_suite_definitions(tmp_path)
    assert [summary.name for summary in summaries] == ["aaa", "bbb"]
    assert all(summary.valid and summary.case_count == 1 for summary in summaries)


def test_list_tolerates_corrupt_file(tmp_path: Path):
    scaffold_suite(tmp_path, "good")
    (tmp_path / "evaluation_suites" / "broken.json").write_text("{ not json", encoding="utf-8")
    by_name = {summary.name: summary for summary in list_suite_definitions(tmp_path)}
    assert by_name["good"].valid is True
    assert by_name["broken"].valid is False and by_name["broken"].error


def test_list_empty_registry(tmp_path: Path):
    assert list_suite_definitions(tmp_path) == []


def test_load_by_name_and_missing(tmp_path: Path):
    scaffold_suite(tmp_path, "s")
    assert load_suite_by_name(tmp_path, "s").name == "s"
    with pytest.raises(FileNotFoundError):
        load_suite_by_name(tmp_path, "nope")


# --- CLI --------------------------------------------------------------------

def test_cli_suite_init_and_collision(tmp_path: Path):
    ok = runner.invoke(app, ["suite-init", "s", "--project-dir", str(tmp_path)])
    assert ok.exit_code == 0
    assert (tmp_path / "evaluation_suites" / "s.json").exists()

    dup = runner.invoke(app, ["suite-init", "s", "--project-dir", str(tmp_path)])
    assert dup.exit_code == 1 and "force" in dup.stderr.lower()

    forced = runner.invoke(app, ["suite-init", "s", "--project-dir", str(tmp_path), "--force"])
    assert forced.exit_code == 0


def test_cli_suite_init_bad_name_exits_1(tmp_path: Path):
    assert runner.invoke(app, ["suite-init", "../evil", "--project-dir", str(tmp_path)]).exit_code == 1


def test_cli_suite_list(tmp_path: Path):
    runner.invoke(app, ["suite-init", "s", "--project-dir", str(tmp_path)])
    human = runner.invoke(app, ["suite-list", "--project-dir", str(tmp_path)])
    assert human.exit_code == 0 and "1 case(s)" in human.stdout

    as_json = runner.invoke(app, ["suite-list", "--project-dir", str(tmp_path), "--json"])
    assert json.loads(as_json.stdout)[0]["name"] == "s"


def test_cli_suite_list_empty(tmp_path: Path):
    result = runner.invoke(app, ["suite-list", "--project-dir", str(tmp_path)])
    assert result.exit_code == 0 and "No suites defined" in result.stdout


def test_cli_suite_run_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_evaluate_suite_case", lambda case, project_dir=None: _report(avg=80))
    runner.invoke(app, ["suite-init", "s", "--project-dir", str(tmp_path)])
    result = runner.invoke(app, ["suite-run", "s", "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["suite"] == "s" and payload["overall_status"] == "pass"


def test_cli_suite_run_unknown_name_exits_1(tmp_path: Path):
    assert runner.invoke(app, ["suite-run", "ghost", "--project-dir", str(tmp_path)]).exit_code == 1


def test_cli_suite_run_name_without_project_dir_exits_1(tmp_path: Path):
    assert runner.invoke(app, ["suite-run", "ghost"]).exit_code == 1  # not a file + no --project-dir


def test_cli_suite_run_by_path_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_evaluate_suite_case", lambda case, project_dir=None: _report(avg=80))
    suite = tmp_path / "s.json"
    suite.write_text(
        json.dumps({"name": "s", "cases": [{"name": "c", "schema": "instruction", "dataset_path": "d", "model": "m"}]}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["suite-run", str(suite)])
    assert result.exit_code == 0, result.output
