import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.evaluation.benchmark import build_benchmark_report
from corpus_studio.evaluation.reports import EvaluationExampleResult, EvaluationReport
from corpus_studio.model_backends.base import BackendGenerateResponse

runner = CliRunner()


def _result(example_id: str, score: float, passed: bool) -> EvaluationExampleResult:
    return EvaluationExampleResult(
        example_id=example_id,
        prompt="p",
        expected_output="e",
        model_output="m",
        score=score,
        passed=passed,
    )


def _report(model: str, results: list[EvaluationExampleResult]) -> EvaluationReport:
    return EvaluationReport.from_results(dataset="d", model=model, results=results)


def test_ranks_models_by_average_score():
    good = _report("good", [_result("row-1", 90, True), _result("row-2", 80, True)])
    bad = _report("bad", [_result("row-1", 40, False), _result("row-2", 30, False)])
    report = build_benchmark_report("d", [bad, good])  # unsorted input
    assert [summary.model for summary in report.models] == ["good", "bad"]
    assert report.models[0].rank == 1
    assert report.best_model == "good"
    assert report.worst_model == "bad"


def test_score_delta_and_spread():
    good = _report("good", [_result("row-1", 100, True)])
    mid = _report("mid", [_result("row-1", 60, False)])
    report = build_benchmark_report("d", [good, mid])
    deltas = {summary.model: summary.score_delta_vs_best for summary in report.models}
    assert deltas["good"] == 0.0
    assert deltas["mid"] == -40.0
    assert report.score_spread == 40.0


def test_commonly_failed_examples_is_intersection():
    a = _report("a", [_result("row-1", 20, False), _result("row-2", 90, True)])
    b = _report("b", [_result("row-1", 30, False), _result("row-2", 40, False)])
    report = build_benchmark_report("d", [a, b])
    # row-1 failed by both; row-2 only by b.
    assert report.commonly_failed_examples == ["row-1"]


def test_pass_rate_computed():
    report = build_benchmark_report(
        "d", [_report("m", [_result("row-1", 90, True), _result("row-2", 40, False)])]
    )
    assert report.models[0].pass_rate == 50.0


def test_single_model_report():
    report = build_benchmark_report("d", [_report("m", [_result("row-1", 90, True)])])
    assert report.model_count == 1
    assert report.best_model == "m"
    assert report.worst_model == "m"
    assert report.score_spread == 0.0


def test_empty_reports():
    report = build_benchmark_report("d", [])
    assert report.model_count == 0
    assert report.models == []


class _ScriptedBackend:
    def __init__(self, text: str):
        self._text = text

    def generate(self, request):
        return BackendGenerateResponse(text=self._text, model_name="fake")


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_benchmark_ranks_models(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "eval.jsonl"
    _write(
        dataset,
        [{"instruction": "Explain recursion.", "output": "Recursion is a function calling itself."}],
    )

    def fake_build_backend(backend, model, base_url, api_key, timeout_seconds):
        if model == "good":
            return _ScriptedBackend("Recursion is a function calling itself.")
        return _ScriptedBackend("banana bread")

    monkeypatch.setattr("corpus_studio.cli._build_backend", fake_build_backend)

    result = runner.invoke(
        app,
        ["benchmark", str(dataset), "instruction", "--model", "bad", "--model", "good"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    benchmark = payload["benchmark"]
    assert benchmark["model_count"] == 2
    assert benchmark["best_model"] == "good"
    assert benchmark["models"][0]["model"] == "good"
    assert benchmark["models"][0]["rank"] == 1
    assert len(payload["model_reports"]) == 2


def test_cli_benchmark_requires_a_model(tmp_path: Path):
    dataset = tmp_path / "eval.jsonl"
    _write(dataset, [{"instruction": "x", "output": "y"}])
    result = runner.invoke(app, ["benchmark", str(dataset), "instruction", "--model", "   "])
    assert result.exit_code == 1
    assert "at least one --model" in result.output
