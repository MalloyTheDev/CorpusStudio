"""Streaming progress callback for long evaluation runs (#191, engine slice)."""

from corpus_studio.evaluation.evaluator import (
    EvaluationRunConfig,
    extract_evaluation_examples,
    run_evaluation,
)
from corpus_studio.model_backends.base import BackendGenerateResponse


class _OkBackend:
    def generate(self, request):
        return BackendGenerateResponse(text="A function that calls itself.", model_name="fake")


class _BrokenBackend:
    def generate(self, request):
        raise OSError("backend down")  # a BACKEND_ERROR_TYPES member — isolated per row


def _examples(count: int):
    rows = [
        {"instruction": f"Q{i}", "input": "", "output": "A function that calls itself.", "tags": ["t"]}
        for i in range(count)
    ]
    return extract_evaluation_examples(rows, "instruction")


def _config() -> EvaluationRunConfig:
    return EvaluationRunConfig(
        dataset="d",
        model="fake",
        schema_id="instruction",
        backend="ollama",
        base_url="http://localhost:11434",
        score_threshold=60.0,
        timeout_seconds=30,
    )


def test_progress_callback_fires_once_per_example_with_running_count():
    calls: list[tuple[int, int]] = []
    run_evaluation(_config(), _examples(3), _OkBackend(), progress_callback=lambda c, n: calls.append((c, n)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_progress_total_respects_limit():
    calls: list[tuple[int, int]] = []
    run_evaluation(_config(), _examples(5), _OkBackend(), limit=2, progress_callback=lambda c, n: calls.append((c, n)))
    assert calls == [(1, 2), (2, 2)]


def test_progress_fires_even_when_a_row_isolates_a_backend_error():
    calls: list[tuple[int, int]] = []
    report = run_evaluation(_config(), _examples(2), _BrokenBackend(), progress_callback=lambda c, n: calls.append((c, n)))
    # Both rows are still counted (each recorded as a scored-0 failure) and progressed.
    assert calls == [(1, 2), (2, 2)]
    assert report.examples_tested == 2
    assert report.failed_examples == 2


def test_progress_callback_errors_do_not_abort_the_run():
    def boom(_completed: int, _total: int) -> None:
        raise RuntimeError("sink exploded")

    # A raising progress sink must not break the evaluation — the report still completes.
    report = run_evaluation(_config(), _examples(2), _OkBackend(), progress_callback=boom)
    assert report.examples_tested == 2


def test_no_callback_still_runs():
    report = run_evaluation(_config(), _examples(2), _OkBackend())
    assert report.examples_tested == 2


# --- CLI wiring: --progress passes a callback (and emits without error) ------


def _run_eval_cli(tmp_path, monkeypatch, extra_args):
    from typer.testing import CliRunner

    from corpus_studio import cli
    from corpus_studio.evaluation.reports import EvaluationReport

    input_path = tmp_path / "d.jsonl"
    input_path.write_text(
        '{"instruction":"Explain recursion in detail.","input":"",'
        '"output":"A function that calls itself."}\n',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_build_backend(**kwargs):
        return _OkBackend()

    def fake_run_evaluation(config, examples, backend, limit=None, scorer=None, progress_callback=None):
        captured["cb"] = progress_callback
        if progress_callback is not None:
            progress_callback(1, 1)  # exercise the CLI's stderr sink without error
        return EvaluationReport.from_results(
            dataset="d",
            model="fake",
            results=[],
            run_settings=config.to_report_settings(),
            metric="keyword_overlap",
        )

    monkeypatch.setattr(cli, "_build_backend", fake_build_backend)
    monkeypatch.setattr(cli, "run_evaluation", fake_run_evaluation)

    result = CliRunner().invoke(
        cli.app, ["eval-run", str(input_path), "instruction", "--model", "x", *extra_args]
    )
    return result, captured


def test_cli_progress_flag_passes_a_callback(tmp_path, monkeypatch):
    result, captured = _run_eval_cli(tmp_path, monkeypatch, ["--progress"])
    assert result.exit_code == 0, result.output
    assert captured["cb"] is not None


def test_cli_without_progress_flag_passes_no_callback(tmp_path, monkeypatch):
    result, captured = _run_eval_cli(tmp_path, monkeypatch, [])
    assert result.exit_code == 0, result.output
    assert captured["cb"] is None
