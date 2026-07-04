import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.gates.basic_gates import regression_gate
from corpus_studio.gates.models import GateStatus, GateThresholds
from corpus_studio.gates.runner import run_training_run_gate
from corpus_studio.training.run_registry import TrainingRunRecord, save_run_record

runner = CliRunner()
THRESHOLDS = GateThresholds()


def _report(model: str, avg: float) -> EvaluationReport:
    return EvaluationReport(dataset="d", model=model, examples_tested=10, average_score=avg, failed_examples=1)


# --- pure gate --------------------------------------------------------------

def test_regression_blocks_on_score_drop():
    result = regression_gate(_report("base", 80), _report("trained", 70), THRESHOLDS, provenance_ok=True)
    assert result.status == GateStatus.BLOCK
    assert "regressed" in result.message


def test_regression_passes_on_improvement():
    result = regression_gate(_report("base", 70), _report("trained", 82), THRESHOLDS, provenance_ok=True)
    assert result.status == GateStatus.PASS
    assert "improved" in result.message


def test_regression_passes_within_tolerance():
    # 1.0 drop < 2.0 default tolerance -> not a regression.
    result = regression_gate(_report("base", 80), _report("trained", 79), THRESHOLDS, provenance_ok=True)
    assert result.status == GateStatus.PASS


def test_regression_warns_on_unverified_linkage():
    result = regression_gate(_report("base", 80), _report("trained", 95), THRESHOLDS, provenance_ok=False)
    assert result.status == GateStatus.WARN
    assert "Unverified linkage" in result.message


def test_regression_warns_on_missing_link():
    result = regression_gate(_report("base", 80), None, THRESHOLDS, provenance_ok=True)
    assert result.status == GateStatus.WARN
    assert "Cannot gate" in result.message


# --- metric-aware comparison (Tier-3 hardening) -----------------------------

def _report_metric(model: str, avg: float, metric: str) -> EvaluationReport:
    return EvaluationReport(
        dataset="d", model=model, examples_tested=10, average_score=avg,
        failed_examples=1, metric=metric,
    )


def test_regression_warns_when_metrics_differ():
    # A keyword-overlap baseline vs an LLM-judge after-eval are non-comparable scales;
    # the -10 "delta" must not be trusted to block or pass.
    result = regression_gate(
        _report_metric("base", 80, "keyword_overlap"),
        _report_metric("trained", 70, "llm_judge"),
        THRESHOLDS,
        provenance_ok=True,
    )
    assert result.status == GateStatus.WARN
    assert "different metrics" in result.message
    assert "keyword-overlap" in result.message and "LLM-judge" in result.message


def test_regression_block_message_uses_actual_metric_label():
    # Matching llm_judge metric with a real drop -> BLOCK, labelled LLM-judge (not the
    # old hardcoded "keyword-overlap").
    result = regression_gate(
        _report_metric("base", 80, "llm_judge"),
        _report_metric("trained", 60, "llm_judge"),
        THRESHOLDS,
        provenance_ok=True,
    )
    assert result.status == GateStatus.BLOCK
    assert "LLM-judge score dropped" in result.message
    assert "keyword-overlap" not in result.message


def test_regression_pass_caveat_only_for_keyword_overlap():
    # The lexical-proxy caveat is specific to keyword-overlap; an LLM-judge pass omits it.
    kw = regression_gate(
        _report_metric("b", 70, "keyword_overlap"),
        _report_metric("t", 82, "keyword_overlap"),
        THRESHOLDS,
        provenance_ok=True,
    )
    judge = regression_gate(
        _report_metric("b", 70, "llm_judge"),
        _report_metric("t", 82, "llm_judge"),
        THRESHOLDS,
        provenance_ok=True,
    )
    assert kw.status == GateStatus.PASS and "lexical proxy" in kw.message
    assert judge.status == GateStatus.PASS and "lexical proxy" not in judge.message
    assert "LLM-judge score improved" in judge.message


# --- runner + provenance ----------------------------------------------------

def _record(**kwargs) -> TrainingRunRecord:
    base = dict(
        run_id="20260702T180000-a",
        created_at="t",
        updated_at="t",
        status="succeeded",
        base_model="base-model",
    )
    base.update(kwargs)
    return TrainingRunRecord(**base)


def test_run_gate_flags_base_vs_base_as_unverified():
    reports = {"before.json": _report("base-model", 80), "after.json": _report("base-model", 95)}
    record = _record(before_eval_path="before.json", after_eval_path="after.json", after_eval_model="base-model")
    report = run_training_run_gate(record, lambda p: reports.get(p), THRESHOLDS)
    assert report.overall_status == GateStatus.WARN  # after-eval targeted the base model


def test_run_gate_blocks_real_regression():
    reports = {"before.json": _report("base-model", 80), "after.json": _report("trained-adapter", 70)}
    record = _record(before_eval_path="before.json", after_eval_path="after.json", after_eval_model="trained-adapter")
    report = run_training_run_gate(record, lambda p: reports.get(p), THRESHOLDS)
    assert report.overall_status == GateStatus.BLOCK


def test_run_gate_warns_when_no_after_model():
    reports = {"before.json": _report("base-model", 80), "after.json": _report("trained", 90)}
    record = _record(before_eval_path="before.json", after_eval_path="after.json", after_eval_model=None)
    report = run_training_run_gate(record, lambda p: reports.get(p), THRESHOLDS)
    assert report.overall_status == GateStatus.WARN  # can't verify provenance


def test_run_gate_rejects_spoofed_after_eval_label():
    # SPOOF: the label claims "trained-adapter" and the score improved (95 > 80), but the
    # linked report actually evaluated the BASE model. Before the fix this passed as a
    # trained-vs-base improvement; provenance must now be unverified.
    reports = {"before.json": _report("base-model", 80), "after.json": _report("base-model", 95)}
    record = _record(
        before_eval_path="before.json",
        after_eval_path="after.json",
        after_eval_model="trained-adapter",  # label disagrees with the report's model
    )
    report = run_training_run_gate(record, lambda p: reports.get(p), THRESHOLDS)
    assert report.overall_status == GateStatus.WARN  # provenance mismatch -> not a trusted pass


# --- CLI --------------------------------------------------------------------

def test_cli_training_run_gate(tmp_path: Path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(_report("base-model", 80).model_dump_json(), encoding="utf-8")
    after.write_text(_report("trained-adapter", 70).model_dump_json(), encoding="utf-8")
    save_run_record(
        tmp_path,
        _record(
            before_eval_path=str(before),
            after_eval_path=str(after),
            after_eval_model="trained-adapter",
        ),
    )

    result = runner.invoke(app, ["training-run-gate", str(tmp_path), "--run-id", "20260702T180000-a"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["scope"] == "training_run"
    assert payload["overall_status"] == "block"
    assert (tmp_path / "gate_reports" / "training_run-20260702T180000-a.json").exists()


def test_cli_training_run_gate_missing_run(tmp_path: Path):
    result = runner.invoke(app, ["training-run-gate", str(tmp_path), "--run-id", "nope"])
    assert result.exit_code == 1
