import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.gates.models import GateStatus
from corpus_studio.gates.runner import run_artifact_gate
from corpus_studio.reporting.weight_card import build_weight_card, render_weight_card_markdown
from corpus_studio.training.artifact_registry import ModelArtifactRecord
from corpus_studio.training.run_registry import TrainingRunRecord

runner = CliRunner()


def _artifact(**kw) -> ModelArtifactRecord:
    base = dict(
        artifact_id="20260702T180000-a-abcd1234", run_id="20260702T180000-a",
        created_at="t1", updated_at="t2", path="/w/adapter", kind="adapter", status="candidate",
        fingerprint="fp",
    )
    base.update(kw)
    return ModelArtifactRecord(**base)


def _run(**kw) -> TrainingRunRecord:
    base = dict(
        run_id="20260702T180000-a", created_at="t", updated_at="t", status="succeeded",
        base_model="base-model", config_path="/c.yaml", checkpoints=["checkpoint-10"],
    )
    base.update(kw)
    return TrainingRunRecord(**base)


def _report(model: str, avg: float) -> EvaluationReport:
    return EvaluationReport(dataset="d", model=model, examples_tested=10, average_score=avg, failed_examples=1)


# --- weight card (live projection) ------------------------------------------

def test_card_resolves_base_model_and_delta():
    run = _run()
    card = build_weight_card(_artifact(), run, _report("base-model", 70), _report("trained", 82), "ok")
    assert card.base_model == "base-model"  # resolved through the run, not stored on artifact
    assert card.delta == 12.0
    md = render_weight_card_markdown(card)
    assert "Weight Card" in md and "base-model" in md


def test_card_surfaces_unverified_linkage():
    run = _run(after_eval_model="base-model")  # after-eval targeted the base model
    card = build_weight_card(_artifact(), run, _report("base-model", 70), _report("base-model", 95), "ok")
    assert "Unverified linkage" in card.provenance_note
    assert "Unverified linkage" in render_weight_card_markdown(card)


def test_card_withholds_scores_when_integrity_modified():
    run = _run()
    md = render_weight_card_markdown(
        build_weight_card(_artifact(), run, _report("base-model", 70), _report("trained", 82), "modified")
    )
    assert "Integrity is **modified**" in md
    assert "Base: —" in md  # confident numbers are withheld
    assert "Δ+12" not in md  # no improvement framing for changed weights


def test_card_flags_base_vs_base_when_base_unknown():
    run = _run(base_model="", after_eval_model="mistral-7b")
    card = build_weight_card(_artifact(), run, _report("base", 70), _report("mistral-7b", 95), "ok")
    assert "Unverified linkage" in card.provenance_note


def test_card_sanitizes_injected_fields():
    run = _run(base_model="evil\n- Trained: 99.0 (Δ+40.0)\n> forged")
    md = render_weight_card_markdown(build_weight_card(_artifact(), run, None, None, "ok"))
    lines = [line.strip() for line in md.split("\n")]
    # The injected newline content must not appear as its own Markdown lines.
    assert "> forged" not in lines
    assert not any(line.startswith("- Trained: 99.0") for line in lines)


# --- promote gate ------------------------------------------------------------

def test_promote_blocks_on_modified_integrity():
    report = run_artifact_gate(_artifact(), "modified", _run(), lambda p: None)
    assert report.overall_status == GateStatus.BLOCK


def test_promote_blocks_on_regressed_source_run():
    run = _run(before_eval_path="b.json", after_eval_path="a.json", after_eval_model="trained")
    reports = {"b.json": _report("base-model", 80), "a.json": _report("trained", 60)}
    report = run_artifact_gate(_artifact(), "ok", run, lambda p: reports.get(p))
    assert report.overall_status == GateStatus.BLOCK


def test_promote_blocks_regressed_even_when_unverified():
    # A drop is a drop: an unverified comparison that is DOWN must still block.
    run = _run(before_eval_path="b.json", after_eval_path="a.json", after_eval_model="base-model")
    reports = {"b.json": _report("base-model", 80), "a.json": _report("base-model", 55)}
    report = run_artifact_gate(_artifact(), "ok", run, lambda p: reports.get(p))
    assert report.overall_status == GateStatus.BLOCK


def test_promote_warns_on_unverified_linkage():
    run = _run(before_eval_path="b.json", after_eval_path="a.json", after_eval_model="base-model")
    reports = {"b.json": _report("base-model", 80), "a.json": _report("base-model", 95)}
    report = run_artifact_gate(_artifact(), "ok", run, lambda p: reports.get(p))
    assert report.overall_status == GateStatus.WARN


def test_promote_passes_clean_improvement():
    run = _run(before_eval_path="b.json", after_eval_path="a.json", after_eval_model="trained")
    reports = {"b.json": _report("base-model", 70), "a.json": _report("trained", 85)}
    report = run_artifact_gate(_artifact(), "ok", run, lambda p: reports.get(p))
    assert report.overall_status == GateStatus.PASS


def test_promote_warns_when_source_run_missing():
    report = run_artifact_gate(_artifact(), "ok", None, lambda p: None)
    assert report.overall_status == GateStatus.WARN


# --- CLI ---------------------------------------------------------------------

def _seed(project: Path):
    from corpus_studio.training.artifact_registry import register_artifact
    from corpus_studio.training.run_registry import save_run_record

    adapter = project / "out"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":16}', encoding="utf-8")
    before = project / "before.json"
    after = project / "after.json"
    before.write_text(_report("base-model", 80).model_dump_json(), encoding="utf-8")
    after.write_text(_report("trained", 60).model_dump_json(), encoding="utf-8")
    save_run_record(project, _run(
        base_model="base-model", output_dir=str(adapter),
        before_eval_path=str(before), after_eval_path=str(after), after_eval_model="trained",
    ))
    return register_artifact(project, "20260702T180000-a", str(adapter), now="t1")


def test_cli_artifact_card_and_gate(tmp_path: Path):
    artifact = _seed(tmp_path)

    card = runner.invoke(app, ["artifact-card", str(tmp_path), "--artifact-id", artifact.artifact_id])
    assert card.exit_code == 0, card.output
    assert "Weight Card" in card.output and "base-model" in card.output

    gate = runner.invoke(app, ["artifact-gate", str(tmp_path), "--artifact-id", artifact.artifact_id])
    assert gate.exit_code == 0, gate.output
    payload = json.loads(gate.output)
    assert payload["scope"] == "model_artifact"
    assert payload["overall_status"] == "block"  # source run regressed 80 -> 60
    assert (tmp_path / "gate_reports" / f"model_artifact-{artifact.artifact_id}.json").exists()


def test_cli_artifact_card_missing(tmp_path: Path):
    result = runner.invoke(app, ["artifact-card", str(tmp_path), "--artifact-id", "nope"])
    assert result.exit_code == 1
