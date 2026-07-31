import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training import run_registry
from corpus_studio.training.run_registry import (
    INTERRUPTED,
    PREPARED,
    RUN_REGISTRY_DIRNAME,
    RUNNING,
    SUCCEEDED,
    TrainingRunRecord,
    list_run_records,
    load_run_record,
    mint_run_id,
    prepare_resumed_run,
    reconcile_running_records,
    record_path,
    save_run_record,
    validate_transition,
)

runner = CliRunner()


def _record(run_id: str, status: str = "prepared", pid: int | None = None) -> TrainingRunRecord:
    return TrainingRunRecord(
        run_id=run_id,
        created_at="2026-07-02T18:00:00Z",
        updated_at="2026-07-02T18:00:00Z",
        status=status,
        pid=pid,
    )


def test_save_load_roundtrip(tmp_path: Path):
    record = _record("20260702T180000-aaa", status="running", pid=123)
    path = save_run_record(tmp_path, record)
    reloaded = load_run_record(path)
    assert reloaded.run_id == record.run_id
    assert reloaded.status == "running"
    assert reloaded.pid == 123


def test_list_is_newest_first(tmp_path: Path):
    for run_id in ["20260702T180000-a", "20260702T190000-b", "20260702T170000-c"]:
        save_run_record(tmp_path, _record(run_id))
    ids = [r.run_id for r in list_run_records(tmp_path)]
    assert ids == ["20260702T190000-b", "20260702T180000-a", "20260702T170000-c"]


def test_list_skips_corrupt_files(tmp_path: Path):
    save_run_record(tmp_path, _record("20260702T180000-a"))
    (tmp_path / "training_runs" / "broken.json").write_text("{ not json", encoding="utf-8")
    assert len(list_run_records(tmp_path)) == 1


def test_mint_run_id_is_sortable():
    assert mint_run_id("20260702T183000", "ab12") == "20260702T183000-ab12"


# --- transition validation ---------------------------------------------------

def test_transition_rejects_leaving_terminal():
    with pytest.raises(ValueError):
        validate_transition(SUCCEEDED, RUNNING)


def test_transition_allows_running_to_failed():
    validate_transition(RUNNING, "failed")  # no raise


def test_transition_rejects_unknown_status():
    with pytest.raises(ValueError):
        validate_transition(RUNNING, "bogus")


def test_terminal_to_same_is_allowed():
    validate_transition(SUCCEEDED, SUCCEEDED)  # idempotent update, no raise


# --- crash reconciliation ----------------------------------------------------

def test_reconcile_flips_dead_running_to_interrupted():
    records = [_record("r1", status="running", pid=999), _record("r2", status="succeeded")]
    reconciled = reconcile_running_records(records, is_alive=lambda pid: False, updated_at="t2")
    assert reconciled[0].status == INTERRUPTED
    assert reconciled[0].updated_at == "t2"
    assert reconciled[1].status == "succeeded"  # terminal untouched


def test_reconcile_keeps_alive_running():
    records = [_record("r1", status="running", pid=123)]
    reconciled = reconcile_running_records(records, is_alive=lambda pid: True, updated_at="t2")
    assert reconciled[0].status == "running"


def test_reconcile_treats_pidless_running_as_interrupted():
    records = [_record("r1", status="running", pid=None)]
    reconciled = reconcile_running_records(records, is_alive=lambda pid: True, updated_at="t2")
    assert reconciled[0].status == INTERRUPTED


# --- CLI ---------------------------------------------------------------------

def test_cli_training_run_list_and_update(tmp_path: Path):
    save_run_record(tmp_path, _record("20260702T180000-a", status="running", pid=1))

    listed = runner.invoke(app, ["training-run-list", str(tmp_path)])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)["runs"][0]["run_id"] == "20260702T180000-a"

    updated = runner.invoke(
        app,
        ["training-run-update", str(tmp_path), "--run-id", "20260702T180000-a", "--status", "succeeded", "--exit-code", "0"],
    )
    assert updated.exit_code == 0, updated.output
    payload = json.loads(updated.output)
    assert payload["status"] == "succeeded"
    assert payload["exit_code"] == 0


def test_cli_update_writes_both_eval_links(tmp_path: Path):
    # S4b: before_eval_path is READ by the model-card before/after diff and the promote gate, but
    # training-run-update only wrote the 'after' side - so a diff could never show a baseline. Both
    # sides now write symmetrically (the 'before' model is the base_model, so no before-eval-model).
    save_run_record(tmp_path, _record("20260702T180000-a", status="running", pid=1))
    result = runner.invoke(
        app,
        [
            "training-run-update", str(tmp_path), "--run-id", "20260702T180000-a",
            "--before-eval-path", "eval/before.json", "--after-eval-path", "eval/after.json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["before_eval_path"] == "eval/before.json"
    assert payload["after_eval_path"] == "eval/after.json"
    # Persisted to the record, not merely echoed.
    persisted = load_run_record(record_path(tmp_path, "20260702T180000-a"))
    assert persisted.before_eval_path == "eval/before.json"
    assert persisted.after_eval_path == "eval/after.json"


def test_save_rejects_invalid_run_id(tmp_path: Path):
    # A run_id with a path separator/space would slug-collide with others.
    with pytest.raises(ValueError):
        save_run_record(tmp_path, _record("20260702T180000-a b"))
    with pytest.raises(ValueError):
        save_run_record(tmp_path, _record("bad/slashes"))


def test_cli_list_reconciles_dead_running(tmp_path: Path, monkeypatch):
    save_run_record(tmp_path, _record("20260702T180000-a", status="running", pid=4242))
    monkeypatch.setattr(run_registry, "pid_alive", lambda pid: False)

    result = runner.invoke(app, ["training-run-list", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["runs"][0]["status"] == "interrupted"
    # Reconciliation is persisted, not just displayed.
    assert load_run_record(record_path(tmp_path, "20260702T180000-a")).status == "interrupted"


def test_cli_list_keeps_alive_running(tmp_path: Path, monkeypatch):
    save_run_record(tmp_path, _record("20260702T180000-a", status="running", pid=1))
    monkeypatch.setattr(run_registry, "pid_alive", lambda pid: True)
    result = runner.invoke(app, ["training-run-list", str(tmp_path)])
    assert json.loads(result.output)["runs"][0]["status"] == "running"


def test_cli_update_rejects_illegal_transition(tmp_path: Path):
    save_run_record(tmp_path, _record("20260702T180000-a", status="succeeded"))
    result = runner.invoke(
        app,
        ["training-run-update", str(tmp_path), "--run-id", "20260702T180000-a", "--status", "running"],
    )
    assert result.exit_code == 1


# --- resume preparation (Phase A of the checkpoint/resume plan; #440/#486) -------------------------

def _sealed_resume_source(tmp_path: Path):
    """A valid sealed checkpoint bound to a demo plan - the reusable checkpoint-test builder + the
    canonical demo RunPlan. Returns (plan, checkpoint_dir)."""

    from test_platform_checkpoint import _build_sealed_checkpoint  # noqa: PLC0415

    from corpus_studio.platform.runners import demo_training_plan  # noqa: PLC0415

    plan = demo_training_plan(plan_id="demo-ckpt")
    checkpoint_dir = tmp_path / "c"
    _build_sealed_checkpoint(checkpoint_dir, plan=plan)
    return plan, checkpoint_dir


def test_prepare_resumed_run_records_lineage_and_mints_a_fresh_id(tmp_path: Path):
    plan, checkpoint_dir = _sealed_resume_source(tmp_path)
    project = tmp_path / "proj"
    record = prepare_resumed_run(
        project, plan, checkpoint_dir, resumed_run_id="20260731T000000-resume", now="2026-07-31T00:00:00Z"
    )
    assert record.status == PREPARED
    assert record.run_id == "20260731T000000-resume"
    assert record.resume_lineage is not None
    # The lineage is the parent's identity + the step the resume continues from (from the sealed state).
    assert record.resume_lineage.parent_run_id == "run-parent01"
    assert record.resume_lineage.resumed_from_global_step == 6
    # Persisted, and the lineage survives the JSON round-trip (a resumed run declares its parent).
    loaded = load_run_record(record_path(project, record.run_id))
    assert loaded.resume_lineage == record.resume_lineage


def test_prepare_resumed_run_fails_closed_on_an_incompatible_plan(tmp_path: Path):
    from corpus_studio.platform.checkpoint import CheckpointError  # noqa: PLC0415

    plan, checkpoint_dir = _sealed_resume_source(tmp_path)
    other = plan.model_copy(update={"plan_hash": "b" * 64})  # a different plan hash is incompatible
    project = tmp_path / "proj"
    with pytest.raises(CheckpointError):
        prepare_resumed_run(project, other, checkpoint_dir, resumed_run_id="20260731T000000-resume", now="t")
    # A refused resume writes NO record (fail closed leaves no half-prepared run behind).
    registry = project / RUN_REGISTRY_DIRNAME
    assert not registry.exists() or not list(registry.glob("*.json"))


def test_prepare_resumed_run_refuses_reusing_the_parent_run_id(tmp_path: Path):
    from corpus_studio.platform.checkpoint import CheckpointError  # noqa: PLC0415

    plan, checkpoint_dir = _sealed_resume_source(tmp_path)
    with pytest.raises(CheckpointError):
        # run-parent01 is the checkpoint's source run id; a resume must mint a fresh id, never reuse it.
        prepare_resumed_run(tmp_path / "proj", plan, checkpoint_dir, resumed_run_id="run-parent01", now="t")


def test_cli_resume_prepare_is_reachable_and_records_lineage(tmp_path: Path):
    plan, checkpoint_dir = _sealed_resume_source(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "training-run-resume-prepare", str(tmp_path / "proj"),
            "--plan", str(plan_path), "--checkpoint-dir", str(checkpoint_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == PREPARED
    assert payload["resume_lineage"]["parent_run_id"] == "run-parent01"


def test_cli_resume_prepare_fails_closed_on_a_corrupt_checkpoint(tmp_path: Path):
    plan, checkpoint_dir = _sealed_resume_source(tmp_path)
    # Corrupt a sealed member (optimizer.pt is mandatory) so integrity verification refuses it.
    (checkpoint_dir / "optimizer.pt").write_bytes(b"tampered")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "training-run-resume-prepare", str(tmp_path / "proj"),
            "--plan", str(plan_path), "--checkpoint-dir", str(checkpoint_dir),
        ],
    )
    assert result.exit_code == 1
    assert "NOT a compatible resume source" in result.output
