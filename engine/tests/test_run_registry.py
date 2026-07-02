import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training.run_registry import (
    INTERRUPTED,
    RUNNING,
    SUCCEEDED,
    TrainingRunRecord,
    list_run_records,
    load_run_record,
    mint_run_id,
    reconcile_running_records,
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


def test_cli_update_rejects_illegal_transition(tmp_path: Path):
    save_run_record(tmp_path, _record("20260702T180000-a", status="succeeded"))
    result = runner.invoke(
        app,
        ["training-run-update", str(tmp_path), "--run-id", "20260702T180000-a", "--status", "running"],
    )
    assert result.exit_code == 1
