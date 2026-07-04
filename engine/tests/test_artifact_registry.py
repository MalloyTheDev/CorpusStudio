import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.training.artifact_registry import (
    MISSING,
    MODIFIED,
    OK,
    artifact_content_integrity,
    artifact_integrity,
    compute_fingerprint,
    list_artifacts,
    load_artifact_record,
    make_artifact_id,
    register_artifact,
    update_artifact_status,
)
from corpus_studio.training.run_registry import TrainingRunRecord, save_run_record

runner = CliRunner()


def _make_run(project: Path, run_id: str = "20260702T180000-a", output_dir: str = "") -> None:
    save_run_record(
        project,
        TrainingRunRecord(
            run_id=run_id, created_at="t", updated_at="t", status="succeeded",
            base_model="Qwen/Qwen2.5-Coder-7B", output_dir=output_dir,
        ),
    )


def _adapter_dir(tmp: Path) -> Path:
    d = tmp / "out" / "adapter"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (d / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    return d


# --- register + idempotency --------------------------------------------------

def test_register_stores_only_non_derivable_fields(tmp_path: Path):
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")

    assert record.run_id == "20260702T180000-a"
    assert record.status == "candidate"
    assert record.fingerprint is not None
    dumped = record.model_dump()
    # base_model / eval_score are resolved through the run, never stored here.
    assert "base_model" not in dumped
    assert "eval_score" not in dumped


def test_register_is_idempotent(tmp_path: Path):
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    first = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    update_artifact_status(tmp_path, first.artifact_id, "kept", now="t2")

    # Re-register the same run+path: same file, preserves created_at + status.
    again = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t3")
    assert again.artifact_id == first.artifact_id
    assert again.created_at == "t1"  # preserved
    assert again.status == "kept"  # preserved
    assert again.updated_at == "t3"  # refreshed
    assert len(list_artifacts(tmp_path)) == 1


def test_register_requires_existing_run(tmp_path: Path):
    with pytest.raises(ValueError):
        register_artifact(tmp_path, "nope", str(tmp_path), now="t")


def test_make_artifact_id_is_deterministic_and_id_safe(tmp_path: Path):
    a = make_artifact_id("run-1", str(tmp_path / "x"))
    b = make_artifact_id("run-1", str(tmp_path / "x"))
    assert a == b
    assert a.startswith("run-1-")


# --- integrity ---------------------------------------------------------------

def test_integrity_ok_missing_modified(tmp_path: Path):
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    assert artifact_integrity(record) == OK

    # Modify the descriptor file -> modified.
    (adapter / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    assert artifact_integrity(record) == MODIFIED

    # Remove the path -> missing.
    import shutil

    shutil.rmtree(adapter)
    assert artifact_integrity(record) == MISSING


def test_fingerprint_null_integrity_is_ok(tmp_path: Path):
    # An empty directory has no descriptor/file -> null fingerprint -> ok (no false alarm).
    empty = tmp_path / "empty"
    empty.mkdir()
    assert compute_fingerprint(str(empty)) is None


def test_file_fingerprint_form(tmp_path: Path):
    f = tmp_path / "adapter.safetensors"
    f.write_text("w", encoding="utf-8")
    fp = compute_fingerprint(str(f))
    assert fp is not None and ":" in fp and "=" not in fp


def test_cheap_fingerprint_now_covers_weight_files(tmp_path: Path):
    # The old fingerprint only stat'd the descriptor; a resized weight went undetected.
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    assert artifact_integrity(record) == OK

    # Overwrite the weight with different-size bytes (descriptor untouched) -> modified.
    (adapter / "adapter_model.safetensors").write_text("different weights", encoding="utf-8")
    assert artifact_integrity(record) == MODIFIED


def test_content_integrity_catches_byte_swap_that_preserves_size_and_mtime(tmp_path: Path):
    import os

    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)  # adapter_model.safetensors = "weights" (7 bytes)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    assert artifact_content_integrity(record) == OK

    weight = adapter / "adapter_model.safetensors"
    original = weight.stat()
    # Swap in different bytes of the SAME length, then restore the original mtime — this fools
    # the cheap size+mtime fingerprint but not the byte-exact content hash.
    weight.write_text("XXXXXXX", encoding="utf-8")  # same 7-byte length
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert artifact_integrity(record) == OK               # cheap check is fooled
    assert artifact_content_integrity(record) == MODIFIED  # byte-exact catches it


def test_content_integrity_missing_when_path_gone(tmp_path: Path):
    import shutil

    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    shutil.rmtree(adapter)
    assert artifact_content_integrity(record) == MISSING


def test_content_integrity_legacy_record_falls_back_to_cheap(tmp_path: Path):
    # A record from before content hashing (content_hash=None) still verifies via size+mtime.
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    legacy = record.model_copy(update={"content_hash": None})
    assert artifact_content_integrity(legacy) == OK

    (adapter / "adapter_model.safetensors").write_text("resized weights", encoding="utf-8")
    assert artifact_content_integrity(legacy) == MODIFIED  # falls back to size+mtime


# --- status transitions ------------------------------------------------------

def test_status_transitions_and_unknown(tmp_path: Path):
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    assert update_artifact_status(tmp_path, record.artifact_id, "kept", now="t2").status == "kept"
    assert update_artifact_status(tmp_path, record.artifact_id, "rejected", now="t3").status == "rejected"
    with pytest.raises(ValueError):
        update_artifact_status(tmp_path, record.artifact_id, "archived", now="t4")


def test_list_tolerates_duplicate_and_corrupt(tmp_path: Path):
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)
    record = register_artifact(tmp_path, "20260702T180000-a", str(adapter), now="t1")
    # A stray duplicate + a corrupt file must not break the listing.
    directory = tmp_path / "model_artifacts"
    (directory / f"{record.artifact_id} - Copy.json").write_text(
        load_artifact_record(directory / f"{record.artifact_id}.json").model_dump_json(),
        encoding="utf-8",
    )
    (directory / "broken.json").write_text("{ not json", encoding="utf-8")
    assert len(list_artifacts(tmp_path)) == 1


# --- CLI ---------------------------------------------------------------------

def test_cli_register_list_update(tmp_path: Path):
    _make_run(tmp_path)
    adapter = _adapter_dir(tmp_path)

    reg = runner.invoke(app, ["artifact-register", str(tmp_path), "--run-id", "20260702T180000-a", "--path", str(adapter)])
    assert reg.exit_code == 0, reg.output
    artifact_id = json.loads(reg.output)["artifact_id"]

    listed = runner.invoke(app, ["artifact-list", str(tmp_path)])
    assert listed.exit_code == 0
    entry = json.loads(listed.output)["artifacts"][0]
    assert entry["integrity"] == "ok"
    assert entry["status"] == "candidate"

    upd = runner.invoke(app, ["artifact-update", str(tmp_path), "--artifact-id", artifact_id, "--status", "kept"])
    assert upd.exit_code == 0
    assert json.loads(upd.output)["status"] == "kept"


def test_cli_register_rejects_unknown_run(tmp_path: Path):
    result = runner.invoke(app, ["artifact-register", str(tmp_path), "--run-id", "ghost", "--path", str(tmp_path)])
    assert result.exit_code == 1
