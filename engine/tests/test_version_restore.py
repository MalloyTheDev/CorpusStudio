import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.versions.row_store import load_rows_by_id, row_id, row_store_path
from corpus_studio.versions.version_registry import (
    capture_dataset,
    fingerprint_dataset,
    record_path,
)
from corpus_studio.versions.version_restore import reconstruct_and_verify

runner = CliRunner()

ROWS = [{"instruction": "A", "output": "1"}, {"instruction": "B", "output": "2"}]


def _write_examples(project: Path, rows: list[dict]) -> None:
    (project / "examples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _create(project: Path, *extra: str) -> dict:
    result = runner.invoke(app, ["dataset-version-create", str(project), *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


# --- pure reconstruct_and_verify ---------------------------------------------

def test_reconstruct_reproduces_fingerprint(tmp_path: Path):
    _write_examples(tmp_path, ROWS + [ROWS[0]])  # a duplicate row
    capture = capture_dataset(tmp_path / "examples.jsonl", tmp_path, store_rows=True)
    rows_by_id = load_rows_by_id(tmp_path, set(capture.row_ids))

    lines, computed, matches, missing = reconstruct_and_verify(
        capture.row_ids, rows_by_id, capture.content_fingerprint
    )
    assert missing == []
    assert computed == capture.content_fingerprint  # faithful restore reproduces the fingerprint
    assert matches is True
    assert len(lines) == 3  # duplicate emitted per occurrence, in order


def test_reconstruct_missing_id_emits_nothing():
    lines, _computed, matches, missing = reconstruct_and_verify(
        ["a", "b"], {"a": {"x": 1}}, "whatever"
    )
    assert missing == ["b"]
    assert lines == []  # never a partial reconstruction
    assert matches is False


def test_reconstruct_empty_manifest():
    lines, computed, matches, _missing = reconstruct_and_verify(
        [], {}, hashlib.sha256(b"").hexdigest()
    )
    assert lines == []
    assert computed == hashlib.sha256(b"").hexdigest()
    assert matches is True


def test_reconstruct_duplicates_in_order():
    r1, r2 = {"a": 1}, {"b": 2}
    rows = {row_id(r1): r1, row_id(r2): r2}
    manifest = [row_id(r1), row_id(r2), row_id(r1)]
    lines, _c, _m, missing = reconstruct_and_verify(manifest, rows, None)
    assert missing == []
    assert lines == [
        json.dumps(r1, ensure_ascii=False, sort_keys=True),
        json.dumps(r2, ensure_ascii=False, sort_keys=True),
        json.dumps(r1, ensure_ascii=False, sort_keys=True),
    ]


# --- CLI round-trip + refusals -----------------------------------------------

def test_cli_restore_round_trip(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    created = _create(tmp_path)
    out = tmp_path / "restored.jsonl"
    result = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", created["version_id"], "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    # ROUND-TRIP PROOF: the restored file re-fingerprints to the version's fingerprint.
    assert fingerprint_dataset(out)[0] == created["content_fingerprint"]
    assert "verified" in result.stdout


def test_cli_restore_refuses_examples_jsonl(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    vid = _create(tmp_path)["version_id"]
    result = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", vid, "--output", str(tmp_path / "examples.jsonl")],
    )
    assert result.exit_code == 1
    assert "never writes the dataset" in result.stderr


def test_cli_restore_refuses_existing_then_force_overwrites(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    created = _create(tmp_path)
    vid = created["version_id"]
    out = tmp_path / "out.jsonl"
    out.write_text("existing\n", encoding="utf-8")

    refused = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", vid, "--output", str(out)]
    )
    assert refused.exit_code == 1 and "exists" in refused.stderr

    forced = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", vid, "--output", str(out), "--force"]
    )
    assert forced.exit_code == 0
    assert fingerprint_dataset(out)[0] == created["content_fingerprint"]  # overwritten with the restore
    assert out.read_text(encoding="utf-8") != "existing\n"


def test_cli_restore_refuses_no_stored_rows(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    vid = _create(tmp_path, "--no-store-rows")["version_id"]
    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", vid, "--output", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code == 1 and "no stored rows" in result.stderr.lower()


def test_cli_restore_refuses_missing_rows_and_writes_nothing(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    vid = _create(tmp_path)["version_id"]
    row_store_path(tmp_path).write_text("", encoding="utf-8")  # wipe the store
    out = tmp_path / "o.jsonl"
    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", vid, "--output", str(out)]
    )
    assert result.exit_code == 1 and "missing from the store" in result.stderr.lower()
    assert not out.exists()  # all-or-nothing: wrote nothing


def test_cli_restore_refuses_corrupt_record(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    vid = _create(tmp_path)["version_id"]
    record_path(tmp_path, vid).write_text("not json {{{", encoding="utf-8")
    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", vid, "--output", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code == 1 and "corrupt record" in result.stderr.lower()


def test_cli_restore_no_verify_still_faithful(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    created = _create(tmp_path)
    out = tmp_path / "o.jsonl"
    result = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", created["version_id"],
         "--output", str(out), "--no-verify", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["verify_skipped"] is True and payload["verified"] is False
    assert fingerprint_dataset(out)[0] == created["content_fingerprint"]  # still a faithful restore
