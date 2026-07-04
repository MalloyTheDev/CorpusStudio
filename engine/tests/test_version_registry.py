import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.exporters.cleaning import exact_row_signature
from corpus_studio.training.run_registry import (
    TrainingRunRecord,
    load_run_record,
    record_path as run_record_path,
    save_run_record,
)
from corpus_studio.versions.version_registry import (
    DatasetVersionRecord,
    capture_dataset,
    compute_content_fingerprint,
    current_integrity,
    fingerprint_dataset,
    integrity_from_fingerprints,
    list_version_records,
    load_row_manifest,
    load_version_record,
    record_path,
    registry_dir,
    save_row_manifest,
    save_version_record,
)

runner = CliRunner()

ROWS = [
    {"instruction": "Explain recursion clearly.", "output": "A function calls itself on subproblems."},
    {"instruction": "Explain binary search.", "output": "It halves a sorted range each step."},
]


def _write_examples(project: Path, rows: list[dict]) -> Path:
    path = project / "examples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _record(version_id: str, fingerprint: str | None = None, **kw) -> DatasetVersionRecord:
    return DatasetVersionRecord(
        version_id=version_id, created_at="t", updated_at="t", content_fingerprint=fingerprint, **kw
    )


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_reuses_exact_signature_and_is_deterministic(tmp_path: Path):
    path = _write_examples(tmp_path, ROWS)
    fingerprint, count = fingerprint_dataset(path)
    expected = hashlib.sha256(
        "\n".join(exact_row_signature(row) for row in ROWS).encode("utf-8")
    ).hexdigest()
    assert fingerprint == expected
    assert count == len(ROWS)
    assert fingerprint == compute_content_fingerprint(path)  # stable across calls


def test_fingerprint_none_on_missing_file(tmp_path: Path):
    assert fingerprint_dataset(tmp_path / "examples.jsonl") == (None, 0)
    assert compute_content_fingerprint(tmp_path / "examples.jsonl") is None


def test_fingerprint_is_order_sensitive(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fp_forward = compute_content_fingerprint(_write_examples(a, ROWS))
    fp_reversed = compute_content_fingerprint(_write_examples(b, list(reversed(ROWS))))
    assert fp_forward is not None and fp_reversed is not None
    assert fp_forward != fp_reversed  # same rows, different order -> different identity


def test_fingerprint_none_on_malformed_line(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    path.write_text('{"ok": 1}\n{ this is not json\n', encoding="utf-8")
    assert compute_content_fingerprint(path) is None  # unreadable, not a wrong hash


def test_fingerprint_none_on_deeply_nested_line(tmp_path: Path):
    # Pathologically nested JSON makes json.loads raise RecursionError (not a
    # ValueError); the loader must still degrade to (None, 0), never crash.
    path = tmp_path / "examples.jsonl"
    path.write_text('{"ok": 1}\n' + "[" * 6000 + "]" * 6000 + "\n", encoding="utf-8")
    assert fingerprint_dataset(path) == (None, 0)


def test_fingerprint_none_on_invalid_utf8(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    path.write_bytes(b'{"ok": 1}\n\xff\xfe not utf-8\n')
    assert compute_content_fingerprint(path) is None


def test_empty_file_has_stable_fingerprint(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    path.touch()
    fingerprint, count = fingerprint_dataset(path)
    assert count == 0
    assert fingerprint == hashlib.sha256(b"").hexdigest()


# --- v1.0.2 capture (single pass) --------------------------------------------

def test_capture_fingerprint_matches_fingerprint_dataset(tmp_path: Path):
    path = _write_examples(tmp_path, ROWS + [ROWS[0]])  # a duplicate row
    capture = capture_dataset(path, tmp_path, store_rows=True)
    # Single-pass capture must produce the exact same fingerprint as the streaming
    # fingerprint_dataset (they share the identity primitive and ordering).
    assert capture.content_fingerprint == fingerprint_dataset(path)[0]
    assert capture.row_count == 3
    assert len(capture.row_ids) == 3  # manifest is ordered, one id per row
    assert capture.new_rows_stored == 2  # the duplicate is stored once


def test_capture_missing_file_stores_nothing(tmp_path: Path):
    capture = capture_dataset(tmp_path / "examples.jsonl", tmp_path, store_rows=True)
    assert capture.content_fingerprint is None
    assert capture.row_count == 0 and capture.row_ids == [] and capture.new_rows_stored == 0
    assert not (registry_dir(tmp_path) / "row_store.jsonl").exists()


def test_capture_no_store_rows_skips_store(tmp_path: Path):
    path = _write_examples(tmp_path, ROWS)
    capture = capture_dataset(path, tmp_path, store_rows=False)
    assert capture.content_fingerprint is not None
    assert capture.rows_stored is False
    assert capture.new_rows_stored == 0
    assert not (registry_dir(tmp_path) / "row_store.jsonl").exists()


def _store_text(project: Path) -> str:
    store = registry_dir(project) / "row_store.jsonl"
    return store.read_text(encoding="utf-8") if store.exists() else ""


def test_capture_rolls_back_store_on_malformed_line(tmp_path: Path):
    # A valid row then a torn line: the dataset is unreadable, and the row already
    # read must NOT be left orphaned in the store ("failed capture stores nothing").
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\n" + "{ torn line\n", encoding="utf-8")
    capture = capture_dataset(path, tmp_path, store_rows=True)
    assert capture.content_fingerprint is None
    assert capture.rows_stored is False
    assert _store_text(tmp_path) == ""  # rolled back


def test_capture_store_failure_keeps_fingerprint(tmp_path: Path, monkeypatch):
    # A store I/O failure on a READABLE dataset must keep the real fingerprint
    # (rows_stored=False), not masquerade as an unreadable dataset.
    import corpus_studio.versions.row_store as rs

    path = _write_examples(tmp_path, ROWS)

    def boom(*args, **kwargs):
        raise OSError("read-only store")

    monkeypatch.setattr(rs, "store_line", boom)
    capture = capture_dataset(path, tmp_path, store_rows=True)
    assert capture.content_fingerprint == fingerprint_dataset(path)[0]  # parity preserved
    assert capture.rows_stored is False
    assert capture.new_rows_stored == 0
    assert _store_text(tmp_path) == ""  # partial store rolled back


def test_row_manifest_round_trip_and_absent(tmp_path: Path):
    save_row_manifest(tmp_path, "20260101T000000-1", ["aa", "bb", "aa"])
    assert load_row_manifest(tmp_path, "20260101T000000-1") == ["aa", "bb", "aa"]
    assert load_row_manifest(tmp_path, "no-such-version") is None  # absent -> None


def test_old_record_without_row_store_fields_loads(tmp_path: Path):
    # A pre-v1.0.2 record JSON must still load, with row-store fields defaulted.
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "version_id": "20260101T000000-1",
                "created_at": "t",
                "updated_at": "t",
                "content_fingerprint": "abc",
            }
        ),
        encoding="utf-8",
    )
    record = load_version_record(path)
    assert record.rows_stored is False
    assert record.stored_row_count == 0
    assert record.row_manifest_algo is None


# --- record storage ----------------------------------------------------------

def test_save_load_round_trip(tmp_path: Path):
    record = _record("20260101T000000-1", fingerprint="abc", label="v1", source_run_ids=["r1"])
    saved_path = save_version_record(tmp_path, record)
    assert load_version_record(saved_path) == record


def test_save_rejects_invalid_version_id(tmp_path: Path):
    bad = _record("bad id with spaces")
    try:
        save_version_record(tmp_path, bad)
        assert False, "expected ValueError for invalid version_id"
    except ValueError:
        pass


def test_list_newest_first_and_corrupt_tolerant(tmp_path: Path):
    save_version_record(tmp_path, _record("20260101T000000-a"))
    save_version_record(tmp_path, _record("20260301T000000-b"))
    (registry_dir(tmp_path) / "corrupt.json").write_text("{ not json", encoding="utf-8")
    records = list_version_records(tmp_path)
    assert [r.version_id for r in records] == ["20260301T000000-b", "20260101T000000-a"]


def test_slug_is_injective_for_underscore_ids(tmp_path: Path):
    # '_abc', 'abc', 'abc_' are all valid ids and must map to distinct files
    # (the old strip('_') slug collapsed them and silently overwrote).
    save_version_record(tmp_path, _record("abc", label="FIRST"))
    save_version_record(tmp_path, _record("_abc", label="SECOND"))
    save_version_record(tmp_path, _record("abc_", label="THIRD"))
    records = list_version_records(tmp_path)
    assert sorted(r.version_id for r in records) == ["_abc", "abc", "abc_"]
    assert load_version_record(record_path(tmp_path, "abc")).label == "FIRST"
    assert load_version_record(record_path(tmp_path, "_abc")).label == "SECOND"
    assert load_version_record(record_path(tmp_path, "abc_")).label == "THIRD"


def test_list_dedupes_same_version_id(tmp_path: Path):
    save_version_record(tmp_path, _record("20260101T000000-a", label="one"))
    # A second file that deserializes to the same version_id -> first wins, not double-counted.
    (registry_dir(tmp_path) / "dup.json").write_text(
        _record("20260101T000000-a", label="two").model_dump_json(), encoding="utf-8"
    )
    records = list_version_records(tmp_path)
    assert [r.version_id for r in records] == ["20260101T000000-a"]


# --- live integrity ----------------------------------------------------------

def test_current_integrity_matches_drifted_unreadable(tmp_path: Path):
    path = _write_examples(tmp_path, ROWS)
    fingerprint = compute_content_fingerprint(path)
    record = _record("20260101T000000-1", fingerprint=fingerprint)

    assert current_integrity(record, path) == "matches"

    _write_examples(tmp_path, ROWS + [{"instruction": "new", "output": "row"}])
    assert current_integrity(record, path) == "drifted"

    path.unlink()
    assert current_integrity(record, path) == "unreadable"


def test_integrity_from_fingerprints_unit():
    assert integrity_from_fingerprints("x", "x") == "matches"
    assert integrity_from_fingerprints("x", "y") == "drifted"
    assert integrity_from_fingerprints(None, "y") == "unreadable"
    assert integrity_from_fingerprints("x", None) == "unreadable"


# --- CLI end-to-end ----------------------------------------------------------

def test_cli_create_list_show_end_to_end(tmp_path: Path):
    _write_examples(tmp_path, ROWS)

    created = runner.invoke(app, ["dataset-version-create", str(tmp_path), "--label", "baseline"])
    assert created.exit_code == 0, created.output
    record = json.loads(created.stdout)
    version_id = record["version_id"]
    assert record["row_count"] == len(ROWS)
    assert record["content_fingerprint"] is not None
    assert record["fingerprint_algo"] == "sha256-ordered-exact-v1"

    listed = runner.invoke(app, ["dataset-version-list", str(tmp_path)])
    assert listed.exit_code == 0, listed.output
    versions = json.loads(listed.stdout)["versions"]
    assert versions[0]["version_id"] == version_id
    assert versions[0]["current_integrity"] == "matches"

    shown = runner.invoke(app, ["dataset-version-show", str(tmp_path), "--version-id", version_id])
    assert shown.exit_code == 0, shown.output
    assert "Dataset Version Card" in shown.stdout
    assert "matches" in shown.stdout

    shown_json = runner.invoke(
        app, ["dataset-version-show", str(tmp_path), "--version-id", version_id, "--json"]
    )
    card = json.loads(shown_json.stdout)
    assert card["current_integrity"] == "matches"


def test_cli_show_reflects_drift(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    created = runner.invoke(app, ["dataset-version-create", str(tmp_path)])
    assert json.loads(created.stdout)["version_id"]

    _write_examples(tmp_path, list(reversed(ROWS)))  # reorder -> drift
    listed = runner.invoke(app, ["dataset-version-list", str(tmp_path)])
    assert json.loads(listed.stdout)["versions"][0]["current_integrity"] == "drifted"


def test_cli_create_missing_examples_warns_and_records_no_fingerprint(tmp_path: Path):
    created = runner.invoke(app, ["dataset-version-create", str(tmp_path)])
    assert created.exit_code == 0, created.output
    record = json.loads(created.stdout)
    assert record["content_fingerprint"] is None
    assert record["row_count"] == 0
    assert "unreadable" in created.stderr.lower()


def test_cli_stamp_run_writes_back_link(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    save_run_record(
        tmp_path,
        TrainingRunRecord(run_id="20260101T000000-r", created_at="t", updated_at="t"),
    )
    created = runner.invoke(
        app, ["dataset-version-create", str(tmp_path), "--stamp-run", "20260101T000000-r"]
    )
    assert created.exit_code == 0, created.output
    record = json.loads(created.stdout)
    assert "20260101T000000-r" in record["source_run_ids"]

    run = load_run_record(run_record_path(tmp_path, "20260101T000000-r"))
    assert run.source_snapshot_id == record["version_id"]


def test_cli_stamp_missing_run_errors(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    created = runner.invoke(app, ["dataset-version-create", str(tmp_path), "--stamp-run", "nope"])
    assert created.exit_code == 1


def test_cli_stamp_not_written_when_version_save_fails(tmp_path: Path, monkeypatch):
    # The version must be committed before the run's back-link; if the version
    # save fails, the run must NOT be left stamped with a phantom snapshot id.
    import corpus_studio.versions.version_registry as vr

    _write_examples(tmp_path, ROWS)
    save_run_record(
        tmp_path, TrainingRunRecord(run_id="20260101T000000-r", created_at="t", updated_at="t")
    )

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(vr, "save_version_record", boom)
    result = runner.invoke(
        app, ["dataset-version-create", str(tmp_path), "--stamp-run", "20260101T000000-r"]
    )
    assert result.exit_code != 0
    run = load_run_record(run_record_path(tmp_path, "20260101T000000-r"))
    assert run.source_snapshot_id is None


def test_cli_two_creates_do_not_collide(tmp_path: Path):
    # A random token in the id prevents two same-tick in-process creates from
    # minting the same id and silently overwriting the first version's file.
    _write_examples(tmp_path, ROWS)
    first = runner.invoke(app, ["dataset-version-create", str(tmp_path), "--label", "first"])
    second = runner.invoke(app, ["dataset-version-create", str(tmp_path), "--label", "second"])
    assert json.loads(first.stdout)["version_id"] != json.loads(second.stdout)["version_id"]
    records = list_version_records(tmp_path)
    assert len(records) == 2
    assert {r.label for r in records} == {"first", "second"}


def test_cli_show_degrades_on_invalid_utf8_eval_report(tmp_path: Path):
    _write_examples(tmp_path, ROWS)
    bad = tmp_path / "eval.json"
    bad.write_bytes(b"\xff\xfe\x00 not valid utf-8")
    created = runner.invoke(
        app, ["dataset-version-create", str(tmp_path), "--eval-report-path", str(bad)]
    )
    version_id = json.loads(created.stdout)["version_id"]
    shown = runner.invoke(app, ["dataset-version-show", str(tmp_path), "--version-id", version_id])
    assert shown.exit_code == 0, shown.output
    assert "missing" in shown.stdout.lower()  # degraded to a flag, not a crash


def test_old_run_record_without_source_snapshot_id_loads_none(tmp_path: Path):
    # A pre-v1.0 record JSON with no source_snapshot_id must still load (field = None).
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"run_id": "20260101T000000-r", "created_at": "t", "updated_at": "t"}),
        encoding="utf-8",
    )
    record = load_run_record(path)
    assert record.source_snapshot_id is None


def test_cli_auto_links_newest_dataset_gate_report(tmp_path: Path):
    from corpus_studio.gates.models import GateReport, GateResult, GateScope, GateStatus
    from corpus_studio.gates.runner import save_gate_report

    _write_examples(tmp_path, ROWS)
    report = GateReport.build(
        GateScope.DATASET,
        "examples.jsonl",
        [GateResult(gate_id="schema", name="Schema", scope=GateScope.DATASET, status=GateStatus.PASS)],
        generated_at="2026-07-02T00:00:00+00:00",
    )
    gate_path = save_gate_report(tmp_path, report)

    created = runner.invoke(app, ["dataset-version-create", str(tmp_path)])
    record = json.loads(created.stdout)
    assert record["gate_report_path"] is not None
    assert Path(record["gate_report_path"]).name == gate_path.name
