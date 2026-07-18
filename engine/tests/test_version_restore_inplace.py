"""In-place dataset-version restore + the create_dataset_version helper (#546 Slice C)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.versions.version_registry import create_dataset_version
from corpus_studio.versions.version_restore import reconstruct_version_lines

runner = CliRunner()

V1 = [{"instruction": "A", "output": "1"}, {"instruction": "B", "output": "2"}]
EXTRA = [{"instruction": "C", "output": "3"}]


def _write(project: Path, rows: list[dict]) -> None:
    (project / "examples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _rows(project: Path) -> list[dict]:
    return list(read_jsonl(project / "examples.jsonl"))


def _create_version(project: Path) -> str:
    result = runner.invoke(app, ["dataset-version-create", str(project), "--store-rows"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["version_id"]


# --- create_dataset_version helper -------------------------------------------

def test_create_dataset_version_stores_and_restores(tmp_path: Path):
    _write(tmp_path, V1)
    rec = create_dataset_version(tmp_path, label="snap", trigger="manual")
    assert rec.rows_stored and rec.row_count == 2
    lines = reconstruct_version_lines(tmp_path, rec.version_id)
    assert [json.loads(line) for line in lines] == V1


# --- in-place restore ---------------------------------------------------------

def test_inplace_restore_swaps_and_undo_round_trips(tmp_path: Path):
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    _write(tmp_path, V1 + EXTRA)  # mutate the live dataset

    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", v1, "--in-place", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["in_place"] is True and payload["undo_version_id"]
    assert payload["verified"] is True
    assert _rows(tmp_path) == V1  # dataset swapped to v1

    # the undo version faithfully restores the pre-restore (mutated) dataset
    undo_out = tmp_path / "undo.jsonl"
    undo = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", payload["undo_version_id"],
         "--output", str(undo_out)],
    )
    assert undo.exit_code == 0, undo.output
    assert list(read_jsonl(undo_out)) == V1 + EXTRA


def test_inplace_and_output_are_mutually_exclusive(tmp_path: Path):
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    result = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", v1,
         "--in-place", "--output", str(tmp_path / "o.jsonl")],
    )
    assert result.exit_code == 1
    assert "not both" in result.output


def test_restore_requires_output_or_inplace(tmp_path: Path):
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    result = runner.invoke(app, ["dataset-version-restore", str(tmp_path), "--version-id", v1])
    assert result.exit_code == 1
    assert "--in-place" in result.output


def test_output_targeting_examples_points_at_inplace(tmp_path: Path):
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    result = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", v1,
         "--output", str(tmp_path / "examples.jsonl")],
    )
    assert result.exit_code == 1
    assert "--in-place" in result.output


def test_inplace_refuses_when_undo_not_restorable(tmp_path, monkeypatch):
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    _write(tmp_path, V1 + EXTRA)

    from corpus_studio.versions.version_registry import DatasetVersionRecord

    def _fake(project_dir, **_kw):  # readable (has a fingerprint) but the row store failed
        return DatasetVersionRecord(
            version_id="fake", created_at="x", updated_at="x", row_count=3,
            content_fingerprint="deadbeef", rows_stored=False,
        )

    monkeypatch.setattr(
        "corpus_studio.versions.version_registry.create_dataset_version", _fake
    )
    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", v1, "--in-place"]
    )
    assert result.exit_code == 1
    assert "restorable undo" in result.output
    assert _rows(tmp_path) == V1 + EXTRA  # dataset left untouched


def test_inplace_refuses_unreadable_current_dataset(tmp_path: Path):
    # (Fable 5 BLOCKER) a torn line must NOT be treated as an empty dataset.
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)  # captured while clean
    (tmp_path / "examples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in V1) + "{ not json\n", encoding="utf-8"
    )
    before = (tmp_path / "examples.jsonl").read_text(encoding="utf-8")
    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", v1, "--in-place"]
    )
    assert result.exit_code == 1
    assert "unreadable" in result.output
    assert (tmp_path / "examples.jsonl").read_text(encoding="utf-8") == before  # NOT destroyed


def test_inplace_refuses_when_undo_not_reconstructable(tmp_path, monkeypatch):
    # (Fable 5 HIGH) rows_stored=True is not proof: reject if the undo can't reconstruct.
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    _write(tmp_path, V1 + EXTRA)
    before = _rows(tmp_path)

    from corpus_studio.versions.version_registry import DatasetVersionRecord

    def _fake(project_dir, **_kw):  # claims stored rows but no manifest exists -> unreconstructable
        return DatasetVersionRecord(
            version_id="ghost", created_at="x", updated_at="x", row_count=3,
            content_fingerprint="deadbeef", rows_stored=True,
        )

    monkeypatch.setattr(
        "corpus_studio.versions.version_registry.create_dataset_version", _fake
    )
    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", v1, "--in-place"]
    )
    assert result.exit_code == 1
    assert "did not verify" in result.output
    assert _rows(tmp_path) == before  # untouched


def test_inplace_refuses_without_verification(tmp_path: Path):
    # (Fable 5 MED) a destructive path must fail closed - no unverified in-place write.
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    _write(tmp_path, V1 + EXTRA)
    before = _rows(tmp_path)
    result = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", v1, "--in-place", "--no-verify"],
    )
    assert result.exit_code == 1
    assert "requires a verified" in result.output
    assert _rows(tmp_path) == before  # untouched


def test_inplace_on_missing_dataset_reports_no_undo(tmp_path: Path):
    _write(tmp_path, V1)
    v1 = _create_version(tmp_path)
    (tmp_path / "examples.jsonl").unlink()  # no live dataset -> nothing to undo
    result = runner.invoke(
        app, ["dataset-version-restore", str(tmp_path), "--version-id", v1, "--in-place", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["undo_version_id"] is None
    assert _rows(tmp_path) == V1  # recreated from v1
