"""import-commit: staging rows -> examples.jsonl + auto version capture (#546 Slice B)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.storage.examples_writer import examples_path, read_existing_lines

runner = CliRunner()

VALID = [{"instruction": "A", "output": "1"}, {"instruction": "B", "output": "2"}]


def _project(tmp_path: Path, schema: str = "instruction") -> Path:
    (tmp_path / "project.json").write_text(
        json.dumps({"id": "p", "name": "P", "schema_id": schema}), encoding="utf-8"
    )
    return tmp_path


def _staging(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "staging.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _rows(project: Path) -> list[dict]:
    return list(read_jsonl(examples_path(project)))


def test_commit_appends_and_captures_version(tmp_path: Path):
    project = _project(tmp_path)
    src = _staging(tmp_path, VALID)
    result = runner.invoke(app, ["import-commit", str(project), "--from", str(src), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["committed"] == 2 and payload["rejected"] == 0 and payload["version_id"]
    assert _rows(project) == VALID
    # the auto-captured version carries the import_commit trigger
    from corpus_studio.versions.version_registry import load_version_record, record_path

    record = load_version_record(record_path(project, payload["version_id"]))
    assert record.trigger == "import_commit" and record.row_count == 2


def test_commit_quarantines_invalid_rows(tmp_path: Path):
    project = _project(tmp_path)
    src = _staging(tmp_path, [VALID[0], {"instruction": "no output"}, VALID[1]])
    result = runner.invoke(app, ["import-commit", str(project), "--from", str(src), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["committed"] == 2 and payload["rejected"] == 1
    assert _rows(project) == VALID  # only the valid rows committed, invalid left out


def test_commit_no_version_flag(tmp_path: Path):
    project = _project(tmp_path)
    src = _staging(tmp_path, VALID)
    result = runner.invoke(
        app, ["import-commit", str(project), "--from", str(src), "--no-version", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["version_id"] is None
    assert _rows(project) == VALID


def test_commit_nothing_valid_commits_nothing(tmp_path: Path):
    project = _project(tmp_path)
    src = _staging(tmp_path, [{"instruction": "no output"}])
    result = runner.invoke(app, ["import-commit", str(project), "--from", str(src), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["committed"] == 0 and payload["rejected"] == 1 and payload["version_id"] is None
    assert read_existing_lines(project) == []  # nothing was appended


def test_commit_missing_project_dir(tmp_path: Path):
    src = _staging(tmp_path, VALID)
    result = runner.invoke(
        app, ["import-commit", str(tmp_path / "nope"), "--from", str(src), "--schema", "instruction"]
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_import_commit_holds_lock_across_version_capture(tmp_path, monkeypatch):
    # #566: append + version capture run under ONE held lock. A create_dataset_version that
    # tries to re-acquire the single-writer lock must fail (the lock is already held),
    # proving there is no unlock window between the append and the capture.
    import pytest

    pytest.importorskip("fcntl")
    from corpus_studio.storage.examples_writer import single_writer_lock

    project = _project(tmp_path)
    src = _staging(tmp_path, VALID)

    def _fake_capture(project_dir, **_kw):
        with single_writer_lock(project_dir):  # would succeed only if the lock were released
            pass
        raise AssertionError("the single-writer lock was NOT held during version capture")

    monkeypatch.setattr(
        "corpus_studio.versions.version_registry.create_dataset_version", _fake_capture
    )
    result = runner.invoke(app, ["import-commit", str(project), "--from", str(src), "--json"])
    assert result.exit_code == 1  # the inner re-acquire raises ExamplesLockedError -> exit 1
    assert _rows(project) == VALID  # the append still happened, before the capture attempt
