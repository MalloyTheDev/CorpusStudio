"""The engine's sanctioned single-writer for examples.jsonl (#546)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.storage.examples_writer import (
    ExamplesLockedError,
    append_examples,
    examples_path,
    read_existing_lines,
    single_writer_lock,
    write_examples,
    write_examples_lines,
)

runner = CliRunner()

VALID = [{"instruction": "A", "output": "1"}, {"instruction": "B", "output": "2"}]


def _project(tmp_path: Path, schema_id: str = "instruction") -> Path:
    (tmp_path / "project.json").write_text(
        json.dumps({"id": "p", "name": "P", "schema_id": schema_id}), encoding="utf-8"
    )
    return tmp_path


def _read_rows(project: Path) -> list[dict]:
    return list(read_jsonl(examples_path(project)))


# --- module: writer primitives ------------------------------------------------

def test_append_creates_and_round_trips(tmp_path: Path):
    n = append_examples(tmp_path, VALID)
    assert n == 2
    assert _read_rows(tmp_path) == VALID


def test_append_preserves_existing_and_order(tmp_path: Path):
    append_examples(tmp_path, [VALID[0]])
    append_examples(tmp_path, [VALID[1]])
    assert _read_rows(tmp_path) == VALID


def test_append_preserves_field_order_and_unicode(tmp_path: Path):
    row = {"instruction": "z", "output": "éü", "tags": ["b", "a"]}
    append_examples(tmp_path, [row])
    # verbatim: field order and non-ASCII are preserved (not re-canonicalized)
    line = read_existing_lines(tmp_path)[0]
    assert line == json.dumps(row, ensure_ascii=False)
    assert line.index('"instruction"') < line.index('"output"') < line.index('"tags"')


def test_write_examples_replaces(tmp_path: Path):
    append_examples(tmp_path, VALID)
    n = write_examples(tmp_path, [{"instruction": "C", "output": "3"}])
    assert n == 1
    assert _read_rows(tmp_path) == [{"instruction": "C", "output": "3"}]


def test_write_examples_lines_verbatim(tmp_path: Path):
    lines = [json.dumps(r) for r in VALID]
    write_examples_lines(tmp_path, lines)
    assert [json.dumps(r) for r in _read_rows(tmp_path)] == lines


def test_read_existing_lines_missing_is_empty(tmp_path: Path):
    assert read_existing_lines(tmp_path) == []


def test_append_leaves_no_temp_files(tmp_path: Path):
    append_examples(tmp_path, VALID)
    assert not list(tmp_path.glob("examples.jsonl.*.tmp"))


def test_append_empty_is_noop(tmp_path: Path):
    append_examples(tmp_path, VALID)
    assert append_examples(tmp_path, []) == 0
    assert _read_rows(tmp_path) == VALID


def test_single_writer_lock_is_exclusive(tmp_path: Path):
    pytest.importorskip("fcntl")
    with single_writer_lock(tmp_path):
        with pytest.raises(ExamplesLockedError):
            with single_writer_lock(tmp_path):
                pass


# --- CLI: examples-append -----------------------------------------------------

def _from_file(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "incoming.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_cli_append_valid_uses_project_schema(tmp_path: Path):
    project = _project(tmp_path)
    src = _from_file(tmp_path, VALID)
    result = runner.invoke(app, ["examples-append", str(project), "--from", str(src), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["appended"] == 2 and payload["skipped_invalid"] == 0
    assert _read_rows(project) == VALID


def test_cli_append_schema_override(tmp_path: Path):
    # no project.json; --schema drives validation
    src = _from_file(tmp_path, VALID)
    result = runner.invoke(
        app, ["examples-append", str(tmp_path), "--from", str(src), "--schema", "instruction"]
    )
    assert result.exit_code == 0, result.output
    assert _read_rows(tmp_path) == VALID


def test_cli_append_refuses_whole_batch_on_invalid(tmp_path: Path):
    project = _project(tmp_path)
    src = _from_file(tmp_path, [VALID[0], {"instruction": "no output"}])
    result = runner.invoke(app, ["examples-append", str(project), "--from", str(src)])
    assert result.exit_code == 1
    assert "Refusing to append" in result.output
    # nothing written - the invariant: no silent partial append
    assert read_existing_lines(project) == []


def test_cli_append_skip_invalid_appends_valid_only(tmp_path: Path):
    project = _project(tmp_path)
    src = _from_file(tmp_path, [VALID[0], {"instruction": "no output"}, VALID[1]])
    result = runner.invoke(
        app, ["examples-append", str(project), "--from", str(src), "--skip-invalid", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["appended"] == 2 and payload["skipped_invalid"] == 1
    assert _read_rows(project) == VALID


def test_cli_append_requires_a_schema(tmp_path: Path):
    src = _from_file(tmp_path, VALID)
    result = runner.invoke(app, ["examples-append", str(tmp_path), "--from", str(src)])
    assert result.exit_code == 1
    assert "No schema" in result.output


def test_cli_append_missing_project_dir(tmp_path: Path):
    src = _from_file(tmp_path, VALID)
    result = runner.invoke(
        app, ["examples-append", str(tmp_path / "nope"), "--from", str(src), "--schema", "instruction"]
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output
