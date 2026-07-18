"""examples-edit / examples-delete: row-level mutation via the single writer, with undo (#577, G2).

Both commands capture a verified undo version FIRST (mirroring dataset-version-restore --in-place),
then write atomically under the single-writer lock. These tests prove the mutation, the honest
refusals (bad address, invalid row, no dataset), and - the safety contract - that the captured undo
reconstructs the exact pre-mutation dataset.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.storage.examples_writer import examples_path, read_existing_lines
from corpus_studio.versions.version_restore import reconstruct_version_lines

runner = CliRunner()

ROWS = [{"instruction": f"i{n}", "output": f"o{n}"} for n in range(4)]


def _project(tmp_path: Path) -> Path:
    (tmp_path / "project.json").write_text(
        json.dumps({"id": "p", "name": "P", "schema_id": "instruction"}), encoding="utf-8"
    )
    examples_path(tmp_path).write_text(
        "".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8"
    )
    return tmp_path


def _rows(project: Path) -> list[dict]:
    return [json.loads(line) for line in read_existing_lines(project)]


# --- examples-delete --------------------------------------------------------------------------


def test_delete_removes_rows_and_captures_restorable_undo(tmp_path: Path):
    project = _project(tmp_path)
    result = runner.invoke(
        app, ["examples-delete", str(project), "--row", "2", "--row", "4", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["deleted"] == 2 and payload["remaining"] == 2
    assert _rows(project) == [ROWS[0], ROWS[2]]  # rows 1 and 3 (1-based) survive
    # the safety contract: the undo version reconstructs the ORIGINAL dataset exactly
    undo_lines = reconstruct_version_lines(project, payload["undo_version_id"])
    assert [json.loads(line) for line in undo_lines] == ROWS


def test_delete_out_of_range_refuses_and_changes_nothing(tmp_path: Path):
    project = _project(tmp_path)
    before = _rows(project)
    result = runner.invoke(app, ["examples-delete", str(project), "--row", "99"])
    assert result.exit_code == 1
    assert "out of range" in result.output
    assert _rows(project) == before  # no silent partial delete


def test_delete_missing_examples_refuses(tmp_path: Path):
    (tmp_path / "project.json").write_text(
        json.dumps({"schema_id": "instruction"}), encoding="utf-8"
    )
    result = runner.invoke(app, ["examples-delete", str(tmp_path), "--row", "1"])
    assert result.exit_code == 1
    assert "no examples.jsonl" in result.output


def test_delete_missing_project_dir(tmp_path: Path):
    result = runner.invoke(app, ["examples-delete", str(tmp_path / "nope"), "--row", "1"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


# --- examples-edit ----------------------------------------------------------------------------


def test_edit_replaces_row_and_captures_restorable_undo(tmp_path: Path):
    project = _project(tmp_path)
    new = {"instruction": "edited", "output": "changed"}
    result = runner.invoke(
        app, ["examples-edit", str(project), "--row", "2", "--to", json.dumps(new), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["edited_row"] == 2 and payload["total"] == 4
    rows = _rows(project)
    assert rows[1] == new and rows[0] == ROWS[0] and rows[2] == ROWS[2]
    undo_lines = reconstruct_version_lines(project, payload["undo_version_id"])
    assert [json.loads(line) for line in undo_lines] == ROWS


def test_edit_from_file(tmp_path: Path):
    project = _project(tmp_path)
    new = {"instruction": "fromfile", "output": "ok"}
    src = tmp_path / "row.json"
    src.write_text(json.dumps(new), encoding="utf-8")
    result = runner.invoke(app, ["examples-edit", str(project), "--row", "1", "--from", str(src)])
    assert result.exit_code == 0, result.output
    assert _rows(project)[0] == new


def test_edit_invalid_row_refused(tmp_path: Path):
    project = _project(tmp_path)
    before = _rows(project)
    # the 'instruction' schema requires instruction + output; this replacement omits output
    result = runner.invoke(
        app, ["examples-edit", str(project), "--row", "1", "--to", json.dumps({"instruction": "x"})]
    )
    assert result.exit_code == 1
    assert "fails the 'instruction' schema" in result.output
    assert _rows(project) == before


def test_edit_out_of_range_refused(tmp_path: Path):
    project = _project(tmp_path)
    before = _rows(project)
    result = runner.invoke(
        app,
        ["examples-edit", str(project), "--row", "99", "--to", json.dumps({"instruction": "a", "output": "b"})],
    )
    assert result.exit_code == 1
    assert "out of range" in result.output
    assert _rows(project) == before


def test_edit_requires_exactly_one_source(tmp_path: Path):
    project = _project(tmp_path)
    neither = runner.invoke(app, ["examples-edit", str(project), "--row", "1"])
    assert neither.exit_code == 1 and "exactly one" in neither.output
    src = tmp_path / "row.json"
    src.write_text("{}", encoding="utf-8")
    both = runner.invoke(
        app, ["examples-edit", str(project), "--row", "1", "--to", "{}", "--from", str(src)]
    )
    assert both.exit_code == 1 and "exactly one" in both.output


def test_edit_invalid_json_refused(tmp_path: Path):
    project = _project(tmp_path)
    before = _rows(project)
    result = runner.invoke(app, ["examples-edit", str(project), "--row", "1", "--to", "{not json"])
    assert result.exit_code == 1
    assert "not valid JSON" in result.output
    assert _rows(project) == before
