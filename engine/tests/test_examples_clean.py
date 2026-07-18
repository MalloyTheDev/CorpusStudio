"""examples-clean: dedupe / drop-low-information through the single writer, with undo (#578, G3).

Reuses the export path's clean_rows (exact + NFKC-normalized duplicates, low-information rows), so an
in-place clean matches what `export --dedupe` would remove. Default is a read-only preview; --in-place
captures a verified undo version first. These tests prove the counts, the preview/apply split, the
honest refusals, and that the undo reconstructs the exact pre-clean dataset.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.storage.examples_writer import examples_path, read_existing_lines
from corpus_studio.versions.version_restore import reconstruct_version_lines

runner = CliRunner()

# Two "rich" rows (enough tokens to not be low-information), plus an exact duplicate of the first and
# a case/whitespace variant that is a NORMALIZED duplicate of the first.
R_A = {"instruction": "please summarize the report", "output": "the quarterly numbers improved"}
R_A_EXACT = {"instruction": "please summarize the report", "output": "the quarterly numbers improved"}
R_A_NORM = {"instruction": "Please  summarize the report", "output": "The quarterly numbers improved"}
R_B = {"instruction": "translate this sentence", "output": "here is the translation done"}
DUPES = [R_A, R_A_EXACT, R_A_NORM, R_B]


def _project(tmp_path: Path, rows: list[dict]) -> Path:
    (tmp_path / "project.json").write_text(
        json.dumps({"id": "p", "name": "P", "schema_id": "instruction"}), encoding="utf-8"
    )
    examples_path(tmp_path).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return tmp_path


def _rows(project: Path) -> list[dict]:
    return [json.loads(line) for line in read_existing_lines(project)]


def test_preview_default_reports_but_writes_nothing(tmp_path: Path):
    project = _project(tmp_path, DUPES)
    result = runner.invoke(app, ["examples-clean", str(project), "--dedupe", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "preview"
    assert payload["removed_rows"] == 2
    assert payload["removed_exact_duplicates"] == 1
    assert payload["removed_normalized_duplicates"] == 1
    assert payload["undo_version_id"] is None
    # preview never mutates the dataset
    assert _rows(project) == DUPES


def test_in_place_dedupe_removes_and_captures_restorable_undo(tmp_path: Path):
    project = _project(tmp_path, DUPES)
    result = runner.invoke(
        app, ["examples-clean", str(project), "--dedupe", "--in-place", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "in_place"
    assert payload["removed_rows"] == 2 and payload["kept_rows"] == 2
    assert _rows(project) == [R_A, R_B]  # first occurrence kept, dupes dropped
    # the safety contract: the undo reconstructs the ORIGINAL dataset exactly
    undo_lines = reconstruct_version_lines(project, payload["undo_version_id"])
    assert [json.loads(line) for line in undo_lines] == DUPES


def test_in_place_drop_low_information(tmp_path: Path):
    rows = [R_A, {"instruction": "a", "output": "b"}]  # second row is low-information (1 token)
    project = _project(tmp_path, rows)
    result = runner.invoke(
        app, ["examples-clean", str(project), "--drop-low-information", "--in-place", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["removed_low_information"] == 1 and payload["removed_rows"] == 1
    assert _rows(project) == [R_A]


def test_in_place_already_clean_changes_nothing_and_creates_no_undo(tmp_path: Path):
    project = _project(tmp_path, [R_A, R_B])
    result = runner.invoke(
        app, ["examples-clean", str(project), "--dedupe", "--in-place", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["removed_rows"] == 0 and payload["undo_version_id"] is None
    assert _rows(project) == [R_A, R_B]


def test_requires_a_clean_flag(tmp_path: Path):
    project = _project(tmp_path, DUPES)
    result = runner.invoke(app, ["examples-clean", str(project)])
    assert result.exit_code == 1
    assert "at least one of --dedupe or --drop-low-information" in result.output


def test_missing_examples_refuses(tmp_path: Path):
    (tmp_path / "project.json").write_text(
        json.dumps({"schema_id": "instruction"}), encoding="utf-8"
    )
    result = runner.invoke(app, ["examples-clean", str(tmp_path), "--dedupe"])
    assert result.exit_code == 1
    assert "no examples.jsonl" in result.output


def test_malformed_row_refuses_and_changes_nothing(tmp_path: Path):
    project = _project(tmp_path, [R_A, R_B])
    # corrupt the file with a torn line
    examples_path(project).write_text(
        json.dumps(R_A) + "\n" + "{not json\n" + json.dumps(R_B) + "\n", encoding="utf-8"
    )
    before = examples_path(project).read_text(encoding="utf-8")
    result = runner.invoke(app, ["examples-clean", str(project), "--dedupe", "--in-place"])
    assert result.exit_code == 1
    assert "not valid JSON" in result.output
    assert examples_path(project).read_text(encoding="utf-8") == before  # untouched


def test_missing_project_dir(tmp_path: Path):
    result = runner.invoke(app, ["examples-clean", str(tmp_path / "nope"), "--dedupe"])
    assert result.exit_code == 1
    assert "does not exist" in result.output
