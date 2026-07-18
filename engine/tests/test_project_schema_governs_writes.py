"""The sanctioned single writer validates against a project's OWN (derived) schema (#581 follow-up).

examples-append / import-commit / examples-edit resolve the schema project-local-first (via
resolve_schema), so a derived schema - e.g. one that pins a label enum a builtin does not have -
actually governs commits, not only the schema-* commands. Resolution never got laxer: an id that is
neither project-local nor builtin is still refused.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.schemas.project_schemas import save_project_schema
from corpus_studio.schemas.registry import load_builtin_schema
from corpus_studio.storage.examples_writer import examples_path, read_existing_lines

runner = CliRunner()

GOOD = {"instruction": "q", "output": "yes"}
BAD = {"instruction": "q", "output": "maybe"}  # violates the derived enum ["yes", "no"]


def _project_with_derived_schema(tmp_path: Path) -> Path:
    # project.json points at a project-local schema id 'graded' that is NOT a builtin.
    (tmp_path / "project.json").write_text(
        json.dumps({"id": "p", "name": "P", "schema_id": "graded"}), encoding="utf-8"
    )
    (tmp_path / "examples.jsonl").touch()
    graded = load_builtin_schema("instruction").model_copy(update={"id": "graded", "name": "Graded"})
    fields = [
        field.model_copy(update={"enum": ["yes", "no"]}) if field.name == "output" else field
        for field in graded.fields
    ]
    save_project_schema(tmp_path, graded.model_copy(update={"fields": fields}))
    return tmp_path


def _staging(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "staging.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _rows(project: Path) -> list[dict]:
    return [json.loads(line) for line in read_existing_lines(project)]


def test_append_honors_project_local_schema(tmp_path: Path):
    project = _project_with_derived_schema(tmp_path)
    src = _staging(tmp_path, [GOOD, BAD])
    result = runner.invoke(
        app, ["examples-append", str(project), "--from", str(src), "--skip-invalid", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_source"] == "project"  # the derived schema, not the builtin
    assert payload["appended"] == 1 and payload["skipped_invalid"] == 1  # BAD failed the pinned enum
    assert _rows(project) == [GOOD]


def test_append_refuses_batch_on_project_schema_violation(tmp_path: Path):
    project = _project_with_derived_schema(tmp_path)
    src = _staging(tmp_path, [GOOD, BAD])
    result = runner.invoke(app, ["examples-append", str(project), "--from", str(src)])
    assert result.exit_code == 1 and "fail the 'graded' schema" in result.output
    assert read_existing_lines(project) == []  # nothing written


def test_import_commit_honors_project_local_schema(tmp_path: Path):
    project = _project_with_derived_schema(tmp_path)
    src = _staging(tmp_path, [GOOD, BAD])
    result = runner.invoke(app, ["import-commit", str(project), "--from", str(src), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_source"] == "project"
    assert payload["committed"] == 1 and payload["rejected"] == 1
    assert _rows(project) == [GOOD]


def test_examples_edit_honors_project_local_schema(tmp_path: Path):
    project = _project_with_derived_schema(tmp_path)
    examples_path(project).write_text(json.dumps(GOOD) + "\n", encoding="utf-8")
    bad = runner.invoke(app, ["examples-edit", str(project), "--row", "1", "--to", json.dumps(BAD)])
    assert bad.exit_code == 1 and "fails the 'graded' schema" in bad.output
    assert _rows(project) == [GOOD]  # unchanged
    ok = runner.invoke(
        app,
        ["examples-edit", str(project), "--row", "1", "--to", json.dumps({"instruction": "q2", "output": "no"})],
    )
    assert ok.exit_code == 0, ok.output


def test_unknown_schema_is_still_refused(tmp_path: Path):
    # Resolution did not get laxer: an id neither project-local nor builtin still fails.
    (tmp_path / "project.json").write_text(
        json.dumps({"schema_id": "nonexistent"}), encoding="utf-8"
    )
    (tmp_path / "examples.jsonl").touch()
    src = _staging(tmp_path, [GOOD])
    result = runner.invoke(app, ["examples-append", str(tmp_path), "--from", str(src)])
    assert result.exit_code == 1 and "Unknown schema" in result.output


def test_human_output_discloses_the_project_schema(tmp_path: Path):
    # a derived schema governing a commit is disclosed even without --json (never silent)
    project = _project_with_derived_schema(tmp_path)
    appended = runner.invoke(
        app, ["examples-append", str(project), "--from", str(_staging(tmp_path, [GOOD]))]
    )
    assert appended.exit_code == 0 and "project-local schema 'graded'" in appended.output
    committed = runner.invoke(
        app, ["import-commit", str(project), "--from", str(_staging(tmp_path, [GOOD])), "--no-version"]
    )
    assert committed.exit_code == 0 and "project schema 'graded'" in committed.output


def test_builtin_schema_still_used_when_no_project_local_copy(tmp_path: Path):
    # A plain builtin project (no schemas/ dir) keeps validating against the builtin - source 'builtin'.
    (tmp_path / "project.json").write_text(
        json.dumps({"schema_id": "instruction"}), encoding="utf-8"
    )
    (tmp_path / "examples.jsonl").touch()
    src = _staging(tmp_path, [GOOD])
    result = runner.invoke(app, ["examples-append", str(tmp_path), "--from", str(src), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["schema_source"] == "builtin"
