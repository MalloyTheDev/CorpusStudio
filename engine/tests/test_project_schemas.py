"""Project-local schema design: derive/edit/validate (#581, G6).

A project can copy a builtin schema, tighten it (pin an enum, add a field, make a field required),
and validate a dataset against it - the derive -> modify -> validate loop - without editing the
shared builtins. resolve_schema prefers a project-local schema over the builtin of the same id.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.schemas.base import DatasetSchema
from corpus_studio.schemas.project_schemas import (
    SchemaError,
    load_project_schema,
    project_schema_path,
    resolve_schema,
    save_project_schema,
)
from corpus_studio.schemas.registry import builtin_schema_dir, load_builtin_schema

runner = CliRunner()


# ---- storage module ----------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path):
    schema = load_builtin_schema("instruction").model_copy(update={"id": "myinstr"})
    path = save_project_schema(tmp_path, schema)
    assert path == project_schema_path(tmp_path, "myinstr")
    assert load_project_schema(tmp_path, "myinstr").id == "myinstr"


def test_resolve_prefers_project_over_builtin(tmp_path: Path):
    # no project copy -> builtin
    schema, source = resolve_schema(tmp_path, "instruction")
    assert source == "builtin" and schema.id == "instruction"
    # a project copy of the same id shadows it (explicitly reported)
    save_project_schema(tmp_path, load_builtin_schema("instruction").model_copy(update={"name": "Mine"}))
    schema, source = resolve_schema(tmp_path, "instruction")
    assert source == "project" and schema.name == "Mine"


def test_project_schema_path_rejects_traversal_id(tmp_path: Path):
    with pytest.raises(SchemaError):
        project_schema_path(tmp_path, "../evil")


def test_load_missing_project_schema_raises(tmp_path: Path):
    with pytest.raises(SchemaError):
        load_project_schema(tmp_path, "nope")


# ---- schema-derive -----------------------------------------------------------


def test_cli_derive_copies_builtin(tmp_path: Path):
    result = runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["id"] == "instruction" and payload["source_builtin"] == "instruction"
    assert load_project_schema(tmp_path, "instruction").id == "instruction"


def test_cli_derive_with_new_id_and_name(tmp_path: Path):
    result = runner.invoke(
        app,
        ["schema-derive", str(tmp_path), "--from", "instruction", "--id", "my_tasks", "--name", "My Tasks", "--json"],
    )
    assert result.exit_code == 0, result.output
    schema = load_project_schema(tmp_path, "my_tasks")
    assert schema.id == "my_tasks" and schema.name == "My Tasks"


def test_cli_derive_refuses_existing_without_force(tmp_path: Path):
    runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction"])
    result = runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction"])
    assert result.exit_code == 1 and "already exists" in result.output
    # --force overwrites
    forced = runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction", "--force"])
    assert forced.exit_code == 0, forced.output


def test_cli_derive_unknown_builtin(tmp_path: Path):
    result = runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "does_not_exist"])
    assert result.exit_code == 1 and "Unknown schema" in result.output


# ---- schema-set-field + the derive -> pin-enum -> validate loop --------------


def test_cli_pin_enum_then_validate_data(tmp_path: Path):
    runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction", "--id", "graded"])
    # pin an enum on the 'output' field
    edit = runner.invoke(
        app, ["schema-set-field", str(tmp_path), "graded", "--name", "output", "--enum", "yes,no", "--json"]
    )
    assert edit.exit_code == 0, edit.output
    assert json.loads(edit.stdout)["action"] == "modified"
    output_field = next(f for f in load_project_schema(tmp_path, "graded").fields if f.name == "output")
    assert output_field.enum == ["yes", "no"]

    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({"instruction": "q", "output": "yes"}) + "\n", encoding="utf-8")
    ok = runner.invoke(app, ["schema-validate", str(tmp_path), "graded", "--data", str(good), "--json"])
    assert ok.exit_code == 0, ok.output
    assert json.loads(ok.stdout)["row_error_count"] == 0

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"instruction": "q", "output": "maybe"}) + "\n", encoding="utf-8")
    fail = runner.invoke(app, ["schema-validate", str(tmp_path), "graded", "--data", str(bad)])
    assert fail.exit_code == 1  # 'maybe' is not in the pinned enum


def test_cli_add_new_field_requires_type(tmp_path: Path):
    runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction", "--id", "s1"])
    missing_type = runner.invoke(app, ["schema-set-field", str(tmp_path), "s1", "--name", "grade"])
    assert missing_type.exit_code == 1 and "requires --type" in missing_type.output
    added = runner.invoke(
        app, ["schema-set-field", str(tmp_path), "s1", "--name", "grade", "--type", "integer", "--required"]
    )
    assert added.exit_code == 0, added.output
    grade = next(f for f in load_project_schema(tmp_path, "s1").fields if f.name == "grade")
    assert grade.type == "integer" and grade.required is True


def test_cli_set_field_invalid_type_is_refused(tmp_path: Path):
    runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction", "--id", "s2"])
    before = load_project_schema(tmp_path, "s2").model_dump()
    result = runner.invoke(
        app, ["schema-set-field", str(tmp_path), "s2", "--name", "output", "--type", "not_a_type"]
    )
    assert result.exit_code == 1 and "invalid" in result.output.lower()
    assert load_project_schema(tmp_path, "s2").model_dump() == before  # unchanged


# ---- schema-validate + schema-list -------------------------------------------


def test_cli_validate_builtin_when_no_project_copy(tmp_path: Path):
    result = runner.invoke(app, ["schema-validate", str(tmp_path), "instruction", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["source"] == "builtin"


def test_cli_validate_unknown_schema(tmp_path: Path):
    result = runner.invoke(app, ["schema-validate", str(tmp_path), "ghost"])
    assert result.exit_code == 1


def test_cli_list_marks_shadowed(tmp_path: Path):
    runner.invoke(app, ["schema-derive", str(tmp_path), "--from", "instruction"])
    result = runner.invoke(app, ["schema-list", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    schemas = json.loads(result.stdout)["schemas"]
    project = [s for s in schemas if s["source"] == "project"]
    assert any(s["id"] == "instruction" for s in project)
    builtin_instruction = next(s for s in schemas if s["source"] == "builtin" and s["id"] == "instruction")
    assert builtin_instruction["shadowed_by_project"] is True


# ---- builtin invariant (locks the resolution assumption) ---------------------


def test_cli_list_surfaces_malformed_project_schema_as_active_shadow(tmp_path: Path):
    # A present-but-malformed project schema shadows the builtin (resolve/validate key on file
    # existence). schema-list must reflect that consistently, not hide the file and claim the builtin
    # is un-shadowed. Regression for the review finding.
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    (schemas_dir / "instruction.schema.json").write_text('{"id": "x", "name": "Broken"', encoding="utf-8")

    listed = runner.invoke(app, ["schema-list", str(tmp_path), "--json"])
    assert listed.exit_code == 0, listed.output
    schemas = json.loads(listed.stdout)["schemas"]
    assert {"id": "instruction", "source": "project", "malformed": True} in schemas
    builtin = next(s for s in schemas if s["source"] == "builtin" and s["id"] == "instruction")
    assert builtin["shadowed_by_project"] is True  # consistent with resolve_schema (existence-based)
    assert "malformed" in listed.stderr  # surfaced, never silent

    # ...and resolution agrees: schema-validate for that id fails on the malformed shadow.
    validated = runner.invoke(app, ["schema-validate", str(tmp_path), "instruction"])
    assert validated.exit_code == 1 and "malformed" in validated.output


def test_every_builtin_id_equals_its_filename_stem():
    # resolve_schema/schema-list assume a builtin's id matches its <id>.schema.json filename; lock it.
    for path in sorted(builtin_schema_dir().glob("*.schema.json")):
        stem = path.name.removesuffix(".schema.json")
        schema = DatasetSchema.model_validate(json.loads(path.read_text(encoding="utf-8")))
        assert schema.id == stem, f"builtin {path.name} has id '{schema.id}' != stem '{stem}'"
