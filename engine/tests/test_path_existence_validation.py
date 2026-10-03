"""The opt-in path-existence check on ``file_path`` / ``image_path`` fields.

Opt-in on BOTH sides, on purpose: the SCHEMA declares ``path_must_exist`` (only its author knows
whether the paths are meant to resolve yet) and the CALLER supplies the root (a staging lane
legitimately references files that have not been fetched, and guessing the root from the project
directory would make the check fire where it was never asked for). Either half missing leaves the
field the string-only check it has always been - and ``schema-validate`` says which declared checks
it skipped, so a validated file is never mistaken for one whose paths were confirmed.

The path value is UNTRUSTED dataset content, so the root is a trust boundary: an absolute path, a
``..`` climb, or a symlink pointing out of the root is REJECTED rather than probed. Without that,
validating a third-party JSONL file would let its rows test for arbitrary paths on the machine
running the validator.
"""

import json
import os
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.schemas.base import DatasetSchema
from corpus_studio.validators.basic_validator import validate_example_fields_against

runner = CliRunner()

_SCHEMA: dict[str, Any] = {
    "id": "with_image",
    "name": "With image",
    "version": "1.0.0",
    "fields": [
        {"name": "image", "type": "image_path", "required": True, "path_must_exist": True},
        {"name": "caption", "type": "text", "required": False},
    ],
}


def _schema(**overrides: Any) -> DatasetSchema:
    payload = json.loads(json.dumps(_SCHEMA))
    payload["fields"][0].update(overrides)
    return DatasetSchema.model_validate(payload)


def _lane(tmp_path: Path) -> Path:
    root = tmp_path / "lane"
    (root / "images").mkdir(parents=True)
    (root / "images" / "present.jpg").write_bytes(b"")
    return root


# --- the check itself ----------------------------------------------------------------

def test_existing_path_passes(tmp_path: Path):
    issues = validate_example_fields_against(
        {"image": "images/present.jpg"}, _schema(), 1, _lane(tmp_path)
    )
    assert issues == []


def test_missing_path_fails_and_names_the_field(tmp_path: Path):
    issues = validate_example_fields_against(
        {"image": "images/absent.jpg"}, _schema(), 1, _lane(tmp_path)
    )
    assert len(issues) == 1
    assert issues[0].field == "image"
    assert issues[0].row_number == 1
    assert "does not exist under the path root" in issues[0].message


def test_a_directory_is_not_a_file(tmp_path: Path):
    issues = validate_example_fields_against({"image": "images"}, _schema(), 1, _lane(tmp_path))
    assert len(issues) == 1
    assert "does not exist under the path root" in issues[0].message


# --- opt-in on both sides ------------------------------------------------------------

def test_no_root_leaves_the_declared_check_inert(tmp_path: Path):
    # A staging file legitimately names files that are not fetched yet.
    assert validate_example_fields_against({"image": "images/absent.jpg"}, _schema(), 1) == []


def test_a_schema_that_does_not_opt_in_is_never_checked(tmp_path: Path):
    schema = _schema(path_must_exist=False)
    issues = validate_example_fields_against(
        {"image": "images/absent.jpg"}, schema, 1, _lane(tmp_path)
    )
    assert issues == []


def test_path_must_exist_defaults_to_false():
    schema = DatasetSchema.model_validate(
        {"id": "d", "name": "d", "version": "1", "fields": [{"name": "p", "type": "file_path"}]}
    )
    assert schema.fields[0].path_must_exist is False


# --- the root is a trust boundary ----------------------------------------------------

def test_absolute_paths_are_rejected_not_probed(tmp_path: Path):
    root = _lane(tmp_path)
    for value in ("/etc/passwd", "C:\\Windows\\win.ini"):
        issues = validate_example_fields_against({"image": value}, _schema(), 1, root)
        assert len(issues) == 1, value
        assert "must be relative to the path root" in issues[0].message, value


def test_parent_traversal_is_rejected(tmp_path: Path):
    root = _lane(tmp_path)
    (tmp_path / "outside.jpg").write_bytes(b"")
    for value in ("../outside.jpg", "../../etc/passwd", "..\\outside.jpg"):
        issues = validate_example_fields_against({"image": value}, _schema(), 1, root)
        assert len(issues) == 1, value
        assert "escapes the path root" in issues[0].message, value


def test_a_symlink_out_of_the_root_is_rejected(tmp_path: Path):
    root = _lane(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"")
    os.symlink(outside, root / "escape")
    issues = validate_example_fields_against({"image": "escape/secret.jpg"}, _schema(), 1, root)
    assert len(issues) == 1
    assert "escapes the path root" in issues[0].message


def test_traversal_that_re_enters_the_root_is_allowed(tmp_path: Path):
    # Containment is about where the path LANDS, not whether it contains "..".
    root = _lane(tmp_path)
    issues = validate_example_fields_against(
        {"image": "images/../images/present.jpg"}, _schema(), 1, root
    )
    assert issues == []


# --- nested inside a list of objects -------------------------------------------------

def test_the_check_reaches_a_path_inside_a_list_of_objects(tmp_path: Path):
    schema = DatasetSchema.model_validate({
        "id": "frames", "name": "Frames", "version": "1",
        "fields": [{
            "name": "frames", "type": "list", "item_type": "object", "required": True,
            "item_fields": [
                {"name": "path", "type": "image_path", "required": True, "path_must_exist": True}
            ],
        }],
    })
    row = {"frames": [{"path": "images/present.jpg"}, {"path": "images/absent.jpg"}]}
    issues = validate_example_fields_against(row, schema, 7, _lane(tmp_path))
    assert len(issues) == 1
    assert issues[0].field == "frames[2].path"
    assert issues[0].row_number == 7


# --- the CLI surface -----------------------------------------------------------------

def _project(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    assert runner.invoke(
        app, ["new-project", "p", "P", "image_caption", "--root", str(root)]
    ).exit_code == 0
    schemas = root / "p" / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "with_image.schema.json").write_text(json.dumps(_SCHEMA), encoding="utf-8")
    return root / "p"


def test_cli_reports_the_skipped_check_when_no_root_is_given(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps({"image": "images/absent.jpg"}) + "\n", encoding="utf-8")

    result = runner.invoke(
        app, ["schema-validate", str(project_dir), "with_image", "--data", str(data)]
    )
    assert result.exit_code == 0, result.output
    assert "Path existence NOT checked for: image" in result.stdout

    as_json = runner.invoke(
        app, ["schema-validate", str(project_dir), "with_image", "--data", str(data), "--json"]
    )
    assert json.loads(as_json.stdout)["path_existence"] == {
        "root": None, "checked_fields": [], "skipped_fields": ["image"]
    }


def test_cli_with_path_root_fails_the_missing_file(tmp_path: Path):
    project_dir = _project(tmp_path)
    root = _lane(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text(
        json.dumps({"image": "images/present.jpg"}) + "\n"
        + json.dumps({"image": "images/absent.jpg"}) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["schema-validate", str(project_dir), "with_image", "--data", str(data),
         "--path-root", str(root), "--json"],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["path_existence"]["checked_fields"] == ["image"]
    assert payload["path_existence"]["skipped_fields"] == []
    assert payload["row_errors"] == [
        {
            "row_number": 2,
            "field": "image",
            "message": "Declared path does not exist under the path root: images/absent.jpg",
        }
    ]


def test_cli_rejects_a_path_root_that_does_not_exist(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps({"image": "images/present.jpg"}) + "\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["schema-validate", str(project_dir), "with_image", "--data", str(data),
         "--path-root", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.stderr
