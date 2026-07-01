"""Input-hardening regressions: ids that build filesystem paths must be safe,
and readers must tolerate a UTF-8 BOM."""

from pathlib import Path

import pytest

from corpus_studio.schemas.registry import load_builtin_schema
from corpus_studio.storage.project import DatasetProject, create_project
from corpus_studio.validators.basic_validator import validate_jsonl_file


@pytest.mark.parametrize("bad_id", ["../secret", "..\\secret", "a/b", "foo.bar", ".."])
def test_load_builtin_schema_rejects_unsafe_ids(bad_id: str):
    with pytest.raises(ValueError):
        load_builtin_schema(bad_id)


def test_load_builtin_schema_accepts_known_id():
    assert load_builtin_schema("instruction").id == "instruction"


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "..", "Evil", ""])
def test_create_project_rejects_unsafe_ids(tmp_path: Path, bad_id: str):
    project = DatasetProject(id=bad_id, name="x", schema_id="instruction")
    with pytest.raises(ValueError):
        create_project(tmp_path, project)


def test_create_project_accepts_safe_id(tmp_path: Path):
    project = DatasetProject(id="my_dataset-1", name="My Dataset", schema_id="instruction")
    created = create_project(tmp_path, project)
    assert created.exists()
    assert (created / "project.json").exists()


def test_validate_jsonl_file_tolerates_bom(tmp_path: Path):
    path = tmp_path / "bom.jsonl"
    path.write_text('{"instruction": "Do it", "output": "Done"}\n', encoding="utf-8-sig")
    report = validate_jsonl_file(path, "instruction")
    assert report.valid, report.model_dump()
