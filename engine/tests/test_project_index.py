import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.storage.index import (
    default_index_path,
    list_projects_from_root,
    read_project_entry,
    rebuild_index,
)

runner = CliRunner()


def _make_project(
    root: Path, project_id: str, name: str, schema_id: str, example_count: int
) -> Path:
    project_dir = root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project.json").write_text(
        json.dumps(
            {
                "id": project_id,
                "name": name,
                "schema_id": schema_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (project_dir / "examples.jsonl").write_text(
        "".join(json.dumps({"row": index}) + "\n" for index in range(example_count)),
        encoding="utf-8",
    )
    return project_dir


def test_rebuild_index_counts_projects_and_examples(tmp_path: Path):
    root = tmp_path / "projects"
    _make_project(root, "alpha", "Alpha Set", "instruction", 3)
    _make_project(root, "beta", "Beta Set", "chat", 1)

    count = rebuild_index(root)
    assert count == 2

    entries = list_projects_from_root(root)
    by_id = {entry.id: entry for entry in entries}
    assert by_id["alpha"].example_count == 3
    assert by_id["beta"].schema_id == "chat"
    assert [entry.id for entry in entries] == ["alpha", "beta"]  # ordered by name


def test_index_file_is_separate_and_leaves_json_untouched(tmp_path: Path):
    root = tmp_path / "projects"
    project_dir = _make_project(root, "alpha", "Alpha", "instruction", 2)
    original = (project_dir / "project.json").read_text(encoding="utf-8")

    rebuild_index(root)

    assert default_index_path(root).exists()
    assert (project_dir / "project.json").read_text(encoding="utf-8") == original


def test_list_projects_filters_by_schema_and_name(tmp_path: Path):
    root = tmp_path / "projects"
    _make_project(root, "alpha", "Customer Support", "instruction", 1)
    _make_project(root, "beta", "Code Review", "instruction", 1)
    _make_project(root, "gamma", "Chat Logs", "chat", 1)
    rebuild_index(root)

    instruction = list_projects_from_root(root, schema_id="instruction")
    assert {entry.id for entry in instruction} == {"alpha", "beta"}

    named = list_projects_from_root(root, name_contains="chat")
    assert {entry.id for entry in named} == {"gamma"}


def test_rebuild_prunes_removed_projects(tmp_path: Path):
    root = tmp_path / "projects"
    _make_project(root, "alpha", "Alpha", "instruction", 1)
    beta_dir = _make_project(root, "beta", "Beta", "instruction", 1)
    rebuild_index(root)
    assert {entry.id for entry in list_projects_from_root(root)} == {"alpha", "beta"}

    shutil.rmtree(beta_dir)
    rebuild_index(root)
    assert {entry.id for entry in list_projects_from_root(root)} == {"alpha"}


def test_rebuild_to_empty_prunes_all_projects(tmp_path: Path):
    root = tmp_path / "projects"
    alpha_dir = _make_project(root, "alpha", "Alpha", "instruction", 1)
    rebuild_index(root)
    assert len(list_projects_from_root(root)) == 1

    # Removing the last project must clear the index; the DELETE has to commit
    # even when the rebuild finds zero projects to upsert.
    shutil.rmtree(alpha_dir)
    assert rebuild_index(root) == 0
    assert list_projects_from_root(root) == []


def test_list_auto_rebuilds_when_index_missing(tmp_path: Path):
    root = tmp_path / "projects"
    _make_project(root, "alpha", "Alpha", "instruction", 2)
    assert not default_index_path(root).exists()

    entries = list_projects_from_root(root)
    assert default_index_path(root).exists()
    assert {entry.id for entry in entries} == {"alpha"}


def test_reindex_updates_example_count(tmp_path: Path):
    root = tmp_path / "projects"
    project_dir = _make_project(root, "alpha", "Alpha", "instruction", 1)
    rebuild_index(root)
    assert list_projects_from_root(root)[0].example_count == 1

    (project_dir / "examples.jsonl").write_text(
        "".join(json.dumps({"row": index}) + "\n" for index in range(5)),
        encoding="utf-8",
    )
    rebuild_index(root)
    assert list_projects_from_root(root)[0].example_count == 5


def test_read_project_entry_returns_none_without_metadata(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert read_project_entry(empty_dir) is None


def test_cli_project_index_rebuild_and_list(tmp_path: Path):
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "alpha", "Alpha Set", "instruction", "--root", str(root)])
    runner.invoke(app, ["new-project", "beta", "Beta Set", "chat", "--root", str(root)])

    rebuilt = runner.invoke(app, ["project-index-rebuild", "--root", str(root)])
    assert rebuilt.exit_code == 0
    assert json.loads(rebuilt.output)["indexed"] == 2

    listed = runner.invoke(app, ["project-list", "--root", str(root)])
    assert listed.exit_code == 0
    payload = json.loads(listed.output)
    assert payload["count"] == 2
    assert {project["id"] for project in payload["projects"]} == {"alpha", "beta"}

    filtered = runner.invoke(app, ["project-list", "--root", str(root), "--schema", "chat"])
    assert filtered.exit_code == 0
    assert {project["id"] for project in json.loads(filtered.output)["projects"]} == {"beta"}


def test_cli_new_project_opt_in_index(tmp_path: Path, monkeypatch):
    root = tmp_path / "projects"
    monkeypatch.setenv("CORPUS_STUDIO_USE_INDEX", "1")

    result = runner.invoke(
        app, ["new-project", "alpha", "Alpha", "instruction", "--root", str(root)]
    )
    assert result.exit_code == 0
    assert default_index_path(root).exists()
    assert {entry.id for entry in list_projects_from_root(root)} == {"alpha"}


def test_cli_new_project_without_opt_in_skips_index(tmp_path: Path):
    root = tmp_path / "projects"
    result = runner.invoke(
        app, ["new-project", "alpha", "Alpha", "instruction", "--root", str(root)]
    )
    assert result.exit_code == 0
    assert not default_index_path(root).exists()
