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


# --- project lifecycle: delete / rename + list --rollup (#582, G7) ----------------------------

INSTR = [{"instruction": f"do task {n}", "output": f"result {n} is complete"} for n in range(3)]


def _seed(root: Path, project_id: str, rows: list[dict]) -> None:
    (root / project_id / "examples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_project_list_rollup_adds_debt_grade(tmp_path: Path):
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "full", "Full", "instruction", "--root", str(root)])
    runner.invoke(app, ["new-project", "empty", "Empty", "instruction", "--root", str(root)])
    _seed(root, "full", INSTR)

    result = runner.invoke(app, ["project-list", "--root", str(root), "--rollup"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rollup"] is True
    by_id = {p["id"]: p for p in payload["projects"]}
    assert by_id["full"]["has_data"] is True and by_id["full"]["debt_grade"] != "N/A"
    assert by_id["empty"]["has_data"] is False and by_id["empty"]["debt_grade"] == "N/A"
    # without --rollup, no debt fields are added
    plain = json.loads(runner.invoke(app, ["project-list", "--root", str(root)]).output)
    assert "debt_grade" not in plain["projects"][0]


def test_project_delete_requires_yes_and_removes_folder(tmp_path: Path):
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "gone", "Gone", "instruction", "--root", str(root)])
    # refused without --yes; folder still there
    refused = runner.invoke(app, ["project-delete", "gone", "--root", str(root)])
    assert refused.exit_code == 1 and "--yes" in refused.output
    assert (root / "gone").is_dir()
    # deleted with --yes
    deleted = runner.invoke(app, ["project-delete", "gone", "--root", str(root), "--yes"])
    assert deleted.exit_code == 0, deleted.output
    assert not (root / "gone").exists()


def test_project_delete_refuses_non_project(tmp_path: Path):
    root = tmp_path / "projects"
    root.mkdir(parents=True)
    (root / "notaproject").mkdir()  # a dir with no project.json
    result = runner.invoke(app, ["project-delete", "notaproject", "--root", str(root), "--yes"])
    assert result.exit_code == 1 and "no project.json" in result.output
    assert (root / "notaproject").is_dir()  # untouched


def test_project_rename_display_name(tmp_path: Path):
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "proj", "Old Name", "instruction", "--root", str(root)])
    result = runner.invoke(app, ["project-rename", "proj", "--name", "New Name", "--root", str(root)])
    assert result.exit_code == 0, result.output
    stored = json.loads((root / "proj" / "project.json").read_text())
    assert stored["name"] == "New Name" and stored["id"] == "proj"


def test_project_rename_id_moves_folder(tmp_path: Path):
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "oldid", "P", "instruction", "--root", str(root)])
    _seed(root, "oldid", INSTR)
    result = runner.invoke(app, ["project-rename", "oldid", "--to", "newid", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert not (root / "oldid").exists() and (root / "newid" / "examples.jsonl").exists()
    stored = json.loads((root / "newid" / "project.json").read_text())
    assert stored["id"] == "newid"


def test_project_rename_refuses_existing_target_and_empty(tmp_path: Path):
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "a", "A", "instruction", "--root", str(root)])
    runner.invoke(app, ["new-project", "b", "B", "instruction", "--root", str(root)])
    clash = runner.invoke(app, ["project-rename", "a", "--to", "b", "--root", str(root)])
    assert clash.exit_code == 1 and "already exists" in clash.output
    assert (root / "a").is_dir()  # unchanged
    nothing = runner.invoke(app, ["project-rename", "a", "--root", str(root)])
    assert nothing.exit_code == 1 and "Nothing to rename" in nothing.output


def test_project_delete_and_rename_update_the_index(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CORPUS_STUDIO_USE_INDEX", "1")
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "keep", "Keep", "instruction", "--root", str(root)])
    runner.invoke(app, ["new-project", "drop", "Drop", "instruction", "--root", str(root)])
    runner.invoke(app, ["project-delete", "drop", "--root", str(root), "--yes"])
    assert {entry.id for entry in list_projects_from_root(root)} == {"keep"}
    runner.invoke(app, ["project-rename", "keep", "--to", "kept", "--root", str(root)])
    assert {entry.id for entry in list_projects_from_root(root)} == {"kept"}


def test_rename_never_fabricates_a_partial_index(tmp_path: Path, monkeypatch):
    # Regression: projects created WITHOUT an index, then a rename must NOT seed a one-entry index
    # that makes project-list silently hide the other on-disk projects.
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "foo", "Foo", "instruction", "--root", str(root)])
    runner.invoke(app, ["new-project", "bar", "Bar", "instruction", "--root", str(root)])
    assert not default_index_path(root).exists()
    monkeypatch.setenv("CORPUS_STUDIO_USE_INDEX", "1")
    assert runner.invoke(app, ["project-rename", "foo", "--to", "qux", "--root", str(root)]).exit_code == 0
    listed = json.loads(runner.invoke(app, ["project-list", "--root", str(root)]).output)
    assert {p["id"] for p in listed["projects"]} == {"bar", "qux"}  # never a partial {qux}


def test_lifecycle_refreshes_an_existing_index_even_with_flag_unset(tmp_path: Path):
    # Regression: if an index file exists, a delete must refresh it (not leave a ghost row) even when
    # CORPUS_STUDIO_USE_INDEX is unset - project-list reads any existing index.
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "keep", "K", "instruction", "--root", str(root)])
    runner.invoke(app, ["new-project", "drop", "D", "instruction", "--root", str(root)])
    runner.invoke(app, ["project-index-rebuild", "--root", str(root)])
    assert default_index_path(root).exists()
    runner.invoke(app, ["project-delete", "drop", "--root", str(root), "--yes"])
    listed = json.loads(runner.invoke(app, ["project-list", "--root", str(root)]).output)
    assert {p["id"] for p in listed["projects"]} == {"keep"}  # no deleted ghost


def test_project_delete_refuses_a_symlinked_project_dir(tmp_path: Path):
    # A destructive op must not follow a symlink (rmtree would fail on it anyway; refuse cleanly).
    root = tmp_path / "projects"
    runner.invoke(app, ["new-project", "real", "R", "instruction", "--root", str(root)])
    (root / "link").symlink_to(root / "real")
    result = runner.invoke(app, ["project-delete", "link", "--root", str(root), "--yes"])
    assert result.exit_code == 1 and "symlink" in result.output
    assert (root / "real").is_dir()  # the real project is untouched
