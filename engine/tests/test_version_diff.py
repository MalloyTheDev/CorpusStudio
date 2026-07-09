import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.versions.version_diff import (
    diff_manifests,
    render_dataset_version_diff_markdown,
)

runner = CliRunner()


# --- pure multiset diff ------------------------------------------------------

def test_diff_added_removed_common():
    base = ["a", "b", "c"]
    other = ["b", "c", "d"]
    diff = diff_manifests(base, other, "v1", "v2")
    assert diff.added_count == 1 and diff.added_row_ids == ["d"]
    assert diff.removed_count == 1 and diff.removed_row_ids == ["a"]
    assert diff.common_count == 2


def test_diff_is_multiset_aware():
    # base has "a" twice, other once -> one "a" removed.
    diff = diff_manifests(["a", "a", "b"], ["a", "b"], "v1", "v2")
    assert diff.removed_count == 1 and diff.removed_row_ids == ["a"]
    assert diff.added_count == 0
    assert diff.common_count == 2


def test_diff_reorder_is_not_a_content_change_but_is_flagged():
    # Same rows, different order (#196): no add/remove, but reordered=True + moved positions.
    diff = diff_manifests(["a", "b", "c"], ["c", "a", "b"], "v1", "v2")
    assert diff.added_count == 0 and diff.removed_count == 0
    assert diff.common_count == 3
    assert diff.reordered is True
    assert diff.moved_count == 3  # all three positions changed


def test_diff_identical_order_is_not_reordered():
    diff = diff_manifests(["a", "b", "c"], ["a", "b", "c"], "v1", "v2")
    assert diff.reordered is False
    assert diff.moved_count == 0


def test_diff_partial_reorder_counts_moved_positions():
    # "a" stays put; "b" and "c" swap → 2 positions moved.
    diff = diff_manifests(["a", "b", "c"], ["a", "c", "b"], "v1", "v2")
    assert diff.reordered is True
    assert diff.moved_count == 2


def test_content_change_is_not_a_pure_reorder():
    diff = diff_manifests(["a", "b", "c"], ["c", "b", "d"], "v1", "v2")
    assert diff.added_count == 1 and diff.removed_count == 1
    assert diff.reordered is False  # content changed, so not a *pure* reorder


def test_render_flags_a_reorder():
    diff = diff_manifests(["a", "b", "c"], ["c", "a", "b"], "v1", "v2")
    markdown = render_dataset_version_diff_markdown(diff)
    assert "Reordered" in markdown and "3 position" in markdown


def test_render_diff_is_injection_safe():
    diff = diff_manifests(["a"], ["b"], "v1\n> pwn", "v2")
    markdown = render_dataset_version_diff_markdown(diff, sample_added=[{"x": "1\n2"}])
    assert "\n> pwn" not in markdown
    assert "Added" in markdown


# --- CLI end-to-end ----------------------------------------------------------

def _write_examples(project: Path, rows: list[dict]) -> None:
    (project / "examples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _create(project: Path, *extra: str) -> str:
    result = runner.invoke(app, ["dataset-version-create", str(project), *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["version_id"]


def test_cli_create_stores_rows_then_diff_shows_added(tmp_path: Path):
    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}])
    v1 = _create(tmp_path)

    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}, {"instruction": "B", "output": "2"}])
    v2 = _create(tmp_path)

    result = runner.invoke(
        app, ["dataset-version-diff", str(tmp_path), "--version-id", v1, "--other", v2, "--json"]
    )
    assert result.exit_code == 0, result.output
    diff = json.loads(result.stdout)
    assert diff["added_count"] == 1
    assert diff["removed_count"] == 0
    assert diff["common_count"] == 1


def test_cli_diff_markdown_shows_added_sample(tmp_path: Path):
    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}])
    v1 = _create(tmp_path)
    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}, {"instruction": "NEW", "output": "row"}])
    v2 = _create(tmp_path)

    result = runner.invoke(app, ["dataset-version-diff", str(tmp_path), "--version-id", v1, "--other", v2])
    assert result.exit_code == 0, result.output
    assert "Added" in result.stdout
    assert "NEW" in result.stdout  # sample row body pulled from the store


def test_cli_diff_refuses_when_rows_not_stored(tmp_path: Path):
    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}])
    stored = _create(tmp_path)
    not_stored = _create(tmp_path, "--no-store-rows")

    result = runner.invoke(
        app, ["dataset-version-diff", str(tmp_path), "--version-id", stored, "--other", not_stored]
    )
    assert result.exit_code == 1
    assert "no stored rows" in result.stderr.lower()


def test_cli_diff_refuses_corrupt_record(tmp_path: Path):
    from corpus_studio.versions.version_registry import record_path

    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}])
    v1 = _create(tmp_path)
    v2 = _create(tmp_path)
    record_path(tmp_path, v1).write_text("not json {{{", encoding="utf-8")

    result = runner.invoke(
        app, ["dataset-version-diff", str(tmp_path), "--version-id", v1, "--other", v2]
    )
    assert result.exit_code == 1
    assert "corrupt record" in result.stderr.lower()  # clean message, not a traceback


def test_cli_no_store_rows_sets_flags(tmp_path: Path):
    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}])
    result = runner.invoke(app, ["dataset-version-create", str(tmp_path), "--no-store-rows"])
    record = json.loads(result.stdout)
    assert record["rows_stored"] is False
    assert record["stored_row_count"] == 0
    assert record["row_manifest_algo"] is None
    # no manifest sidecar written
    assert not list((tmp_path / "dataset_versions").glob("*.rows"))


def test_cli_store_rows_default_sets_flags_and_manifest(tmp_path: Path):
    _write_examples(tmp_path, [{"instruction": "A", "output": "1"}, {"instruction": "B", "output": "2"}])
    result = runner.invoke(app, ["dataset-version-create", str(tmp_path)])
    record = json.loads(result.stdout)
    assert record["rows_stored"] is True
    assert record["stored_row_count"] == 2
    assert record["row_manifest_algo"] == "sha256-exact-v1"
    assert "stored 2 row(s)" in result.stderr.lower()
