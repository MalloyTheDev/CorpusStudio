"""Row-store GC splits the store on newlines only.

``json.dumps(ensure_ascii=False)`` writes U+2028, U+2029 and U+0085 raw inside JSON strings, and
``str.splitlines()`` also breaks on them. A GC that split the store with ``splitlines()`` cut such a
referenced row into two unparseable fragments, kept them as "unclassifiable" lines, and (whenever it
pruned anything) rewrote the store without the row: the version referencing it could never be
restored again. The store's readers (``load_row_id_set`` / ``load_rows_by_id``) split on newlines
only; GC must agree with them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.versions.gc import IncompleteReferenceScanError, gc_row_store
from corpus_studio.versions.row_store import load_row_id_set, row_id, row_store_path, store_line
from corpus_studio.versions.version_registry import create_dataset_version, manifest_path
from corpus_studio.versions.version_restore import reconstruct_version_lines

runner = CliRunner()

_SEPARATORS = {"LINE SEPARATOR": "\u2028", "PARAGRAPH SEPARATOR": "\u2029", "NEL": "\u0085"}
_ORPHAN = {"instruction": "orphan", "output": "pruned"}


def _write_examples(project: Path, rows: list[dict[str, str]]) -> None:
    (project / "examples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _append_orphan(project: Path) -> None:
    with row_store_path(project).open("a", encoding="utf-8") as handle:
        handle.write(store_line(row_id(_ORPHAN), _ORPHAN))


@pytest.mark.parametrize("separator", list(_SEPARATORS.values()), ids=list(_SEPARATORS))
def test_gc_keeps_a_referenced_row_containing_a_unicode_line_break(
    tmp_path: Path, separator: str
) -> None:
    row = {"instruction": f"first line{separator}second line", "output": "plain"}
    _write_examples(tmp_path, [row, {"instruction": "plain row", "output": "x"}])
    keep = create_dataset_version(tmp_path, label="keep")
    assert separator in row_store_path(tmp_path).read_text(encoding="utf-8")  # stored raw
    _append_orphan(tmp_path)  # something to prune, so GC rewrites the store

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1
    assert result.kept_rows == 2
    assert row_id(row) in load_row_id_set(tmp_path)
    assert len(reconstruct_version_lines(tmp_path, keep.version_id)) == 2  # fingerprint verified


def test_gc_then_restore_via_cli_with_every_unicode_line_break(tmp_path: Path) -> None:
    text = "a\u2028b\u2029c\u0085d"
    _write_examples(tmp_path, [{"instruction": text, "output": "x"}])
    created = runner.invoke(app, ["dataset-version-create", str(tmp_path), "--label", "keep"])
    assert created.exit_code == 0, created.output
    version_id = json.loads(created.stdout)["version_id"]
    _append_orphan(tmp_path)

    gc = runner.invoke(app, ["dataset-version-gc", str(tmp_path), "--json"])
    assert gc.exit_code == 0, gc.output
    assert json.loads(gc.stdout)["pruned_rows"] == 1

    output = tmp_path / "restored.jsonl"
    restored = runner.invoke(
        app,
        ["dataset-version-restore", str(tmp_path), "--version-id", version_id, "--output", str(output)],
    )
    assert restored.exit_code == 0, restored.output
    assert json.loads(output.read_text(encoding="utf-8"))["instruction"] == text


def test_gc_classifies_a_crlf_terminated_store(tmp_path: Path) -> None:
    _write_examples(tmp_path, [{"instruction": "a", "output": "1"}])
    keep = create_dataset_version(tmp_path, label="keep")
    _append_orphan(tmp_path)
    store = row_store_path(tmp_path)
    store.write_bytes(store.read_bytes().replace(b"\n", b"\r\n"))  # e.g. re-saved on Windows

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1 and result.kept_rows == 1
    assert reconstruct_version_lines(tmp_path, keep.version_id)


def test_a_unicode_line_break_inside_a_manifest_line_is_a_torn_line(tmp_path: Path) -> None:
    # Manifests are split on newlines only as well: a stray separator does not turn one corrupt
    # line into two row ids GC would trust.
    _write_examples(tmp_path, [{"instruction": "a", "output": "1"}])
    create_dataset_version(tmp_path, label="keep")
    first, second = row_id({"n": 1}), row_id({"n": 2})
    corrupt = manifest_path(tmp_path, "20260101T000000-corrupt")
    corrupt.write_text(f"{first}\u2028{second}\n", encoding="utf-8")
    before = row_store_path(tmp_path).read_bytes()

    with pytest.raises(IncompleteReferenceScanError, match="line 1 is not a sha256 row id"):
        gc_row_store(tmp_path)
    assert row_store_path(tmp_path).read_bytes() == before
