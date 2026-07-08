"""Tests for row-store garbage collection (issue #197).

The invariant under test: GC never removes a row referenced by any version manifest, and aborts
rather than prune when it can't read the full picture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus_studio.versions.gc import collect_referenced_row_ids, gc_row_store
from corpus_studio.versions.row_store import (
    load_rows_by_id,
    row_id,
    row_store_path,
    store_line,
)
from corpus_studio.versions.version_registry import (
    ROW_MANIFEST_SUFFIX,
    registry_dir,
    save_row_manifest,
)

_A = {"instruction": "a", "output": "1"}
_B = {"instruction": "b", "output": "2"}
_C = {"instruction": "c", "output": "3"}


def _write_store(project_dir: Path, rows: list[dict]) -> None:
    path = row_store_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(store_line(row_id(r), r) + "\n" for r in rows), encoding="utf-8")


def test_gc_prunes_unreferenced_and_keeps_referenced(tmp_path: Path) -> None:
    _write_store(tmp_path, [_A, _B, _C])
    save_row_manifest(tmp_path, "v1", [row_id(_A), row_id(_C)])

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1
    assert result.kept_rows == 2
    present = load_rows_by_id(tmp_path, {row_id(_A), row_id(_B), row_id(_C)})
    assert row_id(_A) in present
    assert row_id(_C) in present
    assert row_id(_B) not in present  # the only unreferenced row


def test_gc_keeps_rows_referenced_by_any_version(tmp_path: Path) -> None:
    _write_store(tmp_path, [_A, _B, _C])
    save_row_manifest(tmp_path, "v1", [row_id(_A)])
    save_row_manifest(tmp_path, "v2", [row_id(_B)])

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1  # only C, referenced by neither version
    present = load_rows_by_id(tmp_path, {row_id(_A), row_id(_B), row_id(_C)})
    assert row_id(_A) in present and row_id(_B) in present
    assert row_id(_C) not in present


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    _write_store(tmp_path, [_A, _B])
    save_row_manifest(tmp_path, "v1", [row_id(_A)])
    before = row_store_path(tmp_path).read_text(encoding="utf-8")

    result = gc_row_store(tmp_path, dry_run=True)

    assert result.pruned_rows == 1
    assert row_store_path(tmp_path).read_text(encoding="utf-8") == before  # untouched


def test_no_manifests_means_every_stored_row_is_orphaned(tmp_path: Path) -> None:
    _write_store(tmp_path, [_A, _B])
    result = gc_row_store(tmp_path)
    assert result.pruned_rows == 2
    assert result.kept_rows == 0


def test_unreadable_manifest_aborts_without_pruning(tmp_path: Path) -> None:
    _write_store(tmp_path, [_A, _B])
    save_row_manifest(tmp_path, "v1", [row_id(_A)])
    # A manifest we cannot read (a directory where a *.rows file is expected). GC must abort, not
    # prune on an incomplete live set.
    (registry_dir(tmp_path) / f"corrupt{ROW_MANIFEST_SUFFIX}").mkdir()
    before = row_store_path(tmp_path).read_text(encoding="utf-8")

    with pytest.raises(OSError):
        gc_row_store(tmp_path)

    assert row_store_path(tmp_path).read_text(encoding="utf-8") == before  # nothing removed


def test_torn_store_line_is_kept_never_pruned(tmp_path: Path) -> None:
    _write_store(tmp_path, [_A])
    with row_store_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write('{"row_id": "x", "row": {"broken\n')  # invalid JSON — cannot classify
    save_row_manifest(tmp_path, "v1", [row_id(_A)])

    gc_row_store(tmp_path)

    text = row_store_path(tmp_path).read_text(encoding="utf-8")
    assert '{"row_id": "x", "row": {"broken' in text  # unclassifiable line preserved
    assert row_id(_A) in text


def test_collect_referenced_row_ids_unions_all_manifests(tmp_path: Path) -> None:
    save_row_manifest(tmp_path, "v1", [row_id(_A), row_id(_B)])
    save_row_manifest(tmp_path, "v2", [row_id(_B), row_id(_C)])
    assert collect_referenced_row_ids(tmp_path) == {row_id(_A), row_id(_B), row_id(_C)}


def test_gc_handles_a_bom_prefixed_store(tmp_path: Path) -> None:
    # A store saved with a BOM (e.g. by a Windows editor) must still classify + prune correctly.
    path = row_store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(store_line(row_id(r), r) + "\n" for r in [_A, _B]), encoding="utf-8-sig"
    )
    save_row_manifest(tmp_path, "v1", [row_id(_A)])

    result = gc_row_store(tmp_path)

    assert result.pruned_rows == 1  # B pruned despite the BOM on the first line
    present = load_rows_by_id(tmp_path, {row_id(_A), row_id(_B)})
    assert row_id(_A) in present and row_id(_B) not in present
