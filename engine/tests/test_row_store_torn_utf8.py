"""A row-store tear inside a multi-byte UTF-8 character stays recoverable (#859).

A capture killed mid-append (SIGKILL, OOM, power loss) leaves the row store cut at an arbitrary
byte: the buffered writer flushes fixed-size chunks, so for non-ASCII rows the cut usually lands
inside a multi-byte character. The next capture newline-terminates that fragment at the byte level,
and every store reader decodes the store line by line, skipping an undecodable line exactly like a
torn one. Capture, diff, restore, the examples-mutation undo and import-commit therefore keep
working with no manual repair; the damaged line never hides the rows around it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.versions.gc import gc_row_store
from corpus_studio.versions.row_store import (
    load_row_id_set,
    load_rows_by_id,
    row_id,
    row_store_path,
    store_line,
    terminate_torn_tail,
)
from corpus_studio.versions.version_registry import (
    create_dataset_version,
    fingerprint_dataset,
    list_version_records,
)
from corpus_studio.versions.version_restore import reconstruct_version_lines

runner = CliRunner()

# Escaped so this file stays ASCII: 3-byte CJK characters and a 4-byte emoji in UTF-8.
ROW_A = {"instruction": "\u65e5\u672c\u8a9e", "output": "a"}
ROW_B = {"instruction": "\u4e2d\u6587", "output": "b"}
ROW_C = {"instruction": "\u6f22\u5b57", "output": "c"}
ROW_EMOJI = {"instruction": "smile \U0001f600", "output": "e"}


def _project(project: Path) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    (project / "project.json").write_text(
        json.dumps({"id": "p", "name": "P", "schema_id": "instruction"}), encoding="utf-8"
    )
    return project


def _write_examples(project: Path, rows: list[dict[str, Any]]) -> None:
    (project / "examples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _torn_line(row: dict[str, Any], char: str, bytes_in: int) -> bytes:
    """``row``'s store line cut ``bytes_in`` bytes into the first ``char`` (a dead writer's tail)."""

    line = store_line(row_id(row), row).encode("utf-8")
    start = line.index(char.encode("utf-8"))
    return line[: start + bytes_in]


def _tear(project: Path, row: dict[str, Any], char: str, bytes_in: int = 1) -> bytes:
    fragment = _torn_line(row, char, bytes_in)
    with row_store_path(project).open("ab") as handle:
        handle.write(fragment)
    return fragment


def _create(project: Path, label: str) -> str:
    result = runner.invoke(app, ["dataset-version-create", str(project), "--label", label])
    assert result.exit_code == 0, result.output
    return str(json.loads(result.stdout)["version_id"])


# --- the byte-level tail repair ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "char", "bytes_in"),
    [
        (ROW_B, "\u4e2d", 1),
        (ROW_B, "\u4e2d", 2),
        (ROW_EMOJI, "\U0001f600", 1),
        (ROW_EMOJI, "\U0001f600", 3),
    ],
    ids=["cjk-1-of-3", "cjk-2-of-3", "emoji-1-of-4", "emoji-3-of-4"],
)
def test_a_tear_inside_a_character_is_terminated_and_skipped(
    tmp_path: Path, row: dict[str, Any], char: str, bytes_in: int
) -> None:
    store = row_store_path(tmp_path)
    store.parent.mkdir(parents=True)
    store.write_bytes(store_line(row_id(ROW_A), ROW_A).encode("utf-8"))
    fragment = _tear(tmp_path, row, char, bytes_in)
    with pytest.raises(UnicodeDecodeError):
        fragment.decode("utf-8")  # the precondition: the tail really is undecodable

    assert terminate_torn_tail(store) is True

    assert store.read_bytes().endswith(fragment + b"\n")
    assert load_row_id_set(tmp_path) == {row_id(ROW_A)}
    assert load_rows_by_id(tmp_path, {row_id(ROW_A), row_id(row)}) == {row_id(ROW_A): ROW_A}


# --- every capture and reader path keeps working ----------------------------------------------


def test_capture_diff_and_restore_survive_a_tear_inside_a_character(tmp_path: Path) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A])
    version_a = _create(project, "A")
    fragment = _tear(project, ROW_B, "\u4e2d")
    _write_examples(project, [ROW_A, ROW_C])

    version_b = _create(project, "B")

    records = {record.version_id: record for record in list_version_records(project)}
    assert records[version_b].rows_stored is True
    lines = row_store_path(project).read_bytes().split(b"\n")
    assert fragment in lines  # kept as its own line; ROW_C is not glued onto it
    assert load_row_id_set(project) == {row_id(ROW_A), row_id(ROW_C)}

    diff = runner.invoke(
        app, ["dataset-version-diff", str(project), "--version-id", version_a, "--other", version_b]
    )
    assert diff.exit_code == 0, diff.output
    assert "## Added (sample)" in diff.output  # the added row was read past the damaged line

    for version_id, expected in ((version_a, [ROW_A]), (version_b, [ROW_A, ROW_C])):
        output = tmp_path / f"{version_id}.jsonl"
        restored = runner.invoke(
            app,
            [
                "dataset-version-restore",
                str(project),
                "--version-id",
                version_id,
                "--output",
                str(output),
            ],
        )
        assert restored.exit_code == 0, restored.output
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert rows == expected


def test_examples_delete_captures_a_verified_undo_after_a_tear(tmp_path: Path) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A, ROW_C])
    _create(project, "A")
    _tear(project, ROW_B, "\u4e2d", bytes_in=2)

    result = runner.invoke(app, ["examples-delete", str(project), "--row", "1", "--json"])

    assert result.exit_code == 0, result.output
    undo = json.loads(result.stdout)["undo_version_id"]
    assert len(reconstruct_version_lines(project, undo)) == 2
    assert [json.loads(line) for line in (project / "examples.jsonl").read_text().splitlines()] == [
        ROW_C
    ]


def test_inplace_restore_captures_its_undo_after_a_tear(tmp_path: Path) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A])
    target = _create(project, "target")
    _write_examples(project, [ROW_A, ROW_C])
    _tear(project, ROW_EMOJI, "\U0001f600", bytes_in=2)

    result = runner.invoke(
        app, ["dataset-version-restore", str(project), "--version-id", target, "--in-place"]
    )

    assert result.exit_code == 0, result.output
    assert fingerprint_dataset(project / "examples.jsonl")[1] == 1
    undo = [record for record in list_version_records(project) if record.version_id != target]
    assert len(undo) == 1 and undo[0].rows_stored is True
    assert len(reconstruct_version_lines(project, undo[0].version_id)) == 2


def test_import_commit_after_a_tear_commits_rows_with_their_undo_version(tmp_path: Path) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A])
    _create(project, "A")
    _tear(project, ROW_B, "\u4e2d")
    staging = tmp_path / "staging.jsonl"
    staging.write_text(json.dumps(ROW_C, ensure_ascii=False) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["import-commit", str(project), "--from", str(staging), "--json"])

    assert result.exit_code == 0, result.output
    version_id = json.loads(result.stdout)["version_id"]
    assert version_id is not None
    assert len(list_version_records(project)) == 2
    restored = [json.loads(line) for line in reconstruct_version_lines(project, version_id)]
    assert restored == [ROW_A, ROW_C]


# --- the readers classify line by line ---------------------------------------------------------


def test_readers_skip_damaged_lines_but_keep_the_rows_around_them(tmp_path: Path) -> None:
    store = row_store_path(tmp_path)
    store.parent.mkdir(parents=True)
    glued = _torn_line(ROW_EMOJI, "\U0001f600", 2) + store_line(row_id(ROW_B), ROW_B).encode()
    store.write_bytes(
        b"\xef\xbb\xbf"  # a BOM (e.g. re-saved by a Windows editor): the first row still counts
        + store_line(row_id(ROW_A), ROW_A).encode("utf-8")
        + b"\n   \n"  # blank lines
        + b"[1, 2]\n"  # JSON, but not an object
        + b'{"row": 1}\n'  # an object without a row_id
        + b'{"row_id": 5, "row": 1}\n'  # a non-string row_id
        + glued  # an older writer glued ROW_B onto an undecodable fragment: ROW_B is lost
        + store_line(row_id(ROW_C), ROW_C).encode("utf-8").replace(b"\n", b"\r\n")
        + b'{"row": "torn'  # a torn, unterminated tail
    )
    wanted = {row_id(ROW_A), row_id(ROW_B), row_id(ROW_C)}

    assert load_row_id_set(tmp_path) == {row_id(ROW_A), row_id(ROW_C)}
    assert load_rows_by_id(tmp_path, wanted) == {row_id(ROW_A): ROW_A, row_id(ROW_C): ROW_C}
    assert load_rows_by_id(tmp_path, {row_id(ROW_A)}) == {row_id(ROW_A): ROW_A}  # stops early
    assert load_rows_by_id(tmp_path, set()) == {}


def test_an_unreadable_store_reads_as_empty(tmp_path: Path) -> None:
    row_store_path(tmp_path).mkdir(parents=True)  # opening it raises an OSError
    assert load_row_id_set(tmp_path) == set()
    assert load_rows_by_id(tmp_path, {row_id(ROW_A)}) == {}


# --- the store is opened for writing only when something must be written ----------------------


def _deny_store_writes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every write-mode open of the row store fail (portable: root ignores chmod)."""

    denied: list[str] = []
    real_open = Path.open

    def guarded_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self.name == "row_store.jsonl" and any(flag in mode for flag in "wax+"):
            denied.append(mode)
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    return denied


def test_an_intact_read_only_store_still_stores_an_unchanged_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    create_dataset_version(tmp_path, label="A")
    denied = _deny_store_writes(monkeypatch)

    again = create_dataset_version(tmp_path, label="A again")

    assert denied == []  # nothing needed writing, so nothing tried to
    assert again.rows_stored is True
    assert len(reconstruct_version_lines(tmp_path, again.version_id)) == 1


def test_a_torn_tail_on_an_unwritable_store_yields_a_fingerprint_only_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_examples(tmp_path, [ROW_A])
    create_dataset_version(tmp_path, label="A")
    _tear(tmp_path, ROW_B, "\u4e2d")
    before = row_store_path(tmp_path).read_bytes()
    denied = _deny_store_writes(monkeypatch)

    version = create_dataset_version(tmp_path, label="A torn")

    assert denied == ["ab"]  # the repair was attempted, and refused
    assert version.rows_stored is False
    assert version.content_fingerprint == fingerprint_dataset(tmp_path / "examples.jsonl")[0]
    assert row_store_path(tmp_path).read_bytes() == before


# --- GC keeps an undecodable line byte for byte -----------------------------------------------

ROW_ORPHAN = {"instruction": "orphan", "output": "pruned"}


def _line(row: dict[str, Any]) -> bytes:
    return store_line(row_id(row), row).encode("utf-8")


def test_gc_after_a_tear_prunes_orphans_and_keeps_the_fragment_byte_for_byte(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "proj")
    _write_examples(project, [ROW_A])
    version_a = _create(project, "A")
    fragment = _tear(project, ROW_B, "\u4e2d")
    _write_examples(project, [ROW_A, ROW_C])
    version_c = _create(project, "A+C")
    with row_store_path(project).open("ab") as handle:
        handle.write(_line(ROW_ORPHAN))

    gc = runner.invoke(app, ["dataset-version-gc", str(project), "--json"])

    assert gc.exit_code == 0, gc.output
    assert json.loads(gc.stdout)["pruned_rows"] == 1
    assert row_store_path(project).read_bytes() == (
        _line(ROW_A) + fragment + b"\n" + _line(ROW_C)
    )
    assert reconstruct_version_lines(project, version_a)
    assert len(reconstruct_version_lines(project, version_c)) == 2
    assert gc_row_store(project).pruned_rows == 0  # idempotent: the fragment is no refusal


def test_gc_keeps_undecodable_lines_anywhere_and_rewrites_them_unchanged(tmp_path: Path) -> None:
    _write_examples(tmp_path, [ROW_A, ROW_C])
    keep = create_dataset_version(tmp_path, label="keep")
    glued = _torn_line(ROW_EMOJI, "\U0001f600", 2) + _line(ROW_B)  # older damage, mid-store
    encoded_surrogate = b'{"row_id": "\xed\xa0\x80"}\n'  # not valid UTF-8 either
    no_row_id = b'{"row": 1}\n'  # valid JSON, but nothing GC can classify
    store = row_store_path(tmp_path)
    store.write_bytes(
        b"\xef\xbb\xbf"
        + _line(ROW_A)
        + glued
        + _line(ROW_C)
        + _line(ROW_ORPHAN)
        + encoded_surrogate
        + no_row_id
    )

    result = gc_row_store(tmp_path)

    assert (result.scanned_rows, result.kept_rows, result.pruned_rows) == (3, 2, 1)
    assert store.read_bytes() == (
        _line(ROW_A) + glued + _line(ROW_C) + encoded_surrogate + no_row_id
    )
    assert load_row_id_set(tmp_path) == {row_id(ROW_A), row_id(ROW_C)}
    assert len(reconstruct_version_lines(tmp_path, keep.version_id)) == 2
