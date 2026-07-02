import hashlib
import json
from pathlib import Path

from corpus_studio.exporters.cleaning import exact_row_signature
from corpus_studio.versions.row_store import (
    append_rows,
    load_row_id_set,
    load_rows_by_id,
    row_id,
    row_store_path,
    store_line,
)

ROW_A = {"instruction": "Explain recursion.", "output": "A function calls itself."}
ROW_B = {"instruction": "Explain binary search.", "output": "Halve a sorted range."}


def test_row_id_matches_sha256_of_exact_signature():
    expected = hashlib.sha256(exact_row_signature(ROW_A).encode("utf-8")).hexdigest()
    assert row_id(ROW_A) == expected
    assert row_id(ROW_A) == row_id(ROW_A)  # deterministic
    assert row_id(ROW_A) != row_id(ROW_B)


def test_row_id_ignores_key_order():
    # Canonical (sorted-key) identity: key order does not change the id.
    assert row_id({"a": 1, "b": 2}) == row_id({"b": 2, "a": 1})


def test_append_dedupes_within_and_across_calls(tmp_path: Path):
    existing: set[str] = set()
    n1 = append_rows(tmp_path, [(row_id(ROW_A), ROW_A), (row_id(ROW_A), ROW_A)], existing)
    assert n1 == 1  # same row twice in one call -> stored once
    n2 = append_rows(tmp_path, [(row_id(ROW_A), ROW_A), (row_id(ROW_B), ROW_B)], existing)
    assert n2 == 1  # ROW_A already present -> only ROW_B is new
    lines = row_store_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_load_row_id_set_tolerates_torn_and_blank_lines(tmp_path: Path):
    path = row_store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        store_line(row_id(ROW_A), ROW_A)
        + "\n"  # blank line
        + "{ this is a torn line\n"
        + store_line(row_id(ROW_B), ROW_B),
        encoding="utf-8",
    )
    ids = load_row_id_set(tmp_path)
    assert ids == {row_id(ROW_A), row_id(ROW_B)}


def test_load_row_id_set_missing_file_is_empty(tmp_path: Path):
    assert load_row_id_set(tmp_path) == set()


def test_load_rows_by_id_returns_found_only(tmp_path: Path):
    existing: set[str] = set()
    append_rows(tmp_path, [(row_id(ROW_A), ROW_A), (row_id(ROW_B), ROW_B)], existing)
    got = load_rows_by_id(tmp_path, {row_id(ROW_A), "deadbeef"})
    assert got == {row_id(ROW_A): ROW_A}  # missing id simply omitted


def test_store_line_is_canonical_and_parseable():
    line = store_line(row_id(ROW_A), {"b": 2, "a": 1})
    entry = json.loads(line)
    assert entry["row_id"] == row_id(ROW_A)
    # sort_keys => canonical key order in the stored body
    assert list(entry["row"].keys()) == ["a", "b"]
