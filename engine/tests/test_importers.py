from pathlib import Path

import pytest

from corpus_studio.exporters.jsonl_exporter import export_jsonl, write_jsonl
from corpus_studio.importers.jsonl_importer import iter_jsonl, read_jsonl


def test_read_jsonl_skips_blank_lines_and_parses(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_rejects_non_object_line(tmp_path: Path):
    # A dataset row must be a JSON object; a list/scalar/null line is malformed
    # input and must raise a ValueError (not yield a non-dict that crashes later).
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(ValueError):
        list(read_jsonl(path))


def test_read_jsonl_tolerates_utf8_bom(tmp_path: Path):
    # Excel/Windows tools often prepend a UTF-8 BOM; import must not choke on it.
    path = tmp_path / "bom.jsonl"
    path.write_text('{"text": "hello"}\n', encoding="utf-8-sig")
    assert list(read_jsonl(path)) == [{"text": "hello"}]


def test_write_jsonl_round_trips_with_sorted_keys(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    write_jsonl([{"b": 2, "a": 1}], path)
    assert path.read_text(encoding="utf-8") == '{"a": 1, "b": 2}\n'
    assert list(read_jsonl(path)) == [{"a": 1, "b": 2}]


def test_export_jsonl_copies_file_into_new_dir(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    src.write_text('{"a": 1}\n', encoding="utf-8")
    dst = tmp_path / "nested" / "dst.jsonl"
    export_jsonl(src, dst)
    assert dst.read_text(encoding="utf-8") == '{"a": 1}\n'


def _write_large_dataset(path: Path, rows: int) -> None:
    # Stream the file out so the test itself never holds 100k rows in memory.
    with path.open("w", encoding="utf-8") as f:
        for i in range(rows):
            f.write('{"instruction": "q%d", "output": "a%d"}\n' % (i, i))


def test_read_jsonl_streams_a_100k_row_dataset(tmp_path: Path):
    # The reader is a generator: a large dataset must flow row-by-row without
    # the reader ever materializing the whole file. We assert count + edges
    # while only ever holding one row at a time.
    path = tmp_path / "big.jsonl"
    rows = 100_000
    _write_large_dataset(path, rows)

    seen = 0
    first = last = None
    for row in read_jsonl(path):
        if seen == 0:
            first = row
        last = row
        seen += 1

    assert seen == rows
    assert first == {"instruction": "q0", "output": "a0"}
    assert last == {"instruction": "q99999", "output": "a99999"}


def test_strict_read_raises_on_a_malformed_line_deep_in_a_large_file(tmp_path: Path):
    # Fail-fast contract holds at scale: one bad line among 100k must raise,
    # not silently drop the row or corrupt the stream.
    path = tmp_path / "big_bad.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(50_000):
            f.write('{"instruction": "q%d", "output": "a%d"}\n' % (i, i))
        f.write("{ this is not json\n")
        for i in range(50_000, 100_000):
            f.write('{"instruction": "q%d", "output": "a%d"}\n' % (i, i))

    with pytest.raises(ValueError, match="line 50001"):
        for _ in read_jsonl(path):
            pass


def test_lenient_iter_collects_bad_lines_and_accepts_the_rest_at_scale(tmp_path: Path):
    # The tolerant primitive records every malformed/non-object line in one pass
    # over a large file and accepts all the valid ones — the shared basis for
    # import preview and file validation.
    path = tmp_path / "big_mixed.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(100_000):
            if i in (10, 250, 99_999):
                f.write("{ broken\n")
            elif i in (42, 5000):
                f.write("[1, 2, 3]\n")  # valid JSON, but not an object
            else:
                f.write('{"instruction": "q%d"}\n' % i)

    good = bad = nonobject = 0
    for parsed in iter_jsonl(path):
        if parsed.error is not None:
            bad += 1
        elif not isinstance(parsed.value, dict):
            nonobject += 1
        else:
            good += 1

    assert bad == 3  # malformed JSON lines were reported, never raised
    assert nonobject == 2  # non-object lines decode but aren't objects
    assert good == 100_000 - 5
