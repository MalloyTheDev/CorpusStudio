from pathlib import Path

from corpus_studio.exporters.jsonl_exporter import export_jsonl, write_jsonl
from corpus_studio.importers.jsonl_importer import read_jsonl


def test_read_jsonl_skips_blank_lines_and_parses(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


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
