"""CSV/TSV import → JSONL staging.

Corpus Studio stays JSONL-canonical; tabular files are supported at the import
boundary by converting to a staging JSONL that flows through the same
import-preview → quarantine → commit path as any JSONL. These tests pin the
conversion (delimiter, header→keys, string values, BOM, errors) and the
round-trip into the JSONL reader.
"""

import json
from pathlib import Path

import pytest

from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.importers.tabular_importer import (
    convert_tabular_to_jsonl,
    read_tabular,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_header_becomes_keys_values_are_strings(tmp_path: Path):
    csv_path = _write(tmp_path / "d.csv", "text,label\nHello,positive\nBye,negative\n")

    rows = list(read_tabular(csv_path))

    assert rows == [
        {"text": "Hello", "label": "positive"},
        {"text": "Bye", "label": "negative"},
    ]


def test_quoted_field_with_delimiter_is_preserved(tmp_path: Path):
    csv_path = _write(tmp_path / "d.csv", 'text,note\n"a, b, c",ok\n')

    rows = list(read_tabular(csv_path))

    assert rows == [{"text": "a, b, c", "note": "ok"}]


def test_tsv_uses_tab_delimiter(tmp_path: Path):
    tsv_path = _write(tmp_path / "d.tsv", "text\tlabel\na,b\tpositive\n")

    rows = list(read_tabular(tsv_path))

    # The comma is data, not a delimiter, in a TSV.
    assert rows == [{"text": "a,b", "label": "positive"}]


def test_bom_is_tolerated(tmp_path: Path):
    # Excel/Windows exports often prepend a UTF-8 BOM; it must not corrupt the
    # first header name (matching the JSONL reader's utf-8-sig contract).
    csv_path = tmp_path / "d.csv"
    csv_path.write_bytes(b"\xef\xbb\xbftext,label\nHi,positive\n")

    rows = list(read_tabular(csv_path))

    assert rows == [{"text": "Hi", "label": "positive"}]


def test_short_row_pads_missing_cells_with_empty_string(tmp_path: Path):
    csv_path = _write(tmp_path / "d.csv", "a,b,c\n1,2\n")

    rows = list(read_tabular(csv_path))

    assert rows == [{"a": "1", "b": "2", "c": ""}]


def test_empty_file_raises_valueerror(tmp_path: Path):
    empty = _write(tmp_path / "empty.csv", "")

    with pytest.raises(ValueError, match="empty"):
        list(read_tabular(empty))


def test_convert_writes_valid_jsonl_and_reports_columns(tmp_path: Path):
    csv_path = _write(tmp_path / "d.csv", "text,label\nHi,positive\nBye,negative\n")
    out = tmp_path / "staging.jsonl"

    result = convert_tabular_to_jsonl(csv_path, out)

    assert result.rows_converted == 2
    assert result.columns == ["text", "label"]
    assert result.output_path == str(out)
    # The staging file is real JSONL the strict reader accepts, so it drops
    # straight into the existing import-preview/commit path.
    assert list(read_jsonl(out)) == [
        {"text": "Hi", "label": "positive"},
        {"text": "Bye", "label": "negative"},
    ]


def test_convert_preserves_unicode(tmp_path: Path):
    csv_path = _write(tmp_path / "d.csv", "text,label\n日本語のテキスト,positive\n")
    out = tmp_path / "staging.jsonl"

    convert_tabular_to_jsonl(csv_path, out)

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row == {"text": "日本語のテキスト", "label": "positive"}
