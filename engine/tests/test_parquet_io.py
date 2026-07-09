"""Parquet import/export: optional pyarrow, graceful degradation, typed round-trip."""

import datetime
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio import parquet_support
from corpus_studio.cli import app
from corpus_studio.parquet_support import (
    PARQUET_INSTALL_HINT,
    ParquetSupportError,
    json_default,
    load_pyarrow,
    parquet_available,
)

runner = CliRunner()


# --- Support module: always runnable (no pyarrow needed) ---------------------


def test_json_default_decodes_bytes_and_stringifies_scalars():
    assert json_default(b"hi") == "hi"
    assert json_default(bytearray(b"hi")) == "hi"
    # Undecodable bytes are replaced, never crash.
    assert isinstance(json_default(b"\xff\xfe"), str)
    # datetime / Decimal-like scalars fall back to their string form.
    assert json_default(datetime.date(2026, 7, 9)) == "2026-07-09"


def _force_pyarrow_absent(monkeypatch):
    def _raise():
        raise ImportError("no pyarrow")

    monkeypatch.setattr(parquet_support, "_import_pyarrow", _raise)


def test_load_pyarrow_raises_support_error_with_hint_when_absent(monkeypatch):
    _force_pyarrow_absent(monkeypatch)
    with pytest.raises(ParquetSupportError) as excinfo:
        load_pyarrow()
    assert "pip install" in str(excinfo.value)
    assert "parquet" in str(excinfo.value)


def test_parquet_available_false_when_absent(monkeypatch):
    _force_pyarrow_absent(monkeypatch)
    assert parquet_available() is False


def test_export_parquet_without_pyarrow_fails_fast_with_hint(monkeypatch, tmp_path: Path):
    _force_pyarrow_absent(monkeypatch)
    input_path = tmp_path / "in.jsonl"
    input_path.write_text('{"instruction":"Explain x in detail.","output":"A full answer."}\n', encoding="utf-8")
    output_path = tmp_path / "out.parquet"

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction", "--format", "parquet"])

    assert result.exit_code == 1
    assert PARQUET_INSTALL_HINT in result.output
    assert not output_path.exists()  # no partial output written


def test_import_convert_parquet_without_pyarrow_reports_hint(monkeypatch, tmp_path: Path):
    _force_pyarrow_absent(monkeypatch)
    result = runner.invoke(app, ["import-convert", str(tmp_path / "x.parquet"), str(tmp_path / "out.jsonl")])
    assert result.exit_code == 1
    assert "pip install" in result.output


def test_unknown_format_is_rejected(tmp_path: Path):
    input_path = tmp_path / "in.jsonl"
    input_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    result = runner.invoke(app, ["export", str(input_path), str(tmp_path / "o"), "instruction", "--format", "xml"])
    assert result.exit_code == 1
    assert "parquet" in result.output  # the message lists the accepted formats


# --- Round-trip + CLI: require pyarrow (installed in dev/CI via .[dev]) -------

pa_test = pytest.importorskip("pyarrow")
pq_test = pytest.importorskip("pyarrow.parquet")
from corpus_studio.exporters.parquet_exporter import write_parquet  # noqa: E402
from corpus_studio.importers.parquet_importer import (  # noqa: E402
    convert_parquet_to_jsonl,
    read_parquet,
)


def _write_parquet_table(path: Path, rows: list[dict]) -> None:
    table = pa_test.Table.from_pylist(rows)
    pq_test.write_table(table, str(path))


def test_read_parquet_preserves_types(tmp_path: Path):
    path = tmp_path / "d.parquet"
    _write_parquet_table(path, [{"text": "hi", "score": 3, "ok": True}, {"text": "yo", "score": 5, "ok": False}])
    rows = list(read_parquet(path))
    assert rows == [{"text": "hi", "score": 3, "ok": True}, {"text": "yo", "score": 5, "ok": False}]
    # Integers stay integers (unlike CSV, where every cell is a string).
    assert isinstance(rows[0]["score"], int)


def test_convert_parquet_to_jsonl_row_count_and_columns(tmp_path: Path):
    path = tmp_path / "d.parquet"
    _write_parquet_table(path, [{"text": "a", "label": "x"}, {"text": "b", "label": "y"}])
    out = tmp_path / "staging.jsonl"
    conversion = convert_parquet_to_jsonl(path, out)
    assert conversion.rows_converted == 2
    assert conversion.columns == ["text", "label"]
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [{"text": "a", "label": "x"}, {"text": "b", "label": "y"}]


def test_convert_empty_parquet_reports_columns_from_schema(tmp_path: Path):
    path = tmp_path / "empty.parquet"
    # A table with columns but zero rows.
    table = pa_test.table({"text": pa_test.array([], type=pa_test.string()), "n": pa_test.array([], type=pa_test.int64())})
    pq_test.write_table(table, str(path))
    out = tmp_path / "staging.jsonl"
    conversion = convert_parquet_to_jsonl(path, out)
    assert conversion.rows_converted == 0
    assert conversion.columns == ["text", "n"]


def test_write_parquet_round_trips_nested_chat_rows(tmp_path: Path):
    # Parquet represents nested types natively — chat `messages` becomes list<struct>.
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "tags": ["greet"]},
    ]
    path = tmp_path / "chat.parquet"
    count, columns = write_parquet(rows, path)
    assert count == 1
    assert set(columns) == {"messages", "tags"}
    assert list(read_parquet(path)) == rows


def test_write_parquet_empty_rows(tmp_path: Path):
    count, columns = write_parquet([], tmp_path / "empty.parquet")
    assert count == 0
    assert columns == []


def test_cli_import_convert_parquet(tmp_path: Path):
    path = tmp_path / "d.parquet"
    _write_parquet_table(path, [{"instruction": "Explain recursion fully.", "output": "It calls itself."}])
    out = tmp_path / "staging.jsonl"
    result = runner.invoke(app, ["import-convert", str(path), str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows_converted"] == 1
    assert payload["columns"] == ["instruction", "output"]


def test_cli_export_parquet_flat_schema_round_trips(tmp_path: Path):
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(
        '{"instruction":"Explain recursion in detail.","output":"A function that calls itself."}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "out.parquet"
    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction", "--format", "parquet"])
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    rows = list(read_parquet(output_path))
    assert rows[0]["instruction"].startswith("Explain recursion")


def test_cli_export_parquet_supports_nested_chat_schema(tmp_path: Path):
    # The key advantage over CSV/TSV: a chat schema exports to Parquet (CSV refuses it).
    input_path = tmp_path / "chat.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is recursion?"},
                    {"role": "assistant", "content": "A function calling itself."},
                ],
                "tags": ["programming"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "chat.parquet"
    result = runner.invoke(app, ["export", str(input_path), str(output_path), "chat", "--format", "parquet"])
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    # And CSV is still refused for the same nested schema (the honest boundary).
    csv_result = runner.invoke(app, ["export", str(input_path), str(tmp_path / "c.csv"), "chat", "--format", "csv"])
    assert csv_result.exit_code == 2
