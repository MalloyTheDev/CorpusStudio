"""CSV / TSV export for flat schemas.

JSONL stays the canonical, model-ready export; CSV/TSV is an opt-in convenience
for flat schemas. These tests pin the flatness gate (chat/objects refused), the
cell rendering (scalar lists joined, no data dropped), the delimiter, and the
CLI `export --format csv/tsv` path reusing the whole validate/gate/clean pipeline.
"""

import csv
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.exporters.tabular_exporter import (
    schema_is_csv_exportable,
    write_tabular,
)
from corpus_studio.schemas.base import DatasetSchema, SchemaField
from corpus_studio.schemas.registry import load_builtin_schema

runner = CliRunner()


def test_flat_schemas_are_exportable():
    for schema_id in ("classification", "raw_text", "instruction", "evaluation"):
        ok, blocking = schema_is_csv_exportable(load_builtin_schema(schema_id))
        assert ok, f"{schema_id} should be CSV-exportable, blocked on {blocking}"


def test_chat_is_refused_on_its_messages_field():
    ok, blocking = schema_is_csv_exportable(load_builtin_schema("chat"))
    assert not ok
    assert blocking == ["messages"]


def test_object_and_list_of_object_fields_are_refused():
    schema = DatasetSchema(
        id="custom",
        name="Custom",
        version="0.1.0",
        fields=[
            SchemaField(name="text", type="text"),
            SchemaField(name="meta", type="object"),
            SchemaField(name="events", type="list", item_type="object"),
        ],
    )
    ok, blocking = schema_is_csv_exportable(schema)
    assert not ok
    assert blocking == ["meta", "events"]


def test_write_tabular_joins_scalar_lists_and_keeps_column_order(tmp_path: Path):
    schema = load_builtin_schema("classification")
    rows = [
        {"text": "Great value.", "label": "positive", "tags": ["sentiment", "review"]},
        {"text": "Broke fast.", "label": "negative", "tags": []},
    ]
    out = tmp_path / "out.csv"

    count, columns = write_tabular(rows, out, schema, delimiter=",")

    assert count == 2
    assert columns == ["text", "label", "tags"]
    parsed = list(csv.reader(out.read_text(encoding="utf-8").splitlines()))
    assert parsed[0] == ["text", "label", "tags"]
    assert parsed[1] == ["Great value.", "positive", "sentiment; review"]
    assert parsed[2] == ["Broke fast.", "negative", ""]  # empty list -> empty cell


def test_write_tabular_appends_extra_row_keys_so_nothing_is_dropped(tmp_path: Path):
    # classification declares text/label/tags; an extra (non-schema) key on a row must
    # still be exported as a trailing column, not silently dropped.
    schema = load_builtin_schema("classification")
    rows = [{"text": "hello", "label": "positive", "provenance": "notes.md"}]
    out = tmp_path / "out.csv"

    _, columns = write_tabular(rows, out, schema, delimiter=",")

    assert columns[:3] == ["text", "label", "tags"]  # declared fields, in order
    assert "provenance" in columns  # extra key preserved as a trailing column


def test_cli_export_csv_writes_flat_file(tmp_path: Path):
    src = tmp_path / "cls.jsonl"
    src.write_text('{"text":"Hi","label":"positive"}\n', encoding="utf-8")
    out = tmp_path / "out.tsv"

    result = runner.invoke(
        app, ["export", str(src), str(out), "classification", "--format", "tsv"]
    )

    assert result.exit_code == 0, result.output
    # classification declares text/label/tags, so all three are columns even when a
    # row omits tags (the cell is empty).
    assert out.read_text(encoding="utf-8").splitlines()[0] == "text\tlabel\ttags"
    assert "Hi\tpositive\t" in out.read_text(encoding="utf-8")


def test_cli_export_csv_refuses_chat(tmp_path: Path):
    src = tmp_path / "chat.jsonl"
    src.write_text(
        '{"messages":[{"role":"user","content":"hi"}]}\n', encoding="utf-8"
    )
    out = tmp_path / "out.csv"

    result = runner.invoke(app, ["export", str(src), str(out), "chat", "--format", "csv"])

    assert result.exit_code == 2
    assert not out.exists()  # nothing partial written
