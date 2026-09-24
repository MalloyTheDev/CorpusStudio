"""Parquet export keeps every field, including ones absent from the first row.

Regression for ``pa.Table.from_pylist`` taking its column set from row 1 only: an optional field
(an instruction row's ``input``) or a ``meta`` key that first appeared in a later row was dropped
from every row of the deliverable, with exit 0 and a ``columns`` report that simply left it out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.exporters.parquet_exporter import write_parquet

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

runner = CliRunner()


def test_write_parquet_unions_columns_in_first_seen_order(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"instruction": "Explain recursion.", "output": "A function calling itself."},
        {"instruction": "Summarize.", "input": "LONG CONTEXT", "output": "Short."},
        {"output": "Last.", "instruction": "Tag it.", "meta": {"source": "manual"}},
    ]
    path = tmp_path / "out.parquet"

    count, columns = write_parquet(rows, path)

    table = pq.read_table(str(path))
    assert count == 3
    assert columns == ["instruction", "output", "input", "meta"]
    assert table.column_names == columns
    assert table.to_pylist() == [
        {**rows[0], "input": None, "meta": None},
        {**rows[1], "meta": None},
        {"instruction": "Tag it.", "output": "Last.", "input": None, "meta": {"source": "manual"}},
    ]


def test_write_parquet_homogeneous_rows_match_from_pylist(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"messages": [{"role": "user", "content": "hi"}], "tags": ["greet"], "score": 1},
        {"messages": [{"role": "assistant", "content": "yo"}], "tags": [], "score": None},
    ]
    path = tmp_path / "same.parquet"

    _, columns = write_parquet(rows, path)

    expected = pa.Table.from_pylist(rows)
    written = pq.read_table(str(path))
    assert columns == expected.column_names
    assert written.schema.equals(expected.schema)
    assert written.to_pylist() == expected.to_pylist()


def test_write_parquet_rejects_a_row_that_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    rows: list[Any] = [{"text": "ok"}, ["not", "an", "object"]]

    with pytest.raises(ValueError, match="row 2 is not a JSON object"):
        write_parquet(rows, path)

    assert not path.exists()


@pytest.mark.parametrize("extra_args", [[], ["--dedupe"]], ids=["verbatim", "cleaned"])
def test_cli_export_parquet_keeps_fields_absent_from_the_first_row(
    tmp_path: Path, extra_args: list[str]
) -> None:
    source = tmp_path / "in.jsonl"
    rows = [
        {"instruction": "Explain recursion in detail.", "output": "A function calling itself."},
        {"instruction": "Summarize the passage.", "input": "LONG CONTEXT", "output": "Short."},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "out.parquet"

    result = runner.invoke(
        app,
        ["export", str(source), str(output), "instruction", "--format", "parquet", *extra_args],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    table = pq.read_table(str(output))
    assert payload["columns"] == ["instruction", "output", "input"]
    assert table.column_names == payload["columns"]
    assert [row["input"] for row in table.to_pylist()] == [None, "LONG CONTEXT"]
