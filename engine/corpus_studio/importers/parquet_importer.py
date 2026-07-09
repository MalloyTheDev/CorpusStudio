"""Parquet import → JSONL staging (optional ``pyarrow``, streamed by batch).

Mirrors the CSV/TSV importer: a Parquet source is supported at the **import
boundary only** by converting it to a staging JSONL, which then flows through the
exact same import-preview → quarantine → commit pipeline as any JSONL / CSV / HF
import. One validation/commit path, not a second Parquet-specific one.

Unlike CSV (where every cell is a string), Parquet is typed and columnar — an
integer column stays an integer, a ``list``/``struct`` column stays nested — so
faithful values reach the staging JSONL and validate against the schema directly.
Reading is streamed per record-batch so a large Parquet file does not have to be
materialised in memory. Requires the optional ``[parquet]`` extra; without it,
callers get a clear ``ParquetSupportError`` (see ``corpus_studio.parquet_support``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_studio.parquet_support import json_default, load_pyarrow


def read_parquet(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a Parquet file as one ``dict`` per row (typed values preserved).

    Streamed per record-batch so memory stays bounded. Raises ``ParquetSupportError``
    when pyarrow is not installed, and lets pyarrow's own errors surface for an
    unreadable/corrupt file."""
    _, pq = load_pyarrow()
    parquet_file = pq.ParquetFile(str(path))
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            yield row


@dataclass(frozen=True)
class ParquetConversion:
    """Result of converting a Parquet file to a staging JSONL."""

    output_path: str
    rows_converted: int
    columns: list[str]


def convert_parquet_to_jsonl(input_path: Path, output_path: Path) -> ParquetConversion:
    """Convert a Parquet file to a JSONL staging file (one JSON object per row).

    Columns come from the Parquet schema (so an empty file still reports its
    columns). Values are written verbatim; the few non-JSON scalars a typed column
    can hold (bytes / datetime / Decimal) are handled by ``json_default``. This only
    reshapes Parquet → JSONL — schema validation is the import-preview's job, so a
    value that violates the target schema quarantines exactly like a JSONL row."""
    _, pq = load_pyarrow()
    parquet_file = pq.ParquetFile(str(input_path))
    columns = list(parquet_file.schema_arrow.names)

    rows_converted = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                out.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                rows_converted += 1

    return ParquetConversion(
        output_path=str(output_path),
        rows_converted=rows_converted,
        columns=columns,
    )
