"""Parquet export (optional ``pyarrow``).

Unlike CSV/TSV export — which is refused for chat/nested schemas because a
``messages`` array or object can't become a flat column — Parquet is columnar and
represents nested types natively (a chat ``messages`` field becomes a
``list<struct>`` column, an object field a ``struct``). So Parquet export supports
**every** schema, including chat and nested ones, with no lossy flattening. It is a
model-adjacent, analytics-friendly deliverable (open it in pandas/DuckDB/Spark),
while JSONL stays the canonical, human-diffable, trainer-ready format.

Requires the optional ``[parquet]`` extra; without it the caller gets a clear
``ParquetSupportError`` before any work (see ``corpus_studio.parquet_support``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from corpus_studio.parquet_support import load_pyarrow


def write_parquet(rows: Iterable[dict[str, Any]], output_path: Path) -> tuple[int, list[str]]:
    """Write rows to a Parquet file and return ``(row_count, columns)``.

    The columns are the union of the keys across ALL rows, in first-seen order, and
    a row that lacks a column holds null there, so no field is dropped (the export
    rule in docs/IMPORT_EXPORT.md). The returned ``columns`` are exactly the file's.
    Each column's Arrow type is inferred from its values, which represents nested
    values faithfully. Raises ``ParquetSupportError`` when pyarrow is missing, and
    ``ValueError`` when the rows can't form a consistent columnar table (a row that
    is not an object, or a column with genuinely mixed types - which schema-valid
    export rows won't have)."""
    pa, pq = load_pyarrow()

    materialised = list(rows)
    columns: dict[str, None] = {}
    for row_number, row in enumerate(materialised, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Could not build a Parquet table from the rows: row {row_number} is not "
                "a JSON object. Export as JSONL instead."
            )
        columns.update(dict.fromkeys(row))
    names = list(columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Table.from_pylist takes its column set from the FIRST row only, so a key that
        # first appears in a later row would vanish from every row. Build the columns
        # the same way from_pylist does (one value list per name, None where a row
        # lacks the key), but over the union of names, so homogeneous rows produce the
        # identical table while heterogeneous rows keep every field.
        table = pa.Table.from_arrays(
            [[row[name] if name in row else None for row in materialised] for name in names],
            names=names,
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as exc:
        raise ValueError(
            f"Could not build a Parquet table from the rows: {exc}. Export as JSONL instead."
        ) from exc

    pq.write_table(table, str(output_path))
    return len(materialised), list(table.column_names)
