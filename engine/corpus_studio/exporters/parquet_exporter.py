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

    The Arrow schema is inferred from the (schema-validated) rows via
    ``Table.from_pylist``, which unions keys across rows and represents nested
    values faithfully. Raises ``ParquetSupportError`` when pyarrow is missing, and
    ``ValueError`` when the rows can't form a consistent columnar table (e.g. a
    column with genuinely mixed types — which schema-valid export rows won't have)."""
    pa, pq = load_pyarrow()

    materialised = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table = pa.Table.from_pylist(materialised)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as exc:
        raise ValueError(
            f"Could not build a Parquet table from the rows: {exc}. Export as JSONL instead."
        ) from exc

    pq.write_table(table, str(output_path))
    return len(materialised), list(table.column_names)
