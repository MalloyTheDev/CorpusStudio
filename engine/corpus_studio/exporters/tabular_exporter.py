"""CSV / TSV export for FLAT schemas (dependency-light, stdlib ``csv``).

Corpus Studio is JSONL-canonical — JSONL is the model-ready deliverable and the
only format that can represent a chat ``messages`` array or a nested object. CSV
export is an opt-in **convenience for flat schemas** (classification, raw text,
simple instruction, …): open the cleaned set in Excel/pandas or share it.

A schema is refused for CSV/TSV when a field is genuinely non-tabular — a chat
``messages`` field, an ``object`` field, or a ``list`` of objects — because those
can't become flat columns without lossy JSON-in-a-cell flattening. Scalar fields
export as text; a ``list`` of scalars (e.g. ``tags``) is written as a
``"; "``-joined cell (a normal CSV convention). This mirrors the CSV *import*
honesty boundary: CSV is flat, so structure that can't be flat is refused, not
silently mangled.
"""

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from corpus_studio.schemas.base import DatasetSchema

# Field types that cannot become a flat CSV column.
_NON_TABULAR_TYPES = frozenset({"messages", "object"})

_LIST_JOIN = "; "


def schema_is_csv_exportable(schema: DatasetSchema) -> tuple[bool, list[str]]:
    """Return ``(ok, blocking_fields)``. A field blocks CSV/TSV export when it is a
    chat ``messages`` field, an ``object`` field, or a ``list`` of ``object`` — all
    genuinely nested. Scalar fields and lists of scalars are fine."""
    blocking = [
        field.name
        for field in schema.fields
        if field.type in _NON_TABULAR_TYPES
        or (field.type == "list" and field.item_type == "object")
    ]
    return (not blocking, blocking)


def _cell(value: Any) -> str:
    """Render one row value as a CSV cell. Scalars → text; a list of scalars →
    ``"; "``-joined; a stray dict/list-of-dict → JSON (defensive — a flat schema
    should never reach here, but never silently drop data)."""
    if value is None:
        return ""
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return _LIST_JOIN.join("" if item is None else str(item) for item in value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_tabular(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    schema: DatasetSchema,
    delimiter: str,
) -> tuple[int, list[str]]:
    """Write rows to a CSV/TSV file and return ``(row_count, columns)``.

    Columns are the schema's declared fields in order, then any extra keys found in
    the rows (sorted) so nothing a row carries is dropped. Raises ``ValueError`` if
    the schema is not CSV-exportable (caller should have checked first)."""
    ok, blocking = schema_is_csv_exportable(schema)
    if not ok:
        raise ValueError(
            f"Schema '{schema.id}' has non-tabular field(s) {blocking}; export as JSONL instead."
        )

    materialised = list(rows)
    declared = [field.name for field in schema.fields]
    extra = sorted({key for row in materialised for key in row} - set(declared))
    columns = declared + extra

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        for row in materialised:
            writer.writerow([_cell(row.get(column)) for column in columns])

    return len(materialised), columns
