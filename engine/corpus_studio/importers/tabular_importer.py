"""CSV / TSV import → JSONL staging (dependency-light, stdlib ``csv``).

Corpus Studio is JSONL-canonical: it authors, validates, gates, and exports
JSONL. Tabular sources (CSV/TSV) are supported at the **import boundary only** by
converting them to a staging JSONL, which then flows through the exact same
import-preview → quarantine → commit pipeline as any JSONL (and HF) import. That
keeps one validation/commit path instead of a second tabular one.

Honesty boundary: a CSV cell has no type — every value is imported as a
**string**. So a row whose schema field expects an integer/float/list/object
will fail the normal import-preview validation and land in quarantine for repair,
exactly like a malformed JSONL row. This converter never coerces types or drops
columns silently; the header row defines the keys and every data cell is carried
across verbatim as text.
"""

import csv
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# csv can emit very large fields; keep the stdlib default field-size limit (no
# unbounded growth) — a pathological cell is a data problem, not ours to hide.

_TAB_SUFFIXES = frozenset({".tsv", ".tab"})

# ``csv.DictReader`` collects cells beyond the header count under this key; naming it
# explicitly lets us detect (and refuse) a ragged row instead of silently dropping them.
_OVERFLOW_KEY = "__corpus_studio_overflow__"


def _delimiter_for(path: Path) -> str:
    """Pick the delimiter from the file extension (deterministic + honest — no
    sniffing that could silently mis-split). ``.tsv``/``.tab`` → tab, else comma."""
    return "\t" if path.suffix.lower() in _TAB_SUFFIXES else ","


def read_tabular(path: Path) -> Iterator[dict[str, str]]:
    """Stream a CSV/TSV file as one ``dict[str, str]`` per data row.

    The first row is the header and defines the keys. Values are always strings
    (empty cells -> ``""``). Reads as ``utf-8-sig`` so a BOM from Excel/Windows
    exports is tolerated (matching the JSONL reader). Raises ``ValueError`` for a
    structural problem the converter must NOT hide: an empty/blank header,
    **duplicate column names** (a dict would keep only the last, losing a column),
    or a **ragged row** with more cells than header columns (the extras would
    otherwise be silently dropped). A short row is fine - its missing trailing
    cells pad to ``""``. Also re-raises the csv module's own ``csv.Error`` (e.g. a
    cell over its field-size limit) as a ``ValueError`` naming the row, so it
    never escapes as a raw traceback. Callers' ``(OSError, ValueError)`` handlers
    surface it.
    """
    delimiter = _delimiter_for(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter, restval="", restkey=_OVERFLOW_KEY)
        try:
            fieldnames = reader.fieldnames
        except csv.Error as exc:
            # Reading ``fieldnames`` parses the header line - a header cell over
            # csv's field-size limit raises here, before any data row.
            raise ValueError(f"Could not read the header row: {exc}") from exc
        if fieldnames is None:
            raise ValueError("The file is empty - no header row was found.")
        header = [name for name in fieldnames if name is not None]
        if not any(name.strip() for name in header):
            raise ValueError("The header row is empty - no column names were found.")

        # Duplicate header names collapse to one dict key (last value wins), silently
        # dropping a column. Refuse rather than mangle the source.
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in header:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            joined = ", ".join(repr(name) for name in duplicates)
            raise ValueError(
                f"Duplicate column name(s) in the header: {joined}. "
                "Rename them so every column is distinct."
            )

        row_number = 0
        while True:
            row_number += 1
            try:
                raw_row = next(reader)
            except StopIteration:
                return
            except csv.Error as exc:
                # E.g. a cell over csv's field-size limit - the reader raises its own
                # _csv.Error mid-stream, which isn't a ValueError and would otherwise
                # escape callers' (OSError, ValueError) handlers as a raw traceback.
                raise ValueError(f"Could not read data row {row_number}: {exc}") from exc
            overflow = raw_row.get(_OVERFLOW_KEY)
            if overflow:
                raise ValueError(
                    f"Data row {row_number} has {len(overflow)} more cell(s) than the "
                    f"{len(header)} header column(s) (a ragged row). Fix the source, or "
                    "quote a cell that contains the delimiter."
                )
            row: dict[str, str] = {}
            for key in header:
                value = raw_row.get(key)
                # A short data row leaves later columns as restval (""); a value can
                # also be None if the header had a blank name. Normalise to text.
                row[key] = "" if value is None else str(value)
            yield row


@dataclass(frozen=True)
class TabularConversion:
    """Result of converting a tabular file to a staging JSONL."""

    output_path: str
    rows_converted: int
    columns: list[str]


def convert_tabular_to_jsonl(input_path: Path, output_path: Path) -> TabularConversion:
    """Convert a CSV/TSV file to a JSONL staging file (one JSON object per row).

    Returns the row count and the detected columns. The staging file is what the
    normal import-preview/quarantine/commit flow then consumes — this function
    only reshapes tabular → JSONL, it never validates against a schema (that is
    the import-preview's job, so bad rows quarantine the same way JSONL does).
    Written atomically (unique temp file beside ``output_path``, then
    ``os.replace``) so a mid-stream failure (a ragged row, an oversized field)
    leaves ``output_path`` exactly as it was before the call, never a partial
    prefix of the source.
    """
    rows_converted = 0
    columns: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(output_path.parent), prefix=output_path.name + ".", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for row in read_tabular(input_path):
                if not columns:
                    columns = list(row.keys())
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_converted += 1
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return TabularConversion(
        output_path=str(output_path),
        rows_converted=rows_converted,
        columns=columns,
    )
