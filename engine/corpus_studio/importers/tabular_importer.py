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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# csv can emit very large fields; keep the stdlib default field-size limit (no
# unbounded growth) — a pathological cell is a data problem, not ours to hide.

_TAB_SUFFIXES = frozenset({".tsv", ".tab"})


def _delimiter_for(path: Path) -> str:
    """Pick the delimiter from the file extension (deterministic + honest — no
    sniffing that could silently mis-split). ``.tsv``/``.tab`` → tab, else comma."""
    return "\t" if path.suffix.lower() in _TAB_SUFFIXES else ","


def read_tabular(path: Path) -> Iterator[dict[str, str]]:
    """Stream a CSV/TSV file as one ``dict[str, str]`` per data row.

    The first row is the header and defines the keys. Values are always strings
    (empty cells → ``""``). Reads as ``utf-8-sig`` so a BOM from Excel/Windows
    exports is tolerated (matching the JSONL reader). Raises ``ValueError`` when
    the file has no header row (empty file), so callers' ``(OSError, ValueError)``
    handlers surface it cleanly.
    """
    delimiter = _delimiter_for(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter, restval="")
        if reader.fieldnames is None:
            raise ValueError("The file is empty — no header row was found.")
        header = [name for name in reader.fieldnames if name is not None]
        if not any(name.strip() for name in header):
            raise ValueError("The header row is empty — no column names were found.")

        for raw_row in reader:
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
    """
    rows_converted = 0
    columns: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in read_tabular(input_path):
            if not columns:
                columns = list(row.keys())
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_converted += 1
    return TabularConversion(
        output_path=str(output_path),
        rows_converted=rows_converted,
        columns=columns,
    )
