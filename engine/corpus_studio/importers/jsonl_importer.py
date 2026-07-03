import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    # utf-8-sig tolerates a leading UTF-8 BOM (common in Excel/Windows exports)
    # so import/validate stay consistent with the preview reader.
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            # A JSONL dataset row must be a JSON object. A non-object line
            # (list/scalar/null) is malformed input — raise a ValueError so
            # callers' (OSError, ValueError) handlers surface it cleanly instead
            # of a downstream AttributeError from row.values()/row.items().
            if not isinstance(row, dict):
                raise ValueError(
                    f"line {line_number}: expected a JSON object, got {type(row).__name__}."
                )
            yield row
