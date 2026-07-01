import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    # utf-8-sig tolerates a leading UTF-8 BOM (common in Excel/Windows exports)
    # so import/validate stay consistent with the preview reader.
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
