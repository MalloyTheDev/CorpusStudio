from pathlib import Path
import json
import shutil
from collections.abc import Iterable
from typing import Any


def export_jsonl(input_path: Path, output_path: Path) -> None:
    """Copy JSONL to an export path.

    This is intentionally simple for v0.1 skeleton work. Future exporters should
    perform schema mapping, field projection, split selection, and metadata control.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)


def write_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write rows as deterministic JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
