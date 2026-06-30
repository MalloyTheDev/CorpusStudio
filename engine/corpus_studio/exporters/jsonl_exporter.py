from pathlib import Path
import shutil


def export_jsonl(input_path: Path, output_path: Path) -> None:
    """Copy JSONL to an export path.

    This is intentionally simple for v0.1 skeleton work. Future exporters should
    perform schema mapping, field projection, split selection, and metadata control.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
