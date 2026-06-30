from pathlib import Path


def import_text_file(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    return [{"text": text, "source": str(path)}]
