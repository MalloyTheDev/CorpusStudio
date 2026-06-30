def normalize_text(value: str) -> str:
    """Basic text normalization for v0.1."""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def remove_empty_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if any(str(value).strip() for value in row.values())]
