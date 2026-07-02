"""Project-local persistence for arena comparison reports (inspectable JSON)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from corpus_studio.arena.models import ArenaReport

ARENA_REPORTS_DIRNAME = "arena_reports"


def _slug(text: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(text).name).strip("_")
    return base or "arena"


def save_arena_report(project_dir: Path | str, report: ArenaReport, name: str) -> Path:
    """Write a report to arena_reports/<name>.json atomically; returns the path."""

    directory = Path(project_dir) / ARENA_REPORTS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(name)}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_arena_report(path: Path | str) -> ArenaReport:
    return ArenaReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def list_arena_reports(project_dir: Path | str) -> list[Path]:
    directory = Path(project_dir) / ARENA_REPORTS_DIRNAME
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))
