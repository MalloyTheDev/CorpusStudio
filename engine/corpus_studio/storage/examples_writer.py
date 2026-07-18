"""Sanctioned single-writer for a project's ``examples.jsonl`` (v1.0).

Historically the engine **refused** to write ``examples.jsonl`` and delegated the
single-writer role to the desktop app. As the desktop is decommissioned (#545),
this module makes the engine the sanctioned single writer: it rewrites
``examples.jsonl`` **atomically** (a temp file in the same directory + ``fsync`` +
``os.replace``) under an **exclusive advisory lock**, so the *single-writer*
honesty invariant still holds - one writer at a time, and a reader never observes
a partial file.

Torch-free. Writes only ``examples.jsonl`` and its sibling ``.lock`` file. This
module owns *durability and atomicity*; schema validation is the caller's job
(rows are validated against the project schema before they reach here).
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

EXAMPLES_FILENAME = "examples.jsonl"
_LOCK_SUFFIX = ".lock"


class ExamplesLockedError(RuntimeError):
    """Raised when the single-writer lock is already held by another writer."""


def examples_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / EXAMPLES_FILENAME


def _lock_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / (EXAMPLES_FILENAME + _LOCK_SUFFIX)


@contextmanager
def single_writer_lock(project_dir: Path | str) -> Iterator[None]:
    """Hold an exclusive advisory lock guarding ``examples.jsonl`` writes.

    Uses ``fcntl.flock`` where available (auto-released on process exit, so a
    crash leaves no stale lock). Where ``fcntl`` is unavailable (non-POSIX) it
    degrades to a no-op - the atomic replace still prevents a torn file for
    single-process CLI use. Raises :class:`ExamplesLockedError` if another holder
    currently owns the lock.
    """

    lock_file = _lock_path(project_dir)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        yield
        return
    handle = lock_file.open("w", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ExamplesLockedError(
                f"examples.jsonl is locked by another writer ({lock_file})."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def read_existing_lines(project_dir: Path | str) -> list[str]:
    """The existing rows of ``examples.jsonl`` as verbatim JSON strings (trailing
    newline stripped, blank lines dropped). A missing file is an empty list. Row
    content is preserved exactly - append never reformats already-authored rows."""

    path = examples_path(project_dir)
    if not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.rstrip("\n").rstrip("\r")
            if stripped.strip() == "":
                continue
            lines.append(stripped)
    return lines


def read_examples_page(
    project_dir: Path | str, *, offset: int = 0, limit: int | None = None
) -> tuple[int, list[dict[str, Any]]]:
    """Read a paged window of ``examples.jsonl`` for inspection/curation (read-only).

    Returns ``(total, rows)`` where ``total`` is the whole file's row count and ``rows`` is the
    ``limit`` rows starting at ``offset`` (0-based; ``limit=None`` returns all from ``offset``). Each
    entry carries its 1-based ``row_number`` (its absolute, stable address - the same address the
    mutation commands use) and the parsed ``example``. A line that is not valid JSON is surfaced (never
    silently dropped) as ``parse_error`` + a short ``raw_preview`` so a curation screen can show and
    fix it. The client must not parse ``examples.jsonl`` itself; row identity is engine logic."""

    lines = read_existing_lines(project_dir)
    total = len(lines)
    start = max(offset, 0)
    window = lines[start:] if limit is None else lines[start : start + max(limit, 0)]
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(window):
        row_number = start + index + 1
        try:
            rows.append({"row_number": row_number, "example": json.loads(line)})
        except json.JSONDecodeError as exc:
            rows.append(
                {"row_number": row_number, "parse_error": str(exc), "raw_preview": line[:200]}
            )
    return total, rows


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    """Write each item of ``lines`` as one ``\\n``-terminated row to ``path``
    atomically: a temp file in the same directory, ``fsync``, then ``os.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()  # best-effort cleanup if the replace never happened


def _rows_to_lines(rows: Iterable[Any]) -> list[str]:
    # ensure_ascii off keeps non-ASCII rows human-inspectable; dict insertion
    # order is preserved, so authored field order survives.
    return [json.dumps(row, ensure_ascii=False) for row in rows]


def append_examples_locked(project_dir: Path | str, rows: list[Any]) -> int:
    """Append ``rows`` to ``examples.jsonl`` atomically **without acquiring the lock** -
    the caller MUST already hold :func:`single_writer_lock`. For a compound critical
    section (e.g. append + version capture) that must be one atomic unit so a concurrent
    writer cannot interleave. Returns the count appended; the caller validates ``rows`` first."""

    new_lines = _rows_to_lines(rows)
    existing = read_existing_lines(project_dir)
    _atomic_write_lines(examples_path(project_dir), existing + new_lines)
    return len(new_lines)


def append_examples(project_dir: Path | str, rows: list[Any]) -> int:
    """Append ``rows`` to ``examples.jsonl`` atomically under the single-writer lock
    (read existing + concat + atomic replace, so a reader never sees a partial file).
    Returns the number of rows appended. The caller validates ``rows`` first."""

    with single_writer_lock(project_dir):
        return append_examples_locked(project_dir, rows)


def write_examples(project_dir: Path | str, rows: list[Any]) -> int:
    """Atomically **replace** ``examples.jsonl`` with exactly ``rows`` (serialized
    here), under the single-writer lock. Returns the row count written."""

    lines = _rows_to_lines(rows)
    with single_writer_lock(project_dir):
        _atomic_write_lines(examples_path(project_dir), lines)
    return len(lines)


def write_examples_lines(project_dir: Path | str, lines: Iterable[str]) -> int:
    """Atomically **replace** ``examples.jsonl`` with pre-serialized JSONL ``lines``
    (each one JSON row, no trailing newline), under the single-writer lock. Used to
    adopt a verified version reconstruction verbatim. Returns the row count."""

    materialized = list(lines)
    with single_writer_lock(project_dir):
        _atomic_write_lines(examples_path(project_dir), materialized)
    return len(materialized)


def replace_examples_lines_locked(project_dir: Path | str, lines: Iterable[str]) -> int:
    """Atomically replace ``examples.jsonl`` **without acquiring the lock** - the caller
    MUST already hold :func:`single_writer_lock`. For a compound critical section
    (snapshot -> verify -> swap) that must be one atomic unit, so a concurrent writer
    cannot interleave between the snapshot and the replace. Returns the row count."""

    materialized = list(lines)
    _atomic_write_lines(examples_path(project_dir), materialized)
    return len(materialized)
