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


def append_examples(project_dir: Path | str, rows: list[Any]) -> int:
    """Append ``rows`` to ``examples.jsonl`` atomically (read existing + concat +
    atomic replace, so a reader never sees a partial file), under the single-writer
    lock. Returns the number of rows appended. The caller validates ``rows`` first."""

    new_lines = _rows_to_lines(rows)
    with single_writer_lock(project_dir):
        existing = read_existing_lines(project_dir)
        _atomic_write_lines(examples_path(project_dir), existing + new_lines)
    return len(new_lines)


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
