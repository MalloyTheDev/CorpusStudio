"""Optional SQLite-backed index over local dataset projects.

The index is a derived cache. ``project.json`` and ``examples.jsonl`` remain the
source of truth and stay fully human-inspectable; the index can be rebuilt from
the filesystem at any time with :func:`rebuild_index`. Indexing is opt-in --
nothing in the engine writes to SQLite unless a caller explicitly asks for it,
so projects created without the index behave exactly as before.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

DEFAULT_INDEX_FILENAME = "index.sqlite3"


class ProjectIndexEntry(BaseModel):
    """A single indexed project row, mirrored from its ``project.json``."""

    id: str
    name: str
    schema_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    example_count: int = 0
    path: str = ""
    indexed_at: str = ""


def default_index_path(projects_root: Path) -> Path:
    """Return the conventional index location for a projects root."""
    return projects_root / DEFAULT_INDEX_FILENAME


def open_index(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite index and ensure its schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            schema_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            example_count INTEGER NOT NULL DEFAULT 0,
            path TEXT NOT NULL DEFAULT '',
            indexed_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    connection.commit()
    return connection


def _count_examples(project_dir: Path) -> int:
    examples_path = project_dir / "examples.jsonl"
    if not examples_path.exists():
        return 0

    count = 0
    with examples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def read_project_entry(project_dir: Path) -> Optional[ProjectIndexEntry]:
    """Build an index entry from a project directory, or ``None`` if unreadable."""
    project_file = project_dir / "project.json"
    if not project_file.exists():
        return None

    try:
        metadata = json.loads(project_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(metadata, dict):
        return None

    project_id = str(metadata.get("id") or project_dir.name)
    return ProjectIndexEntry(
        id=project_id,
        name=str(metadata.get("name", project_dir.name)),
        schema_id=str(metadata.get("schema_id", "")),
        created_at=str(metadata.get("created_at", "")),
        updated_at=str(metadata.get("updated_at", "")),
        example_count=_count_examples(project_dir),
        path=str(project_dir),
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )


def upsert_project(connection: sqlite3.Connection, entry: ProjectIndexEntry) -> None:
    """Insert or update a single project entry."""
    connection.execute(
        """
        INSERT INTO projects (
            id, name, schema_id, created_at, updated_at, example_count, path, indexed_at
        )
        VALUES (
            :id, :name, :schema_id, :created_at, :updated_at, :example_count, :path, :indexed_at
        )
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            schema_id=excluded.schema_id,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at,
            example_count=excluded.example_count,
            path=excluded.path,
            indexed_at=excluded.indexed_at
        """,
        entry.model_dump(),
    )
    connection.commit()


def remove_project(connection: sqlite3.Connection, project_id: str) -> None:
    """Drop a project from the index (used when a project folder is deleted)."""
    connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    connection.commit()


def index_project_dir(
    connection: sqlite3.Connection, project_dir: Path
) -> Optional[ProjectIndexEntry]:
    """Read and upsert one project directory. Returns the entry or ``None``."""
    entry = read_project_entry(project_dir)
    if entry is None:
        return None
    upsert_project(connection, entry)
    return entry


def _iter_project_dirs(projects_root: Path) -> Iterator[Path]:
    if not projects_root.is_dir():
        return
    for child in sorted(projects_root.iterdir()):
        if child.is_dir() and (child / "project.json").exists():
            yield child


def rebuild_index(projects_root: Path, db_path: Optional[Path] = None) -> int:
    """Rebuild the index from the filesystem source of truth.

    Every ``project.json`` under ``projects_root`` is re-read and upserted, and
    entries whose folders no longer exist are pruned. Returns the entry count.
    """
    resolved_db = db_path or default_index_path(projects_root)
    connection = open_index(resolved_db)
    try:
        connection.execute("DELETE FROM projects")
        count = 0
        for project_dir in _iter_project_dirs(projects_root):
            entry = read_project_entry(project_dir)
            if entry is not None:
                upsert_project(connection, entry)
                count += 1
        return count
    finally:
        connection.close()


def index_single_project(
    projects_root: Path, project_dir: Path, db_path: Optional[Path] = None
) -> Optional[ProjectIndexEntry]:
    """Open the index, upsert one project directory, and close. Opt-in helper."""
    resolved_db = db_path or default_index_path(projects_root)
    connection = open_index(resolved_db)
    try:
        return index_project_dir(connection, project_dir)
    finally:
        connection.close()


def list_projects(
    connection: sqlite3.Connection,
    schema_id: Optional[str] = None,
    name_contains: Optional[str] = None,
) -> list[ProjectIndexEntry]:
    """Query indexed projects with optional schema and name-substring filters."""
    query = "SELECT * FROM projects"
    clauses: list[str] = []
    params: list[object] = []
    if schema_id:
        clauses.append("schema_id = ?")
        params.append(schema_id)
    if name_contains:
        clauses.append("LOWER(name) LIKE ?")
        params.append(f"%{name_contains.lower()}%")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY name COLLATE NOCASE"

    rows = connection.execute(query, params).fetchall()
    return [ProjectIndexEntry(**dict(row)) for row in rows]


def list_projects_from_root(
    projects_root: Path,
    db_path: Optional[Path] = None,
    schema_id: Optional[str] = None,
    name_contains: Optional[str] = None,
    rebuild: bool = False,
) -> list[ProjectIndexEntry]:
    """List projects via the index, rebuilding it from disk when missing or stale.

    The index is created on first use only; it never replaces the JSON/JSONL
    files and can be deleted at any time without data loss.
    """
    resolved_db = db_path or default_index_path(projects_root)
    if rebuild or not resolved_db.exists():
        rebuild_index(projects_root, resolved_db)

    connection = open_index(resolved_db)
    try:
        return list_projects(
            connection, schema_id=schema_id, name_contains=name_contains
        )
    finally:
        connection.close()
