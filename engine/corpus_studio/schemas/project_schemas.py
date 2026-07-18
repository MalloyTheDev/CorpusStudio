"""Project-local dataset schemas (torch-free, control-plane).

Only builtin schemas ship in the repo (``schemas/builtin/<id>.schema.json``). This module lets a
project keep its OWN editable schema copies under ``<project>/schemas/<id>.schema.json`` - derive a
copy of a builtin, tighten it (pin a label enum, add a field, make a field required), then validate
against it. Storage mirrors the other project-local configs (``gate_thresholds.json`` /
``provenance_allowlist.json``): a ``*_path`` helper plus load/save.

``resolve_schema`` prefers a project-local schema over the builtin of the same id - an explicit,
reported shadowing (``source == 'project'``), never a silent one. The schema id rule is the builtin
loader's exact allowlist, so a project schema id can never traverse the filesystem either.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from corpus_studio.schemas.base import DatasetSchema
from corpus_studio.schemas.registry import _SCHEMA_ID_PATTERN, load_builtin_schema

PROJECT_SCHEMA_DIRNAME = "schemas"


class SchemaError(ValueError):
    """A project-local schema id is invalid, or the schema could not be read/parsed."""


def _validate_id(schema_id: str) -> str:
    if not _SCHEMA_ID_PATTERN.fullmatch(schema_id or ""):
        raise SchemaError(
            f"invalid schema id '{schema_id}': use lowercase letters, digits, and underscore only."
        )
    return schema_id


def project_schema_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / PROJECT_SCHEMA_DIRNAME


def project_schema_path(project_dir: Path | str, schema_id: str) -> Path:
    return project_schema_dir(project_dir) / f"{_validate_id(schema_id)}.schema.json"


def has_project_schema(project_dir: Path | str, schema_id: str) -> bool:
    return project_schema_path(project_dir, schema_id).exists()


def load_project_schema(project_dir: Path | str, schema_id: str) -> DatasetSchema:
    """Load a project-local schema. Raises :class:`SchemaError` if it is absent or malformed."""
    path = project_schema_path(project_dir, schema_id)
    if not path.exists():
        raise SchemaError(
            f"no project-local schema '{schema_id}' under {project_schema_dir(project_dir)}."
        )
    try:
        return DatasetSchema.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValueError) as exc:
        raise SchemaError(f"project-local schema '{schema_id}' is malformed: {exc}") from exc


def save_project_schema(project_dir: Path | str, schema: DatasetSchema) -> Path:
    """Write ``schema`` to ``<project>/schemas/<schema.id>.schema.json`` atomically (temp + replace)."""
    path = project_schema_path(project_dir, schema.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(schema.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def resolve_schema(project_dir: Path | str | None, schema_id: str) -> tuple[DatasetSchema, str]:
    """Resolve ``schema_id`` to ``(schema, source)``.

    A project-local schema shadows the builtin of the same id (``source == 'project'``); otherwise the
    builtin (``source == 'builtin'``). Raises :class:`SchemaError` / ``ValueError`` if neither exists.
    """
    if project_dir is not None and has_project_schema(project_dir, schema_id):
        return load_project_schema(project_dir, schema_id), "project"
    return load_builtin_schema(schema_id), "builtin"


def project_schema_entries(project_dir: Path | str) -> list[tuple[str, DatasetSchema | None]]:
    """Every project-local schema FILE as ``(id, schema-or-None)``, sorted by id.

    ``id`` is the filename stem - the RESOLUTION key: a file present shadows the builtin of that id
    even when malformed (that is exactly what :func:`resolve_schema` keys on, via file existence).
    ``schema`` is ``None`` when the file cannot be parsed, so a caller (e.g. schema-list) can SURFACE
    the broken shadow instead of hiding it - keeping the inventory consistent with resolution."""
    directory = project_schema_dir(project_dir)
    if not directory.exists():
        return []
    entries: list[tuple[str, DatasetSchema | None]] = []
    for path in sorted(directory.glob("*.schema.json")):
        schema_id = path.name.removesuffix(".schema.json")
        try:
            entries.append(
                (schema_id, DatasetSchema.model_validate(json.loads(path.read_text(encoding="utf-8"))))
            )
        except (json.JSONDecodeError, ValueError):
            entries.append((schema_id, None))
    return entries


def list_project_schemas(project_dir: Path | str) -> list[DatasetSchema]:
    """Every readable project-local schema, sorted by id (malformed files omitted). For an inventory
    that must stay consistent with resolution, use :func:`project_schema_entries` instead."""
    return [schema for _, schema in project_schema_entries(project_dir) if schema is not None]
