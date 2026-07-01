import json
import re
from pathlib import Path

from corpus_studio.schemas.base import DatasetSchema

# Schema ids are simple lowercase identifiers; anything else (e.g. path
# separators or "..") is rejected so a schema id can never traverse the
# filesystem when composed into a .schema.json path.
_SCHEMA_ID_PATTERN = re.compile(r"[a-z0-9_]+")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def builtin_schema_dir() -> Path:
    return repository_root() / "schemas" / "builtin"


def load_builtin_schema(schema_id: str) -> DatasetSchema:
    if not _SCHEMA_ID_PATTERN.fullmatch(schema_id or ""):
        raise ValueError(f"Unknown schema: {schema_id}")

    path = builtin_schema_dir() / f"{schema_id}.schema.json"
    if not path.exists():
        raise ValueError(f"Unknown schema: {schema_id}")
    return DatasetSchema.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_builtin_schemas() -> list[DatasetSchema]:
    schemas = []
    for path in sorted(builtin_schema_dir().glob("*.schema.json")):
        schemas.append(DatasetSchema.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    return schemas
