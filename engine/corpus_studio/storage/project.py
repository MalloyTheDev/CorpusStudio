from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class DatasetProject(BaseModel):
    id: str
    name: str
    schema_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def create_project(root: Path, project: DatasetProject) -> Path:
    project_dir = root / project.id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project.json").write_text(project.model_dump_json(indent=2), encoding="utf-8")
    (project_dir / "examples.jsonl").touch()
    return project_dir
