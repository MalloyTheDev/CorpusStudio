import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

# A project id becomes a directory name under the projects root, so restrict it
# to a safe slug that cannot traverse out of the root (no "/", "\\", or "..").
_PROJECT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


class SplitSettings(BaseModel):
    train_ratio: float = 0.9
    validation_ratio: float = 0.05
    seed: int = 42


class DatasetProject(BaseModel):
    id: str
    name: str
    schema_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    split_settings: SplitSettings = Field(default_factory=SplitSettings)


def create_project(root: Path, project: DatasetProject) -> Path:
    if not _PROJECT_ID_PATTERN.fullmatch(project.id or ""):
        raise ValueError(
            "Project id must start with a lowercase letter or digit and use only "
            "lowercase letters, digits, underscores, or hyphens."
        )

    project_dir = root / project.id
    if project_dir.exists():
        raise FileExistsError(f"Project already exists: {project_dir}")

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project.json").write_text(project.model_dump_json(indent=2), encoding="utf-8")
    (project_dir / "examples.jsonl").touch()
    return project_dir
