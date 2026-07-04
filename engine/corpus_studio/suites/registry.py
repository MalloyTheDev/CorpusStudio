"""Project-local suite registry (v1.3 M2): suites become first-class files under
``evaluation_suites/<name>.json``. Create (scaffold), list, and resolve-by-name — the
run itself is unchanged M1 (``runner.run_suite``). The filename stem is the registry key.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from corpus_studio.suites.models import SuiteDefinition, SuiteSummary
from corpus_studio.suites.runner import load_suite_definition

EVALUATION_SUITES_DIRNAME = "evaluation_suites"

# Same charset as SuiteDefinition.name; fullmatch BEFORE composing a path so a crafted
# name (``../x``, ``a/b``) can never escape evaluation_suites/ (mirrors load_builtin_schema).
_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def suites_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / EVALUATION_SUITES_DIRNAME


def suite_definition_path(project_dir: Path | str, name: str) -> Path:
    """The registry path for a suite name. Raises ValueError on a bad/traversal name."""

    if not _NAME_PATTERN.fullmatch(name or ""):
        raise ValueError(f"Invalid suite name: {name!r} (allowed: letters, digits, . _ -).")
    return suites_dir(project_dir) / f"{name}.json"


def _example_definition(name: str) -> dict:
    return {
        "name": name,
        "cases": [
            {
                "name": "example-case",
                "schema": "instruction",
                "dataset_path": "data/your_dataset.jsonl",
                "model": "your-model",
                "backend": "ollama",
                "metric": "keyword_overlap",
                "min_score": 70,
                "min_pass_rate": 0.5,
            }
        ],
    }


def scaffold_suite(project_dir: Path | str, name: str, force: bool = False) -> Path:
    """Write an example suite definition to evaluation_suites/<name>.json (atomic).
    Refuses to overwrite an existing suite unless ``force`` (never silent clobber)."""

    path = suite_definition_path(project_dir, name)
    if path.exists() and not force:
        raise FileExistsError(f"Suite '{name}' already exists at {path}; pass --force to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_example_definition(name), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def list_suite_definitions(project_dir: Path | str) -> list[SuiteSummary]:
    """List registered suites (by filename stem), tolerating malformed files — a bad
    definition is reported as valid=False with an error, never a crash."""

    directory = suites_dir(project_dir)
    if not directory.is_dir():
        return []

    summaries: list[SuiteSummary] = []
    for path in sorted(directory.glob("*.json")):
        name = path.stem
        try:
            definition = load_suite_definition(path)
            summaries.append(SuiteSummary(name=name, case_count=len(definition.cases), valid=True))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            summaries.append(SuiteSummary(name=name, case_count=0, valid=False, error=str(exc)))
    return summaries


def load_suite_by_name(project_dir: Path | str, name: str) -> SuiteDefinition:
    """Resolve + load a registered suite by name. Raises ValueError (bad name / invalid
    file) or FileNotFoundError (no such suite)."""

    path = suite_definition_path(project_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"No suite named '{name}' in {EVALUATION_SUITES_DIRNAME}/.")
    return load_suite_definition(path)
