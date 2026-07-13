"""Generate the language-neutral JSON Schemas from the pydantic contracts.

The pydantic models in :mod:`corpus_studio.platform.contracts` are the single source of truth; the
Rust core, Avalonia, and Tauri shells consume the JSON Schemas this module emits. Keeping generation
here (rather than hand-maintaining parallel `.json` files) means the schemas can never drift from the
Python models. Pure — no heavy imports.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import CONTRACT_VERSION
from .contracts import (
    ArtifactManifest,
    BackendManifest,
    CapabilityReport,
    ContractModel,
    DatasetManifest,
    DependencyResolution,
    EnvironmentDescriptor,
    EnvironmentHealthReport,
    EnvironmentInstallation,
    EnvironmentLock,
    EnvironmentProfile,
    EnvironmentRecipe,
    EvaluationResult,
    FailureRecord,
    FitClassification,
    ModelDescriptor,
    ObjectiveCompatibilityReport,
    ProjectManifest,
    PythonRuntime,
    RunEvent,
    RunManifest,
    RunPlan,
    StorageProfile,
    TokenizerDescriptor,
    TrainingObjective,
    WorkerMessage,
)

# The root contracts, in a stable documented order. Name → model class.
ROOT_CONTRACTS: dict[str, type[ContractModel]] = {
    "ProjectManifest": ProjectManifest,
    "DatasetManifest": DatasetManifest,
    "ModelDescriptor": ModelDescriptor,
    "TokenizerDescriptor": TokenizerDescriptor,
    "TrainingObjective": TrainingObjective,
    "ObjectiveCompatibilityReport": ObjectiveCompatibilityReport,
    "EnvironmentProfile": EnvironmentProfile,
    "StorageProfile": StorageProfile,
    "PythonRuntime": PythonRuntime,
    "EnvironmentRecipe": EnvironmentRecipe,
    "DependencyResolution": DependencyResolution,
    "EnvironmentInstallation": EnvironmentInstallation,
    "EnvironmentLock": EnvironmentLock,
    "EnvironmentDescriptor": EnvironmentDescriptor,
    "EnvironmentHealthReport": EnvironmentHealthReport,
    "BackendManifest": BackendManifest,
    "CapabilityReport": CapabilityReport,
    "RunPlan": RunPlan,
    "RunManifest": RunManifest,
    "RunEvent": RunEvent,
    "ArtifactManifest": ArtifactManifest,
    "EvaluationResult": EvaluationResult,
    "FailureRecord": FailureRecord,
    "FitClassification": FitClassification,
    "WorkerMessage": WorkerMessage,
}


def contract_schemas() -> dict[str, dict]:
    """Return ``{contract_name: json_schema_dict}`` for every root contract."""
    return {name: model.model_json_schema() for name, model in ROOT_CONTRACTS.items()}


def export_json_schemas(out_dir: str | Path) -> list[Path]:
    """Write ``<name>.schema.json`` for every root contract into ``out_dir`` plus an ``index.json``
    describing the set. Returns the paths written (schemas first, then the index). Deterministic
    output (sorted keys) so a regenerated schema diffs cleanly in review."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index: list[dict[str, str]] = []
    for name, schema in contract_schemas().items():
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
        index.append({"contract": name, "file": path.name})
    index_path = out / "index.json"
    index_path.write_text(
        json.dumps(
            {"contract_version": CONTRACT_VERSION, "contracts": index}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(index_path)
    return written
