"""CorpusStudio platform contracts — the language-neutral boundary substrate.

This package defines the versioned contracts that bind the (Python → Rust) platform core, the Python
AI backend worker(s), and the UI shell (Avalonia now, Tauri later): the immutable ``RunPlan`` a core
dispatches, the ``RunEvent`` stream a worker emits back, the ``BackendManifest``/``CapabilityReport``
that let the core pick a backend before dispatch, the ``EnvironmentProfile`` signature, the
``FailureRecord``/``FitClassification`` taxonomies that make a silent WDDM spill machine-actionable,
and the ``WorkerMessage`` protocol envelope — plus the existing engine records (Project/Dataset/Run/
Artifact/Evaluation) formalized as versioned contracts.

The pydantic models here are the source of truth; :func:`export_json_schemas` generates the
language-neutral JSON Schemas the non-Python clients consume. Pure — ``import corpus_studio.platform``
pulls no torch/transformers/etc.
"""

from __future__ import annotations

from .common import (
    CONTRACT_VERSION,
    ContractModel,
    HashRef,
    License,
    MemoryMetrics,
    PackageLock,
    Ref,
    TokenStats,
)
from .contracts import (
    ArtifactManifest,
    BackendManifest,
    CapabilityReport,
    DatasetManifest,
    EnvironmentProfile,
    EnvironmentDescriptor,
    EnvironmentHealthReport,
    EnvironmentInstallation,
    EnvironmentLock,
    EnvironmentRecipe,
    EvaluationResult,
    FailureRecord,
    FitClassification,
    ModelDescriptor,
    ProjectManifest,
    PythonRuntime,
    RunEvent,
    RunManifest,
    RunPlan,
    TokenizerDescriptor,
    WorkerMessage,
    WORKER_BODY_BY_TYPE,
)
from .profiler import build_environment_profile
from .probes import BUILTIN_PROBES, ProbeOutcome, run_capability_probes
from .schema_export import ROOT_CONTRACTS, contract_schemas, export_json_schemas

__all__ = [
    "build_environment_profile",
    "run_capability_probes",
    "BUILTIN_PROBES",
    "ProbeOutcome",
    "CONTRACT_VERSION",
    "ContractModel",
    "HashRef",
    "Ref",
    "PackageLock",
    "License",
    "MemoryMetrics",
    "TokenStats",
    "ProjectManifest",
    "DatasetManifest",
    "ModelDescriptor",
    "TokenizerDescriptor",
    "EnvironmentProfile",
    "PythonRuntime",
    "EnvironmentRecipe",
    "EnvironmentInstallation",
    "EnvironmentLock",
    "EnvironmentDescriptor",
    "EnvironmentHealthReport",
    "BackendManifest",
    "CapabilityReport",
    "RunPlan",
    "RunManifest",
    "RunEvent",
    "ArtifactManifest",
    "EvaluationResult",
    "FailureRecord",
    "FitClassification",
    "WorkerMessage",
    "WORKER_BODY_BY_TYPE",
    "ROOT_CONTRACTS",
    "contract_schemas",
    "export_json_schemas",
]
