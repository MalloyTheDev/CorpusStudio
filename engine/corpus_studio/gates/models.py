"""Serializable gate result/report types."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

GATE_THRESHOLDS_FILENAME = "gate_thresholds.json"


class GateScope(str, Enum):
    DATASET = "dataset"
    ROW = "row"
    IMPORT = "import"
    EXPORT = "export"
    SPLIT = "split"
    EVALUATION_REPORT = "evaluation_report"
    TRAINING_RUN = "training_run"
    MODEL_ARTIFACT = "model_artifact"
    CHAT_SUITE = "chat_suite"


class GateStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


_STATUS_ORDER = {GateStatus.PASS: 0, GateStatus.WARN: 1, GateStatus.BLOCK: 2}


def worst_status(statuses: list[GateStatus]) -> GateStatus:
    return max(statuses, key=lambda status: _STATUS_ORDER[status], default=GateStatus.PASS)


class GateResult(BaseModel):
    """Outcome of one gate check."""

    gate_id: str
    name: str
    scope: GateScope
    status: GateStatus
    observed: str = ""
    expected: str = ""
    affected: list[str] = Field(default_factory=list)
    message: str = ""
    repair: str | None = None


class GateReport(BaseModel):
    """A set of gate results for one target, with an overall status."""

    scope: GateScope
    target: str
    generated_at: str | None = None
    overall_status: GateStatus = GateStatus.PASS
    pass_count: int = 0
    warn_count: int = 0
    block_count: int = 0
    results: list[GateResult] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        scope: GateScope,
        target: str,
        results: list[GateResult],
        generated_at: str | None = None,
    ) -> "GateReport":
        statuses = [result.status for result in results]
        return cls(
            scope=scope,
            target=target,
            generated_at=generated_at,
            overall_status=worst_status(statuses),
            pass_count=sum(1 for status in statuses if status == GateStatus.PASS),
            warn_count=sum(1 for status in statuses if status == GateStatus.WARN),
            block_count=sum(1 for status in statuses if status == GateStatus.BLOCK),
            results=results,
        )

    @property
    def blocked(self) -> bool:
        return self.overall_status == GateStatus.BLOCK


class GateThresholds(BaseModel):
    """Default gate thresholds (designed for future per-project configuration)."""

    max_exact_duplicates: int = 0  # exceeding this is a finding
    block_exact_duplicates: bool = True  # block (True) vs warn (False) on exact dups
    max_normalized_duplicates: int = 0  # warn above this
    max_low_information: int = 0  # warn above this
    warn_synthetic_pattern_issues: int = 1  # warn at/above this many issues
    block_on_high_severity_pii: bool = True
    warn_on_medium_severity_pii: bool = True
    min_eval_average_score: float = 70.0
    min_eval_pass_rate: float = 0.5  # fraction of examples that must pass
    max_regression_score_drop: float = 2.0  # block if trained avg drops more than this


def gate_thresholds_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / GATE_THRESHOLDS_FILENAME


def load_gate_thresholds(project_dir: Path | str) -> GateThresholds:
    """Effective thresholds: project-local ``gate_thresholds.json`` merged over the
    defaults. Missing keys keep their default, unknown keys are ignored, and an
    absent/unreadable/invalid file falls back to defaults (never crashes).
    """

    path = gate_thresholds_path(project_dir)
    if not path.exists():
        return GateThresholds()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return GateThresholds()
    if not isinstance(data, dict):
        return GateThresholds()

    known = {key: data[key] for key in GateThresholds.model_fields if key in data}
    try:
        return GateThresholds(**known)
    except Exception:  # noqa: BLE001 - a bad value type falls back to defaults.
        return GateThresholds()
