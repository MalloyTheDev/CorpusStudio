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


class GateThresholds(BaseModel):
    """Gate thresholds (defaults; overridable per project via gate_thresholds.json).

    Every field is bounded so a hand-edited override cannot silently disable or
    invert a gate: counts are non-negative, scores are non-negative and finite,
    and the pass-rate is a fraction in [0, 1]. NaN/inf and out-of-range values
    raise a ``ValidationError`` that ``load_gate_thresholds`` turns into a safe
    fall-back to these strict defaults (fail-closed) rather than a broken gate.
    """

    max_exact_duplicates: int = Field(default=0, ge=0)  # exceeding this is a finding
    block_exact_duplicates: bool = True  # block (True) vs warn (False) on exact dups
    max_normalized_duplicates: int = Field(default=0, ge=0)  # warn above this
    max_low_information: int = Field(default=0, ge=0)  # warn above this
    warn_synthetic_pattern_issues: int = Field(default=1, ge=0)  # warn at/above this many
    block_on_high_severity_pii: bool = True
    warn_on_medium_severity_pii: bool = True
    min_eval_average_score: float = Field(default=70.0, ge=0.0, allow_inf_nan=False)
    min_eval_pass_rate: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)  # fraction that must pass
    max_regression_score_drop: float = Field(default=2.0, ge=0.0, allow_inf_nan=False)  # block if avg drops more than this


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
    thresholds: GateThresholds | None = None  # effective thresholds behind the verdict (for reproducibility)

    @classmethod
    def build(
        cls,
        scope: GateScope,
        target: str,
        results: list[GateResult],
        generated_at: str | None = None,
        thresholds: GateThresholds | None = None,
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
            thresholds=thresholds,
        )

    @property
    def blocked(self) -> bool:
        return self.overall_status == GateStatus.BLOCK


def gate_thresholds_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / GATE_THRESHOLDS_FILENAME


def load_gate_thresholds(project_dir: Path | str) -> GateThresholds:
    """Effective thresholds: project-local ``gate_thresholds.json`` merged over the
    defaults. Missing keys keep their default, unknown keys are ignored, and an
    absent/unreadable/invalid file falls back to defaults (never crashes).

    The file is read as UTF-8 with a tolerated BOM (``utf-8-sig``) so overrides
    saved by Notepad or PowerShell — which prepend a BOM — are not silently
    discarded. A value that is out of range or non-finite fails validation and
    falls back to the strict defaults rather than producing a broken gate.
    """

    path = gate_thresholds_path(project_dir)
    if not path.exists():
        return GateThresholds()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return GateThresholds()
    if not isinstance(data, dict):
        return GateThresholds()

    known = {key: data[key] for key in GateThresholds.model_fields if key in data}
    try:
        return GateThresholds(**known)
    except Exception:  # noqa: BLE001 - a bad/out-of-range value falls back to defaults.
        return GateThresholds()
