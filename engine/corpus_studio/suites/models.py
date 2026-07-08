"""Serializable evaluation-suite definition + report types (v1.3 M1, file-driven).

A suite is a project-local JSON file the user writes; ``suite-run`` consumes it
immediately (no registry in M1). Reports roll up PER METRIC — a suite verdict never
folds non-comparable metric scales (keyword_overlap vs llm_judge) into one number.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from corpus_studio.gates.models import GateReport, GateStatus

_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")

SuiteMetric = Literal["keyword_overlap", "llm_judge"]
SuiteBackend = Literal["ollama", "openai-compatible"]
SuiteCaseStatus = Literal["pass", "warn", "block", "error"]


class SuiteCase(BaseModel):
    """One evaluation case: a dataset × model × metric × pass bars."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    # The dataset schema id (instruction/chat/…). Aliased to "schema" in the JSON;
    # the field is named schema_id to avoid shadowing pydantic's BaseModel.schema.
    schema_id: str = Field(alias="schema")
    # Exactly one dataset source: a mutable path, OR a pinned dataset version_id
    # (resolved to its VERIFIED reconstruction at run time — true reproducibility).
    dataset_path: str | None = None
    version_id: str | None = None
    model: str
    backend: SuiteBackend = "ollama"
    base_url: str | None = None
    metric: SuiteMetric = "keyword_overlap"
    judge_model: str | None = None
    judge_backend: str = "ollama"
    judge_base_url: str | None = None
    limit: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=0.0)
    min_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_PATTERN.fullmatch(value or ""):
            raise ValueError(f"Invalid case name: {value!r} (allowed: letters, digits, . _ -).")
        return value

    @model_validator(mode="after")
    def _validate_case(self) -> "SuiteCase":
        if self.metric == "llm_judge" and not self.judge_model:
            raise ValueError(f"Case '{self.name}': metric 'llm_judge' requires a judge_model.")
        has_path = bool(self.dataset_path)
        has_version = bool(self.version_id)
        if has_path == has_version:  # neither, or both
            raise ValueError(
                f"Case '{self.name}': set exactly one of dataset_path or version_id."
            )
        return self


class SuiteDefinition(BaseModel):
    name: str
    cases: list[SuiteCase] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_PATTERN.fullmatch(value or ""):
            raise ValueError(f"Invalid suite name: {value!r} (allowed: letters, digits, . _ -).")
        return value


class SuiteCaseResult(BaseModel):
    case: str
    model: str
    metric: str
    version_id: str | None = None  # the pinned version this case ran (reproducibility record)
    dataset_fingerprint: str | None = None  # content fingerprint at run time (honest record of WHAT ran)
    examples_tested: int | None = None
    average_score: float | None = None
    pass_rate: float | None = None
    gate: GateReport | None = None  # the per-case evaluation_report gate; None on error
    error: str | None = None
    status: SuiteCaseStatus


class SuiteMetricRollup(BaseModel):
    metric: str
    total: int
    passed: int
    warned: int
    blocked: int
    errored: int


class SuiteReport(BaseModel):
    suite: str
    generated_at: str | None = None
    cases: list[SuiteCaseResult] = Field(default_factory=list)
    per_metric: list[SuiteMetricRollup] = Field(default_factory=list)  # per-metric roll-up, never folded
    overall_status: GateStatus = GateStatus.PASS  # worst per-case; an errored case blocks the suite
    summary: str = ""


class SuiteHistoryEntry(BaseModel):
    """One point in a suite's run history (issue #190): the run time, the aggregate verdict, and the
    per-status case counts — enough to trend pass/warn/block over time without storing every case.
    The per-metric roll-up is intentionally NOT folded into a single score; these counts are a sum of
    case outcomes across metrics, which is a count, not a quality number."""

    generated_at: str | None = None
    overall_status: GateStatus = GateStatus.PASS
    total: int = 0
    passed: int = 0
    warned: int = 0
    blocked: int = 0
    errored: int = 0
    summary: str = ""

    @classmethod
    def from_report(cls, report: SuiteReport) -> "SuiteHistoryEntry":
        totals = {"total": 0, "passed": 0, "warned": 0, "blocked": 0, "errored": 0}
        for rollup in report.per_metric:
            totals["total"] += rollup.total
            totals["passed"] += rollup.passed
            totals["warned"] += rollup.warned
            totals["blocked"] += rollup.blocked
            totals["errored"] += rollup.errored
        return cls(
            generated_at=report.generated_at,
            overall_status=report.overall_status,
            summary=report.summary,
            **totals,
        )


class SuiteSummary(BaseModel):
    """One row in `suite-list`: a registered suite's filename-stem key + case count. A
    malformed registry file is surfaced as valid=False with an error, never a crash."""

    name: str  # the registry key = the filename stem
    case_count: int = 0
    valid: bool = True
    error: str | None = None
