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
    dataset_path: str
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
    def _judge_required_for_llm_judge(self) -> "SuiteCase":
        if self.metric == "llm_judge" and not self.judge_model:
            raise ValueError(f"Case '{self.name}': metric 'llm_judge' requires a judge_model.")
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
