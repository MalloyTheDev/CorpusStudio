"""Multi-model benchmark comparison.

Runs one dataset across several models (each producing an ``EvaluationReport``)
and ranks them, exposing per-model deltas versus the best model and the set of
examples every model failed (systematically hard or mislabeled rows). The
per-model runs reuse the ordinary evaluation path; this module only compares
the resulting reports, so the comparison logic is pure and testable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from corpus_studio.evaluation.reports import EvaluationReport


class BenchmarkModelSummary(BaseModel):
    model: str
    examples_tested: int
    average_score: float
    pass_rate: float
    failed_examples: int
    average_manual_score: float | None = None
    rank: int
    score_delta_vs_best: float


class BenchmarkReport(BaseModel):
    dataset: str
    model_count: int = 0
    examples_tested: int = 0
    best_model: str = ""
    worst_model: str = ""
    score_spread: float = 0.0
    models: list[BenchmarkModelSummary] = Field(default_factory=list)
    commonly_failed_examples: list[str] = Field(default_factory=list)


def _pass_rate(report: EvaluationReport) -> float:
    if report.examples_tested == 0:
        return 0.0
    passed = report.examples_tested - report.failed_examples
    return round(passed / report.examples_tested * 100, 2)


def build_benchmark_report(dataset: str, reports: list[EvaluationReport]) -> BenchmarkReport:
    if not reports:
        return BenchmarkReport(dataset=dataset)

    ranked = sorted(
        reports,
        key=lambda report: (-report.average_score, -_pass_rate(report), report.model),
    )
    best_average = ranked[0].average_score

    summaries = [
        BenchmarkModelSummary(
            model=report.model,
            examples_tested=report.examples_tested,
            average_score=report.average_score,
            pass_rate=_pass_rate(report),
            failed_examples=report.failed_examples,
            average_manual_score=report.average_manual_score,
            rank=index + 1,
            score_delta_vs_best=round(report.average_score - best_average, 2),
        )
        for index, report in enumerate(ranked)
    ]

    failed_sets = [
        {result.example_id for result in report.results if not result.passed}
        for report in reports
    ]
    commonly_failed = sorted(set.intersection(*failed_sets)) if failed_sets else []

    return BenchmarkReport(
        dataset=dataset,
        model_count=len(reports),
        examples_tested=max((report.examples_tested for report in reports), default=0),
        best_model=ranked[0].model,
        worst_model=ranked[-1].model,
        score_spread=round(ranked[0].average_score - ranked[-1].average_score, 2),
        models=summaries,
        commonly_failed_examples=commonly_failed,
    )
