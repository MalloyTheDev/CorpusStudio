"""Report models for the planned Evaluation Lab."""

from pydantic import BaseModel, Field


class EvaluationExampleResult(BaseModel):
    """Per-example result captured during an evaluation run."""

    example_id: str
    prompt: str
    expected_output: str
    model_output: str
    score: float
    passed: bool
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    manual_score: float | None = None
    manual_notes: str | None = None


class EvaluationTagSummary(BaseModel):
    """Aggregate Evaluation Lab scores and failures for a dataset tag."""

    tag: str
    examples: int
    failed_examples: int
    average_score: float


class EvaluationFailureReasonSummary(BaseModel):
    """Aggregate failed examples by failure note or automatic reason."""

    reason: str
    failed_examples: int


class EvaluationScoreBandSummary(BaseModel):
    """Aggregate Evaluation Lab results by score band."""

    band: str
    examples: int
    failed_examples: int
    average_score: float


class EvaluationRunSettings(BaseModel):
    """Repeatable settings captured for a saved evaluation run."""

    dataset_path: str | None = None
    schema_id: str
    backend: str
    base_url: str | None = None
    model: str
    limit: int | None = None
    score_threshold: float = 70.0
    timeout_seconds: int = 120


class EvaluationReport(BaseModel):
    """Serializable summary for a dataset/model evaluation run."""

    dataset: str
    model: str
    # Which scorer produced the automatic `score`/`average_score` values. The default
    # "keyword_overlap" is a lexical recall heuristic (not a semantic/quality judgment);
    # "llm_judge" is the opt-in judge-model scorer. Surfaced so reports, gates, and the UI
    # never present the number as a quality score without saying what it measures.
    metric: str = "keyword_overlap"
    examples_tested: int
    average_score: float
    failed_examples: int
    weak_tags: list[str] = Field(default_factory=list)
    tag_summary: list[EvaluationTagSummary] = Field(default_factory=list)
    failure_reason_summary: list[EvaluationFailureReasonSummary] = Field(
        default_factory=list
    )
    score_band_summary: list[EvaluationScoreBandSummary] = Field(default_factory=list)
    manually_scored_examples: int = 0
    average_manual_score: float | None = None
    run_settings: EvaluationRunSettings | None = None
    results: list[EvaluationExampleResult] = Field(default_factory=list)

    @classmethod
    def from_results(
        cls,
        dataset: str,
        model: str,
        results: list[EvaluationExampleResult],
        run_settings: EvaluationRunSettings | None = None,
        metric: str = "keyword_overlap",
    ) -> "EvaluationReport":
        """Create a report summary from per-example results."""

        examples_tested = len(results)
        average_score = (
            round(sum(result.score for result in results) / examples_tested, 2)
            if examples_tested
            else 0.0
        )
        failed_results = [result for result in results if not result.passed]
        weak_tags = sorted({tag for result in failed_results for tag in result.tags})
        manual_scores = [
            result.manual_score for result in results if result.manual_score is not None
        ]

        return cls(
            dataset=dataset,
            model=model,
            metric=metric,
            examples_tested=examples_tested,
            average_score=average_score,
            failed_examples=len(failed_results),
            weak_tags=weak_tags,
            tag_summary=_build_tag_summary(results),
            failure_reason_summary=_build_failure_reason_summary(failed_results),
            score_band_summary=_build_score_band_summary(results),
            manually_scored_examples=len(manual_scores),
            average_manual_score=round(sum(manual_scores) / len(manual_scores), 2)
            if manual_scores
            else None,
            run_settings=run_settings,
            results=results,
        )


def _build_tag_summary(
    results: list[EvaluationExampleResult],
) -> list[EvaluationTagSummary]:
    grouped: dict[str, list[EvaluationExampleResult]] = {}
    for result in results:
        tags = _normalized_tags(result.tags)
        for tag in tags:
            grouped.setdefault(tag, []).append(result)

    return [
        EvaluationTagSummary(
            tag=tag,
            examples=len(tag_results),
            failed_examples=sum(1 for result in tag_results if not result.passed),
            average_score=_average_score(tag_results),
        )
        for tag, tag_results in sorted(
            grouped.items(),
            key=lambda item: (
                -sum(1 for result in item[1] if not result.passed),
                item[0].lower(),
            ),
        )
    ]


def _build_failure_reason_summary(
    failed_results: list[EvaluationExampleResult],
) -> list[EvaluationFailureReasonSummary]:
    grouped: dict[str, int] = {}
    for result in failed_results:
        reason = _failure_reason(result)
        grouped[reason] = grouped.get(reason, 0) + 1

    return [
        EvaluationFailureReasonSummary(reason=reason, failed_examples=count)
        for reason, count in sorted(
            grouped.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
    ]


def _build_score_band_summary(
    results: list[EvaluationExampleResult],
) -> list[EvaluationScoreBandSummary]:
    grouped: dict[str, list[EvaluationExampleResult]] = {}
    for result in results:
        grouped.setdefault(_score_band(result.score), []).append(result)

    return [
        EvaluationScoreBandSummary(
            band=band,
            examples=len(band_results),
            failed_examples=sum(1 for result in band_results if not result.passed),
            average_score=_average_score(band_results),
        )
        for band, band_results in sorted(
            grouped.items(),
            key=lambda item: _score_band_sort_key(item[0]),
        )
    ]


def _normalized_tags(tags: list[str]) -> list[str]:
    normalized = sorted({tag.strip() for tag in tags if tag.strip()})
    return normalized or ["untagged"]


def _failure_reason(result: EvaluationExampleResult) -> str:
    if result.notes and result.notes.strip():
        return result.notes.strip()

    return "score_below_threshold"


def _score_band(score: float) -> str:
    if score < 50:
        return "0-49"
    if score < 70:
        return "50-69"
    if score < 85:
        return "70-84"
    return "85-100"


def _score_band_sort_key(band: str) -> int:
    order = {
        "0-49": 0,
        "50-69": 1,
        "70-84": 2,
        "85-100": 3,
    }
    return order.get(band, 99)


def _average_score(results: list[EvaluationExampleResult]) -> float:
    if not results:
        return 0.0

    return round(sum(result.score for result in results) / len(results), 2)
