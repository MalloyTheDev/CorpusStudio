"""Evaluation orchestration for local model testing."""

from collections.abc import Callable

from pydantic import BaseModel, Field

from corpus_studio.evaluation.reports import (
    EvaluationExampleResult,
    EvaluationReport,
    EvaluationRunSettings,
)
from corpus_studio.evaluation.scoring import score_text_overlap
from corpus_studio.evaluation.scorers import KeywordOverlapScorer, Scorer
from corpus_studio.model_backends.base import BackendGenerateRequest, ModelBackend
from corpus_studio.model_backends.retry import BACKEND_ERROR_TYPES, format_backend_error


class EvaluationRunConfig(BaseModel):
    """Configuration for one local evaluation run."""

    dataset: str
    model: str
    schema_id: str
    dataset_path: str | None = None
    backend: str = "unknown"
    base_url: str | None = None
    limit: int | None = None
    score_threshold: float = 70.0
    timeout_seconds: int = 120
    tags: list[str] = Field(default_factory=list)
    # Reasoning/trace-aware eval: when the model emits <think>…</think>answer, score the ANSWER only —
    # its reasoning is not the reference and would corrupt the score. Off by default.
    reasoning: bool = False

    def to_report_settings(self) -> EvaluationRunSettings:
        """Return the repeatable settings stored with an evaluation report."""

        return EvaluationRunSettings(
            dataset_path=self.dataset_path,
            schema_id=self.schema_id,
            backend=self.backend,
            base_url=self.base_url,
            model=self.model,
            limit=self.limit,
            score_threshold=self.score_threshold,
            timeout_seconds=self.timeout_seconds,
        )


class EvaluationDatasetExample(BaseModel):
    """Normalized evaluation example extracted from a dataset row."""

    example_id: str
    prompt: str
    expected_output: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def build_report_from_outputs(
    config: EvaluationRunConfig,
    outputs: list[tuple[str, str, str]],
) -> EvaluationReport:
    """Build a report from already-produced model outputs.

    Args:
        config: Evaluation run metadata.
        outputs: Tuples of ``(prompt, expected_output, model_output)``.

    TODO: Replace this helper with backend-driven execution once v0.2 model
    backends are wired.
    """

    results: list[EvaluationExampleResult] = []
    for index, (prompt, expected_output, model_output) in enumerate(outputs, start=1):
        score = score_text_overlap(expected_output, model_output)
        results.append(
            EvaluationExampleResult(
                example_id=f"example-{index}",
                prompt=prompt,
                expected_output=expected_output,
                model_output=model_output,
                score=score,
                passed=score >= config.score_threshold,
                tags=config.tags,
                notes=_score_failure_note(score, config.score_threshold),
            )
        )

    return EvaluationReport.from_results(
        dataset=config.dataset,
        model=config.model,
        results=results,
        run_settings=config.to_report_settings(),
    )


def extract_evaluation_examples(
    rows: list[dict],
    schema_id: str,
) -> list[EvaluationDatasetExample]:
    """Extract instruction/chat rows into Evaluation Lab examples."""

    if schema_id == "instruction":
        return [
            EvaluationDatasetExample(
                example_id=f"row-{index}",
                prompt=_instruction_prompt(row),
                expected_output=str(row.get("output", "")),
                tags=_coerce_tags(row.get("tags")),
            )
            for index, row in enumerate(rows, start=1)
        ]

    if schema_id == "chat":
        return [
            example
            for index, row in enumerate(rows, start=1)
            if (example := _chat_example(index, row)) is not None
        ]

    raise ValueError("Evaluation Lab MVP supports instruction and chat schemas.")


def should_report_progress(completed: int, total: int) -> bool:
    """Throttle per-example progress to at most ~100 updates (always the first and the
    last), so a large run streams a readable trickle instead of one line per example.

    A progress *callback* still fires for every example (that is the caller's signal);
    this only decides when a human-facing *sink* should actually print, so a 10k-row
    eval doesn't flood a terminal or log file."""
    if completed <= 1 or completed >= total:
        return True
    step = max(1, total // 100)
    return completed % step == 0


def _evaluate_example(
    example: EvaluationDatasetExample,
    backend: ModelBackend,
    active_scorer: Scorer,
    config: EvaluationRunConfig,
) -> EvaluationExampleResult:
    """Evaluate one example, isolating any backend/scorer failure into a scored-0
    result (never raising) so one bad row can't discard the whole run."""
    try:
        response = backend.generate(
            BackendGenerateRequest(
                prompt=example.prompt if not example.messages else None,
                messages=example.messages,
            )
        )
    except BACKEND_ERROR_TYPES as exc:
        # Isolate one example's backend failure: record it as a scored-0
        # failure so the run finishes and the rest of the dataset is scored.
        return EvaluationExampleResult(
            example_id=example.example_id,
            prompt=example.prompt,
            expected_output=example.expected_output,
            model_output="",
            score=0.0,
            passed=False,
            tags=example.tags or config.tags,
            notes="backend_error",
            error=format_backend_error(exc),
        )

    # Trace-aware eval: score the ANSWER only (strip <think>…</think>) so the reference is not compared
    # against the model's reasoning, which would corrupt the score. `had_reasoning=False` in this mode
    # flags a "reasoning" model that emitted no reasoning.
    scoring_output = response.text
    reasoning_missing = False
    if config.reasoning:
        from corpus_studio.training.traces import answer_for_scoring  # noqa: PLC0415

        scoring_output, had_reasoning = answer_for_scoring(response.text)
        reasoning_missing = not had_reasoning

    try:
        scored = active_scorer.score(example.prompt, example.expected_output, scoring_output)
    except BACKEND_ERROR_TYPES as exc:
        # BACKEND_ERROR_TYPES = (OSError, ValueError), so this also covers JSON decode
        # errors (json.JSONDecodeError subclasses ValueError) from a judge response.
        # Isolate a scorer failure (e.g. an LLM-judge backend outage or an unparseable
        # judge response) to THIS row: the model output already succeeded, so keep it,
        # record the row as a scored-0 failure, and finish the rest of the run — one bad
        # judge call must not discard the whole evaluation. Mirrors the backend-error path.
        return EvaluationExampleResult(
            example_id=example.example_id,
            prompt=example.prompt,
            expected_output=example.expected_output,
            model_output=response.text,
            score=0.0,
            passed=False,
            tags=example.tags or config.tags,
            notes="scorer_error",
            error=format_backend_error(exc),
        )

    failure_note = _score_failure_note(scored.score, config.score_threshold)
    notes = (
        ("no_reasoning" + (f"; {failure_note}" if failure_note else ""))
        if reasoning_missing
        else failure_note
    )
    return EvaluationExampleResult(
        example_id=example.example_id,
        prompt=example.prompt,
        expected_output=example.expected_output,
        model_output=response.text,
        score=scored.score,
        passed=scored.score >= config.score_threshold,
        tags=example.tags or config.tags,
        notes=notes,
        rationale=scored.rationale,
    )


def run_evaluation(
    config: EvaluationRunConfig,
    examples: list[EvaluationDatasetExample],
    backend: ModelBackend,
    limit: int | None = None,
    scorer: Scorer | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationReport:
    """Run examples through a backend and return a JSON-serializable report.

    ``scorer`` selects the automatic metric; it defaults to keyword-overlap recall (offline,
    no judge). Pass an ``LlmJudgeScorer`` to score with an evaluator model instead.

    ``progress_callback`` (optional) is invoked ``(completed, total)`` after each example is
    scored — for a live progress bar / streamed count on a long run. It is best-effort: a
    callback that raises must not abort the evaluation, so its errors are swallowed.
    """

    active_scorer: Scorer = scorer if scorer is not None else KeywordOverlapScorer()
    selected_examples = examples[:limit] if limit is not None else examples
    total = len(selected_examples)
    results: list[EvaluationExampleResult] = []
    for example in selected_examples:
        results.append(_evaluate_example(example, backend, active_scorer, config))
        if progress_callback is not None:
            try:
                progress_callback(len(results), total)
            except Exception:  # noqa: BLE001 - a progress sink must never break the run
                pass

    return EvaluationReport.from_results(
        dataset=config.dataset,
        model=config.model,
        results=results,
        run_settings=config.to_report_settings(),
        metric=active_scorer.metric,
    )


def _instruction_prompt(row: dict) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if not input_text:
        return instruction

    return f"{instruction}\n\nInput:\n{input_text}"


def _chat_example(index: int, row: dict) -> EvaluationDatasetExample | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    normalized_messages = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict)
    ]
    expected_index = next(
        (
            message_index
            for message_index in range(len(normalized_messages) - 1, -1, -1)
            if normalized_messages[message_index]["role"] == "assistant"
        ),
        -1,
    )
    if expected_index < 0:
        return None

    request_messages = normalized_messages[:expected_index]
    expected_output = normalized_messages[expected_index]["content"]
    prompt = "\n".join(
        f"{message['role']}: {message['content']}" for message in request_messages
    )
    return EvaluationDatasetExample(
        example_id=f"row-{index}",
        prompt=prompt,
        messages=request_messages,
        expected_output=expected_output,
        tags=_coerce_tags(row.get("tags")),
    )


def _coerce_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    return [str(item) for item in value if str(item).strip()]


def _score_failure_note(score: float, score_threshold: float) -> str | None:
    if score >= score_threshold:
        return None

    return "score_below_threshold"
