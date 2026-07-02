"""Arena prompt/response/judgment/report models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArenaPrompt(BaseModel):
    id: str
    prompt: str
    system: str | None = None


class ArenaResponse(BaseModel):
    prompt_id: str
    model: str
    text: str


class ArenaJudgment(BaseModel):
    """One evaluator judgment over the candidate responses for a prompt."""

    prompt_id: str
    winner: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    parsed: bool = True


class ArenaModelSummary(BaseModel):
    model: str
    response_count: int
    empty_response_count: int
    win_count: int = 0
    average_judge_score: float | None = None


class ArenaReport(BaseModel):
    prompt_count: int = 0
    models: list[str] = Field(default_factory=list)
    prompts: list[ArenaPrompt] = Field(default_factory=list)
    responses: list[ArenaResponse] = Field(default_factory=list)
    model_summaries: list[ArenaModelSummary] = Field(default_factory=list)
    judge_model: str | None = None
    judgments: list[ArenaJudgment] = Field(default_factory=list)
    generated_at: str | None = None


def build_model_summaries(
    models: list[str],
    responses: list[ArenaResponse],
    judgments: list[ArenaJudgment] | None = None,
) -> list[ArenaModelSummary]:
    """Per-model response counts, plus win/score aggregates when judged."""

    judgments = judgments or []
    summaries: list[ArenaModelSummary] = []
    for model in models:
        model_scores = [
            judgment.scores[model]
            for judgment in judgments
            if model in judgment.scores
        ]
        summaries.append(
            ArenaModelSummary(
                model=model,
                response_count=sum(1 for r in responses if r.model == model),
                empty_response_count=sum(
                    1 for r in responses if r.model == model and not r.text.strip()
                ),
                win_count=sum(1 for judgment in judgments if judgment.winner == model),
                average_judge_score=round(sum(model_scores) / len(model_scores), 2)
                if model_scores
                else None,
            )
        )
    return summaries


def build_arena_report(
    prompts: list[ArenaPrompt],
    models: list[str],
    responses: list[ArenaResponse],
    generated_at: str | None = None,
) -> ArenaReport:
    """Assemble responses into a report with per-model summaries (unjudged)."""

    return ArenaReport(
        prompt_count=len(prompts),
        models=list(models),
        prompts=list(prompts),
        responses=responses,
        model_summaries=build_model_summaries(models, responses),
        generated_at=generated_at,
    )
