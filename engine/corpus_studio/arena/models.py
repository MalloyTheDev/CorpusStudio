"""Arena prompt/response/report models."""

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


class ArenaModelSummary(BaseModel):
    model: str
    response_count: int
    empty_response_count: int


class ArenaReport(BaseModel):
    prompt_count: int = 0
    models: list[str] = Field(default_factory=list)
    responses: list[ArenaResponse] = Field(default_factory=list)
    model_summaries: list[ArenaModelSummary] = Field(default_factory=list)
    generated_at: str | None = None


def build_arena_report(
    prompts: list[ArenaPrompt],
    models: list[str],
    responses: list[ArenaResponse],
    generated_at: str | None = None,
) -> ArenaReport:
    """Assemble responses into a report with per-model summaries."""

    summaries = [
        ArenaModelSummary(
            model=model,
            response_count=sum(1 for r in responses if r.model == model),
            empty_response_count=sum(
                1 for r in responses if r.model == model and not r.text.strip()
            ),
        )
        for model in models
    ]
    return ArenaReport(
        prompt_count=len(prompts),
        models=list(models),
        responses=responses,
        model_summaries=summaries,
        generated_at=generated_at,
    )
