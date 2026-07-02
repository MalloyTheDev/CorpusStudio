"""Weight card — a live projection of a model artifact (never stored).

Rendered on demand from the artifact record + its source run + eval reports, so
it can never drift from the underlying state. It carries the v0.8.1 provenance
caveat: if the after-eval targeted the base model (or its target was not
recorded), the before/after numbers are labelled unverified rather than
presented as a confident improvement.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WeightCard(BaseModel):
    artifact_id: str
    run_id: str
    kind: str
    status: str
    path: str
    integrity: str
    base_model: str = ""
    config_path: str = ""
    checkpoints: list[str] = Field(default_factory=list)
    before_score: float | None = None
    after_score: float | None = None
    delta: float | None = None
    provenance_note: str = ""
    created_at: str = ""
    updated_at: str = ""


def build_weight_card(
    artifact: Any,
    run: Any,
    before_report: Any,
    after_report: Any,
    integrity: str,
) -> WeightCard:
    """Assemble a weight card. ``run``/reports may be None (resolved live)."""

    base_model = getattr(run, "base_model", "") if run else ""
    config_path = getattr(run, "config_path", "") if run else ""
    checkpoints = list(getattr(run, "checkpoints", []) or []) if run else []

    before_score = before_report.average_score if before_report is not None else None
    after_score = after_report.average_score if after_report is not None else None
    delta = (
        round(after_score - before_score, 2)
        if before_score is not None and after_score is not None
        else None
    )

    provenance_note = ""
    if after_report is not None:
        after_model = getattr(run, "after_eval_model", None) if run else None
        if not after_model:
            provenance_note = (
                "Unverified linkage: the after-eval's target model was not recorded; "
                "treat the before/after numbers with caution."
            )
        elif base_model and after_model == base_model:
            provenance_note = (
                "Unverified linkage: the after-eval appears to target the base model, not the "
                "trained adapter; the before/after numbers are not trustworthy."
            )

    return WeightCard(
        artifact_id=artifact.artifact_id,
        run_id=artifact.run_id,
        kind=artifact.kind,
        status=artifact.status,
        path=artifact.path,
        integrity=integrity,
        base_model=base_model,
        config_path=config_path,
        checkpoints=checkpoints,
        before_score=before_score,
        after_score=after_score,
        delta=delta,
        provenance_note=provenance_note,
        created_at=getattr(artifact, "created_at", ""),
        updated_at=getattr(artifact, "updated_at", ""),
    )


def _score(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


def render_weight_card_markdown(card: WeightCard) -> str:
    lines = [
        f"# Weight Card — {card.artifact_id}",
        "",
        f"- **Kind**: {card.kind}",
        f"- **Status**: {card.status}",
        f"- **Integrity**: {card.integrity}",
        f"- **Path**: {card.path}",
        f"- **Source run**: {card.run_id}",
        f"- **Base model**: {card.base_model or '(unknown — source run not found)'}",
    ]
    if card.config_path:
        lines.append(f"- **Config**: {card.config_path}")
    if card.checkpoints:
        lines.append(f"- **Checkpoints**: {len(card.checkpoints)} ({', '.join(card.checkpoints[:4])}{' …' if len(card.checkpoints) > 4 else ''})")

    lines += [
        "",
        "## Evaluation (before → after)",
        "",
        f"- Base: {_score(card.before_score)}",
        f"- Trained: {_score(card.after_score)}"
        + (f" (Δ{card.delta:+.1f})" if card.delta is not None else ""),
    ]

    if card.integrity != "ok":
        lines += ["", f"> ⚠ Integrity is **{card.integrity}**: the weights changed or are gone since evaluation. These numbers may not describe the weights at this path."]
    if card.provenance_note:
        lines += ["", f"> ⚠ {card.provenance_note}"]

    lines += ["", f"_Registered {card.created_at}, updated {card.updated_at}._"]
    return "\n".join(lines)
