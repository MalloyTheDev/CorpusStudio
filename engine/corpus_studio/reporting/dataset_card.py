"""Dataset card builder.

A dataset card aggregates the artifacts Corpus Studio already produces for a
project -- metadata, schema, example counts, quality checks, split sizes, and the
latest evaluation summary -- into a single inspectable Markdown/JSON document.
Building a card never mutates the project; it only reads existing files.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, computed_field

from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.quality.basic_quality import QualityReport, build_basic_quality_report
from corpus_studio.schemas.base import DatasetSchema


class DatasetCardField(BaseModel):
    """Schema field described on the card."""

    name: str
    type: str
    required: bool = False
    description: str | None = None


class DatasetCardSplits(BaseModel):
    """Row counts for generated train/validation/test splits."""

    train: int = 0
    validation: int = 0
    test: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return self.train + self.validation + self.test


class DatasetCardEvaluation(BaseModel):
    """Condensed summary of the most recent evaluation run."""

    model: str
    dataset: str
    examples_tested: int
    average_score: float
    failed_examples: int
    weak_tags: list[str] = Field(default_factory=list)
    source_report: str | None = None

    @classmethod
    def from_report(
        cls,
        report: EvaluationReport,
        source_report: str | None = None,
    ) -> "DatasetCardEvaluation":
        return cls(
            model=report.model,
            dataset=report.dataset,
            examples_tested=report.examples_tested,
            average_score=report.average_score,
            failed_examples=report.failed_examples,
            weak_tags=list(report.weak_tags),
            source_report=source_report,
        )


class DatasetCard(BaseModel):
    """Serializable dataset card summary."""

    project_id: str
    project_name: str
    schema_id: str
    schema_name: str
    schema_version: str
    created_at: str | None = None
    updated_at: str | None = None
    generated_at: str
    example_count: int
    fields: list[DatasetCardField] = Field(default_factory=list)
    quality: QualityReport
    splits: DatasetCardSplits | None = None
    evaluation: DatasetCardEvaluation | None = None
    warnings: list[str] = Field(default_factory=list)


def build_dataset_card(
    *,
    project_id: str,
    project_name: str,
    schema: DatasetSchema,
    rows: list[dict],
    created_at: str | None = None,
    updated_at: str | None = None,
    generated_at: str | None = None,
    splits: DatasetCardSplits | None = None,
    evaluation: DatasetCardEvaluation | None = None,
) -> DatasetCard:
    """Assemble a dataset card from already-loaded project artifacts."""

    quality = build_basic_quality_report(rows)
    card = DatasetCard(
        project_id=project_id,
        project_name=project_name,
        schema_id=schema.id,
        schema_name=schema.name,
        schema_version=schema.version,
        created_at=created_at,
        updated_at=updated_at,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        example_count=len(rows),
        fields=[
            DatasetCardField(
                name=field.name,
                type=field.type,
                required=field.required,
                description=field.description,
            )
            for field in schema.fields
        ],
        quality=quality,
        splits=splits,
        evaluation=evaluation,
    )
    card.warnings = _build_card_warnings(card)
    return card


def _build_card_warnings(card: DatasetCard) -> list[str]:
    warnings: list[str] = []

    if card.example_count == 0:
        warnings.append("Dataset has no examples yet; add or import rows before exporting.")

    quality = card.quality
    if quality.empty_row_count:
        warnings.append(f"{quality.empty_row_count} empty row(s) detected.")
    if quality.duplicate_exact_count:
        warnings.append(f"{quality.duplicate_exact_count} exact duplicate row(s) detected.")
    if quality.duplicate_normalized_count:
        warnings.append(
            f"{quality.duplicate_normalized_count} near-duplicate row(s) detected "
            "(normalized text match)."
        )
    if quality.low_information_count:
        warnings.append(
            f"{quality.low_information_count} low-information row(s) below "
            f"{quality.low_information_token_threshold} tokens."
        )
    if quality.synthetic_pattern_count:
        warnings.append(
            f"{quality.synthetic_pattern_count} synthetic-pattern issue(s) flagged; "
            "review before training."
        )

    if card.splits is None:
        warnings.append("No train/validation/test splits have been generated yet.")
    else:
        if card.splits.validation == 0:
            warnings.append("Validation split has no rows.")
        if card.splits.test == 0:
            warnings.append("Test split has no rows.")

    if card.evaluation is None:
        warnings.append("No evaluation run has been recorded for this dataset yet.")

    return warnings


def render_dataset_card_markdown(card: DatasetCard) -> str:
    """Render a dataset card as inspectable Markdown."""

    lines: list[str] = []
    lines.append(f"# Dataset Card: {card.project_name}")
    lines.append("")
    lines.append(
        "_Generated by Corpus Studio. Local-first summary; this dataset is not "
        "automatically published or uploaded._"
    )
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Project ID | `{card.project_id}` |")
    lines.append(f"| Schema | {card.schema_name} (`{card.schema_id}` v{card.schema_version}) |")
    lines.append(f"| Examples | {card.example_count} |")
    lines.append(f"| Created | {card.created_at or 'unknown'} |")
    lines.append(f"| Updated | {card.updated_at or 'unknown'} |")
    lines.append(f"| Card generated | {card.generated_at} |")
    lines.append("")

    lines.append("## Schema Fields")
    lines.append("")
    if card.fields:
        lines.append("| Field | Type | Required | Description |")
        lines.append("| --- | --- | --- | --- |")
        for field in card.fields:
            required = "yes" if field.required else "no"
            description = (field.description or "").replace("|", "\\|")
            lines.append(f"| `{field.name}` | {field.type} | {required} | {description} |")
    else:
        lines.append("_No fields are defined for this schema._")
    lines.append("")

    lines.append("## Quality Summary")
    lines.append("")
    quality = card.quality
    lines.append("| Check | Count |")
    lines.append("| --- | --- |")
    lines.append(f"| Examples | {quality.example_count} |")
    lines.append(f"| Empty rows | {quality.empty_row_count} |")
    lines.append(f"| Exact duplicates | {quality.duplicate_exact_count} |")
    lines.append(f"| Near duplicates | {quality.duplicate_normalized_count} |")
    lines.append(
        f"| Low-information rows (< {quality.low_information_token_threshold} tokens) "
        f"| {quality.low_information_count} |"
    )
    lines.append(f"| Synthetic-pattern issues | {quality.synthetic_pattern_count} |")
    lines.append("")

    lines.append("## Splits")
    lines.append("")
    if card.splits is not None:
        splits = card.splits
        lines.append("| Split | Rows |")
        lines.append("| --- | --- |")
        lines.append(f"| Train | {splits.train} |")
        lines.append(f"| Validation | {splits.validation} |")
        lines.append(f"| Test | {splits.test} |")
        lines.append(f"| Total | {splits.total} |")
    else:
        lines.append("_No train/validation/test splits have been generated yet._")
    lines.append("")

    lines.append("## Evaluation")
    lines.append("")
    if card.evaluation is not None:
        evaluation = card.evaluation
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Model | {evaluation.model} |")
        lines.append(f"| Dataset | {evaluation.dataset} |")
        lines.append(f"| Examples tested | {evaluation.examples_tested} |")
        lines.append(f"| Average score | {evaluation.average_score} |")
        lines.append(f"| Failed examples | {evaluation.failed_examples} |")
        weak_tags = ", ".join(evaluation.weak_tags) if evaluation.weak_tags else "none"
        lines.append(f"| Weak tags | {weak_tags} |")
    else:
        lines.append("_No evaluation run has been recorded for this dataset yet._")
    lines.append("")

    lines.append("## Notes & Warnings")
    lines.append("")
    if card.warnings:
        for warning in card.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No outstanding warnings.")
    lines.append("")

    return "\n".join(lines)
