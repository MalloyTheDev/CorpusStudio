import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.reporting.dataset_card import (
    DatasetCardEvaluation,
    DatasetCardSplits,
    build_dataset_card,
    render_dataset_card_markdown,
)
from corpus_studio.schemas.registry import load_builtin_schema


runner = CliRunner()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_project(root: Path, schema_id: str = "instruction") -> Path:
    project_dir = root / "demo_project"
    project_dir.mkdir(parents=True)
    (project_dir / "project.json").write_text(
        json.dumps(
            {
                "id": "demo_project",
                "name": "Demo Project",
                "schema_id": schema_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-02-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return project_dir


def test_build_dataset_card_summarizes_quality_and_warnings():
    schema = load_builtin_schema("instruction")
    rows = [
        {"instruction": f"Explain concept {index}.", "output": f"Answer {index} here."}
        for index in range(4)
    ]

    card = build_dataset_card(
        project_id="demo_project",
        project_name="Demo Project",
        schema=schema,
        rows=rows,
    )

    assert card.example_count == 4
    assert card.schema_id == "instruction"
    assert {field.name for field in card.fields} >= {"instruction", "output"}
    # No splits and no evaluation yet -> both should be flagged.
    assert any("splits" in warning for warning in card.warnings)
    assert any("evaluation" in warning for warning in card.warnings)


def test_build_dataset_card_includes_splits_and_evaluation():
    schema = load_builtin_schema("instruction")
    rows = [{"instruction": "Explain variables.", "output": "A named value."}]
    splits = DatasetCardSplits(train=8, validation=1, test=1)
    evaluation = DatasetCardEvaluation(
        model="llama3",
        dataset="examples",
        examples_tested=10,
        average_score=82.5,
        failed_examples=2,
        weak_tags=["math"],
    )

    card = build_dataset_card(
        project_id="demo_project",
        project_name="Demo Project",
        schema=schema,
        rows=rows,
        splits=splits,
        evaluation=evaluation,
    )
    markdown = render_dataset_card_markdown(card)

    assert card.splits is not None and card.splits.total == 10
    assert card.evaluation is not None and card.evaluation.average_score == 82.5
    assert "# Dataset Card: Demo Project" in markdown
    assert "llama3" in markdown
    assert not any("evaluation run has been recorded" in warning for warning in card.warnings)


def test_dataset_card_command_writes_markdown(tmp_path: Path):
    project_dir = make_project(tmp_path)
    write_rows(
        project_dir / "examples.jsonl",
        [
            {"instruction": f"Explain item {index}.", "output": f"Item {index} answer."}
            for index in range(6)
        ],
    )

    export_dir = tmp_path / "exports" / "demo_project"
    split_dir = export_dir / "splits"
    split_dir.mkdir(parents=True)
    write_rows(split_dir / "train.jsonl", [{"instruction": "a", "output": "b"}] * 4)
    write_rows(split_dir / "validation.jsonl", [{"instruction": "a", "output": "b"}])
    write_rows(split_dir / "test.jsonl", [{"instruction": "a", "output": "b"}])

    evaluation_dir = export_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "20260101000000_evaluation_report.json").write_text(
        json.dumps(
            {
                "dataset": "examples",
                "model": "llama3",
                "examples_tested": 6,
                "average_score": 77.0,
                "failed_examples": 1,
                "weak_tags": ["logic"],
            }
        ),
        encoding="utf-8",
    )

    card_path = tmp_path / "dataset_card.md"
    result = runner.invoke(
        app,
        [
            "dataset-card",
            str(project_dir),
            "--output-path",
            str(card_path),
            "--export-dir",
            str(export_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert card_path.exists()
    markdown = card_path.read_text(encoding="utf-8")
    assert "Demo Project" in markdown
    assert "llama3" in markdown

    payload = json.loads(result.output)
    assert payload["output_path"] == str(card_path)
    assert payload["card"]["splits"]["total"] == 6
    assert payload["card"]["evaluation"]["model"] == "llama3"


def test_dataset_card_command_rejects_missing_project(tmp_path: Path):
    result = runner.invoke(app, ["dataset-card", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "Project metadata was not found" in result.output
