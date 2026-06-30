import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app


runner = CliRunner()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_schemas_command_lists_builtin_schemas():
    result = runner.invoke(app, ["schemas"])

    assert result.exit_code == 0
    schemas = json.loads(result.output)
    assert "instruction" in {schema["id"] for schema in schemas}


def test_new_project_command_creates_project_files(tmp_path: Path):
    result = runner.invoke(
        app,
        ["new-project", "demo_project", "Demo Project", "instruction", "--root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert (tmp_path / "demo_project" / "project.json").exists()
    assert (tmp_path / "demo_project" / "examples.jsonl").exists()


def test_export_command_rejects_invalid_rows(tmp_path: Path):
    input_path = tmp_path / "invalid.jsonl"
    output_path = tmp_path / "export.jsonl"
    write_rows(input_path, [{"instruction": "Explain variables."}])

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 1
    assert "Missing required field: output" in result.output
    assert not output_path.exists()


def test_export_command_rejects_wrong_field_type(tmp_path: Path):
    input_path = tmp_path / "invalid_type.jsonl"
    output_path = tmp_path / "export.jsonl"
    write_rows(
        input_path,
        [{"instruction": "Explain variables.", "output": "A value.", "tags": "bad"}],
    )

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 1
    assert "Expected list." in result.output
    assert not output_path.exists()


def test_split_command_writes_train_validation_and_test_files(tmp_path: Path):
    input_path = tmp_path / "instruction.jsonl"
    output_dir = tmp_path / "splits"
    write_rows(
        input_path,
        [
            {"instruction": f"Explain item {index}.", "output": f"Item {index} explanation."}
            for index in range(20)
        ],
    )

    result = runner.invoke(app, ["split", str(input_path), str(output_dir), "instruction"])

    assert result.exit_code == 0
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()


def test_quality_command_reports_duplicates(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    duplicate_row = {"instruction": "Explain variables.", "output": "A variable stores a value."}
    write_rows(input_path, [duplicate_row, duplicate_row])

    result = runner.invoke(app, ["quality", str(input_path)])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["example_count"] == 2
    assert report["duplicate_exact_count"] == 1


def test_import_preview_command_reports_failed_rows(tmp_path: Path):
    input_path = tmp_path / "mixed.jsonl"
    write_rows(
        input_path,
        [
            {"instruction": "Explain variables.", "output": "A variable stores a value."},
            {"instruction": "Missing output."},
        ],
    )

    result = runner.invoke(app, ["import-preview", str(input_path), "instruction"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["valid"] is False
    assert report["accepted_rows"] == 1
    assert report["rejected_rows"] == 1
    assert report["failed_rows"][0]["row_number"] == 2
