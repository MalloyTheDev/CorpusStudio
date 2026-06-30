from pathlib import Path

from corpus_studio.validators.basic_validator import validate_jsonl_file


def test_instruction_validation_accepts_valid_row(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"instruction":"Explain variables.","input":"","output":"A variable stores a value."}\n',
        encoding="utf-8",
    )

    report = validate_jsonl_file(path, "instruction")

    assert report.valid
    assert report.checked_rows == 1
    assert report.errors == []


def test_instruction_validation_rejects_missing_output(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"instruction":"Explain variables."}\n', encoding="utf-8")

    report = validate_jsonl_file(path, "instruction")

    assert not report.valid
    assert report.errors
