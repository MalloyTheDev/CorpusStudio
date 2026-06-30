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


def test_validation_rejects_non_object_json_rows(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    report = validate_jsonl_file(path, "instruction")

    assert not report.valid
    assert report.errors[0].message == "Row must be a JSON object."


def test_instruction_validation_rejects_wrong_field_type(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"instruction":"Explain variables.","output":"A variable stores a value.","tags":"bad"}\n',
        encoding="utf-8",
    )

    report = validate_jsonl_file(path, "instruction")

    assert not report.valid
    assert report.errors[0].field == "tags"
    assert report.errors[0].message == "Expected list."


def test_instruction_validation_rejects_non_string_text_field(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"instruction":123,"output":"A variable stores a value."}\n',
        encoding="utf-8",
    )

    report = validate_jsonl_file(path, "instruction")

    assert not report.valid
    assert report.errors[0].field == "instruction"
    assert report.errors[0].message == "Expected text string."


def test_chat_validation_rejects_invalid_messages_shape(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"messages":[{"role":"critic","content":""},"bad message"]}\n',
        encoding="utf-8",
    )

    report = validate_jsonl_file(path, "chat")

    assert not report.valid
    messages = [error.message for error in report.errors]
    assert "Message 1 role must be one of: assistant, system, tool, user." in messages
    assert "Message 1 content must be a non-empty string." in messages
    assert "Message 2 must be an object." in messages
