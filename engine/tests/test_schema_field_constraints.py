from corpus_studio.schemas.base import DatasetSchema, SchemaField
from corpus_studio.validators.basic_validator import (
    validate_example_fields_against,
    validate_jsonl_row,
)


def _schema(*fields: SchemaField) -> DatasetSchema:
    return DatasetSchema(id="test", name="Test", version="0.1.0", fields=list(fields))


def test_instruction_tags_must_be_strings():
    row = {"instruction": "Explain x.", "output": "y", "tags": ["ok", 5]}
    issues = validate_jsonl_row(row, "instruction")
    assert any("List element 2 must be string" in issue.message for issue in issues)


def test_instruction_string_tags_pass():
    row = {"instruction": "Explain x.", "output": "y", "tags": ["a", "b"]}
    assert validate_jsonl_row(row, "instruction") == []


def test_list_item_type_flags_bad_element_via_crafted_schema():
    schema = _schema(SchemaField(name="scores", type="list", item_type="integer", required=True))
    issues = validate_example_fields_against({"scores": [1, 2, "x"]}, schema)
    assert any("List element 3 must be integer" in issue.message for issue in issues)


def test_list_without_item_type_accepts_mixed_elements():
    schema = _schema(SchemaField(name="items", type="list", required=True))
    assert validate_example_fields_against({"items": [1, "a", {"k": 1}]}, schema) == []


def test_enum_rejects_value_outside_set():
    schema = _schema(SchemaField(name="label", type="string", required=True, enum=["pos", "neg"]))
    issues = validate_example_fields_against({"label": "maybe"}, schema)
    assert any("must be one of" in issue.message.lower() for issue in issues)


def test_enum_accepts_value_in_set():
    schema = _schema(SchemaField(name="label", type="string", required=True, enum=["pos", "neg"]))
    assert validate_example_fields_against({"label": "pos"}, schema) == []


def test_enum_still_reports_type_error_first():
    # A wrong-typed value should get the type error, not just the enum error.
    schema = _schema(SchemaField(name="label", type="string", required=True, enum=["pos", "neg"]))
    issues = validate_example_fields_against({"label": 5}, schema)
    assert any("Expected string string." in issue.message or "Expected" in issue.message for issue in issues)
