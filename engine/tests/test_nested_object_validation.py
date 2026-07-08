"""Tests for validating lists-of-objects against a per-element shape (issue #195)."""

from __future__ import annotations

from corpus_studio.schemas.base import DatasetSchema, SchemaField
from corpus_studio.validators.basic_validator import validate_example_fields_against

_STEP_FIELDS = [
    SchemaField(name="action", type="string", required=True),
    SchemaField(name="order", type="integer", required=True),
]


def _schema(item_fields: list[SchemaField] | None) -> DatasetSchema:
    return DatasetSchema(
        id="t",
        name="t",
        version="1",
        fields=[
            SchemaField(name="prompt", type="string", required=True),
            SchemaField(
                name="steps",
                type="list",
                required=True,
                item_type="object",
                item_fields=item_fields,
            ),
        ],
    )


def test_valid_list_of_objects_has_no_issues() -> None:
    row = {"prompt": "go", "steps": [{"action": "start", "order": 1}, {"action": "stop", "order": 2}]}
    assert validate_example_fields_against(row, _schema(_STEP_FIELDS)) == []


def test_flags_missing_required_field_in_an_element_with_indexed_path() -> None:
    row = {"prompt": "go", "steps": [{"action": "start", "order": 1}, {"order": 2}]}
    issues = validate_example_fields_against(row, _schema(_STEP_FIELDS))
    assert any(issue.field == "steps[2].action" for issue in issues)


def test_flags_wrong_element_field_type_with_indexed_path() -> None:
    row = {"prompt": "go", "steps": [{"action": "start", "order": "first"}]}
    issues = validate_example_fields_against(row, _schema(_STEP_FIELDS))
    assert any(issue.field == "steps[1].order" for issue in issues)


def test_without_item_fields_only_the_element_type_is_checked() -> None:
    # Backward compatible: no per-element shape means any dict element passes; a non-dict fails.
    schema = _schema(None)
    assert validate_example_fields_against({"prompt": "go", "steps": [{"anything": 1}]}, schema) == []
    bad = validate_example_fields_against({"prompt": "go", "steps": ["not-an-object"]}, schema)
    assert any(issue.field == "steps" for issue in bad)


def test_list_of_objects_validates_recursively() -> None:
    schema = DatasetSchema(
        id="t",
        name="t",
        version="1",
        fields=[
            SchemaField(
                name="groups",
                type="list",
                required=True,
                item_type="object",
                item_fields=[
                    SchemaField(
                        name="items",
                        type="list",
                        required=True,
                        item_type="object",
                        item_fields=[SchemaField(name="k", type="string", required=True)],
                    )
                ],
            )
        ],
    )
    assert validate_example_fields_against({"groups": [{"items": [{"k": "v"}]}]}, schema) == []
    issues = validate_example_fields_against({"groups": [{"items": [{"wrong": "v"}]}]}, schema)
    assert any(issue.field == "groups[1].items[1].k" for issue in issues)
