"""Guardrail: every built-in schema ships a description and a valid example.

The desktop new-project flow shows each schema's example and pre-fills the
editor with it, so a broken example would confuse exactly the people it is meant
to help. These tests fail loudly if a template drifts out of sync with its
schema.
"""

from corpus_studio.schemas.registry import list_builtin_schemas
from corpus_studio.validators.basic_validator import validate_jsonl_row


def test_builtin_schemas_exist():
    assert list_builtin_schemas(), "no built-in schemas were found"


def test_every_builtin_schema_has_description_and_example():
    for schema in list_builtin_schemas():
        assert schema.description, f"{schema.id} is missing a description"
        assert schema.example is not None, f"{schema.id} is missing an example template"


def test_every_schema_example_is_valid_against_its_schema():
    for schema in list_builtin_schemas():
        assert schema.example is not None
        issues = validate_jsonl_row(schema.example, schema.id)
        assert issues == [], f"{schema.id} example is invalid: {issues}"
