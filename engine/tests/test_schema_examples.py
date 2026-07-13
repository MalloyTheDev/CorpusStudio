"""Guardrail: every built-in schema ships a description and a valid example.

The desktop new-project flow shows each schema's example and pre-fills the
editor with it, so a broken example would confuse exactly the people it is meant
to help. These tests fail loudly if a template drifts out of sync with its
schema.
"""

from corpus_studio.schemas.registry import list_builtin_schemas, load_builtin_schema
from corpus_studio.validators.basic_validator import validate_jsonl_row


def test_builtin_schemas_exist():
    assert list_builtin_schemas(), "no built-in schemas were found"


def test_readme_advertised_schemas_all_ship():
    """The README advertises these dataset types as supported schemas; each must ship a
    built-in so the docs never promise a schema the app can't create (see the classification
    gap that prompted this)."""
    advertised = {
        "raw_text",
        "instruction",
        "chat",
        "preference",
        "code",
        "image_caption",
        "classification",
        "retrieval",
        "evaluation",
        "trace",
    }
    shipped = {schema.id for schema in list_builtin_schemas()}
    missing = advertised - shipped
    assert not missing, f"README advertises schemas with no built-in: {sorted(missing)}"


def test_classification_schema_shape_and_validation():
    schema = load_builtin_schema("classification")
    required = {field.name for field in schema.fields if field.required}
    assert required == {"text", "label"}, f"classification required fields drifted: {required}"

    assert validate_jsonl_row({"text": "Great value.", "label": "positive"}, "classification") == []
    # A row missing the required label must be rejected.
    assert validate_jsonl_row({"text": "Great value."}, "classification") != []


def test_every_builtin_schema_has_description_and_example():
    for schema in list_builtin_schemas():
        assert schema.description, f"{schema.id} is missing a description"
        assert schema.example is not None, f"{schema.id} is missing an example template"


def test_every_schema_example_is_valid_against_its_schema():
    for schema in list_builtin_schemas():
        assert schema.example is not None
        issues = validate_jsonl_row(schema.example, schema.id)
        assert issues == [], f"{schema.id} example is invalid: {issues}"
