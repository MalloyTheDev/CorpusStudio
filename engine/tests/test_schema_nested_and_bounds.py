from corpus_studio.schemas.base import DatasetSchema, SchemaField
from corpus_studio.validators.basic_validator import validate_example_fields_against


def _schema(*fields: SchemaField) -> DatasetSchema:
    return DatasetSchema(id="test", name="Test", version="0.1.0", fields=list(fields))


# --- numeric bounds ----------------------------------------------------------

def test_numeric_minimum_enforced():
    schema = _schema(SchemaField(name="score", type="integer", required=True, minimum=0))
    issues = validate_example_fields_against({"score": -1}, schema)
    assert any("must be >= 0" in issue.message for issue in issues)


def test_numeric_maximum_enforced():
    schema = _schema(SchemaField(name="ratio", type="float", required=True, maximum=1.0))
    issues = validate_example_fields_against({"ratio": 1.5}, schema)
    assert any("must be <= 1.0" in issue.message for issue in issues)


def test_numeric_in_range_passes():
    schema = _schema(SchemaField(name="score", type="integer", required=True, minimum=0, maximum=100))
    assert validate_example_fields_against({"score": 50}, schema) == []


# --- nested object shapes ----------------------------------------------------

def _metadata_schema(*sub_fields: SchemaField) -> DatasetSchema:
    return _schema(SchemaField(name="metadata", type="object", required=True, fields=list(sub_fields)))


def test_nested_required_subfield_missing():
    schema = _metadata_schema(SchemaField(name="source", type="string", required=True))
    # Non-empty object missing the required sub-field.
    issues = validate_example_fields_against({"metadata": {"other": "y"}}, schema)
    assert any(issue.message == "Missing required field: metadata.source" for issue in issues)


def test_nested_subfield_type_error_is_pathed():
    schema = _metadata_schema(SchemaField(name="source", type="string", required=True))
    issues = validate_example_fields_against({"metadata": {"source": 5}}, schema)
    assert any(issue.field == "metadata.source" for issue in issues)


def test_nested_numeric_bound_enforced():
    schema = _metadata_schema(
        SchemaField(name="score", type="float", required=True, minimum=0, maximum=100)
    )
    issues = validate_example_fields_against({"metadata": {"score": 150}}, schema)
    assert any(issue.field == "metadata.score" and "<= 100" in issue.message for issue in issues)


def test_nested_object_valid_passes():
    schema = _metadata_schema(
        SchemaField(name="source", type="string", required=True),
        SchemaField(name="score", type="float", required=False, minimum=0, maximum=100),
    )
    assert validate_example_fields_against({"metadata": {"source": "x", "score": 50.0}}, schema) == []
