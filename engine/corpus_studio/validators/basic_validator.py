import json
from pathlib import Path
from typing import Any

from corpus_studio.schemas.base import DatasetSchema, SchemaField
from corpus_studio.schemas.registry import load_builtin_schema
from corpus_studio.validators.results import ValidationIssue, ValidationReport


TEXT_FIELD_TYPES = {"string", "text", "markdown", "code", "file_path", "image_path"}
VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def _issue(
    message: str,
    row_number: int | None = None,
    field: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        level="error",
        message=message,
        row_number=row_number,
        field=field,
    )


def _is_empty_required_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, dict)):
        return len(value) == 0

    return False


def _validate_messages(value: Any, field_name: str, row_number: int | None) -> list[ValidationIssue]:
    if not isinstance(value, list):
        return [_issue("Expected messages list.", row_number, field_name)]

    issues: list[ValidationIssue] = []
    for index, message in enumerate(value, start=1):
        if not isinstance(message, dict):
            issues.append(_issue(f"Message {index} must be an object.", row_number, field_name))
            continue

        role = message.get("role")
        if not isinstance(role, str) or role not in VALID_MESSAGE_ROLES:
            roles = ", ".join(sorted(VALID_MESSAGE_ROLES))
            issues.append(
                _issue(
                    f"Message {index} role must be one of: {roles}.",
                    row_number,
                    field_name,
                )
            )

        content = message.get("content")
        if not isinstance(content, str) or content.strip() == "":
            issues.append(
                _issue(
                    f"Message {index} content must be a non-empty string.",
                    row_number,
                    field_name,
                )
            )

    return issues


def _matches_scalar_type(field_type: str, value: Any) -> bool:
    if field_type in TEXT_FIELD_TYPES:
        return isinstance(value, str)
    if field_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field_type == "boolean":
        return isinstance(value, bool)
    if field_type == "list":
        return isinstance(value, list)
    if field_type == "object":
        return isinstance(value, dict)
    return True


def _type_error_message(field_type: str) -> str:
    if field_type in TEXT_FIELD_TYPES:
        return f"Expected {field_type} string."
    if field_type == "integer":
        return "Expected integer."
    if field_type == "float":
        return "Expected number."
    if field_type == "boolean":
        return "Expected boolean."
    if field_type == "list":
        return "Expected list."
    if field_type == "object":
        return "Expected object."
    return "Invalid value."


def _validate_field_type(field: SchemaField, value: Any, row_number: int | None) -> list[ValidationIssue]:
    field_type = field.type

    if field_type == "messages":
        return _validate_messages(value, field.name, row_number)

    if not _matches_scalar_type(field_type, value):
        return [_issue(_type_error_message(field_type), row_number, field.name)]

    issues: list[ValidationIssue] = []

    if field_type == "list" and field.item_type is not None:
        for index, element in enumerate(value, start=1):
            if not _matches_scalar_type(field.item_type, element):
                issues.append(
                    _issue(
                        f"List element {index} must be {field.item_type}.",
                        row_number,
                        field.name,
                    )
                )

    if field.enum is not None and value not in field.enum:
        allowed = ", ".join(str(option) for option in field.enum)
        issues.append(
            _issue(
                f"Value must be one of: {allowed}.",
                row_number,
                field.name,
            )
        )

    return issues


def validate_example_fields(
    row: dict[str, Any],
    schema_id: str,
    row_number: int | None = None,
) -> list[ValidationIssue]:
    return validate_example_fields_against(row, load_builtin_schema(schema_id), row_number)


def validate_example_fields_against(
    row: dict[str, Any],
    schema: DatasetSchema,
    row_number: int | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for field in schema.fields:
        if field.name not in row:
            if field.required:
                issues.append(
                    _issue(
                        f"Missing required field: {field.name}",
                        row_number,
                        field.name,
                    )
                )
            continue

        value = row[field.name]
        if field.required and _is_empty_required_value(value):
            issues.append(
                _issue(
                    f"Required field is empty: {field.name}",
                    row_number,
                    field.name,
                )
            )
            continue

        if value is None:
            continue

        issues.extend(_validate_field_type(field, value, row_number))

    return issues


def validate_jsonl_row(
    row: Any,
    schema_id: str,
    row_number: int | None = None,
) -> list[ValidationIssue]:
    if not isinstance(row, dict):
        return [_issue("Row must be a JSON object.", row_number)]

    return validate_example_fields(row, schema_id, row_number)


def validate_jsonl_file(path: Path, schema_id: str) -> ValidationReport:
    report = ValidationReport(valid=True, schema_id=schema_id)

    with path.open("r", encoding="utf-8-sig") as f:
        for row_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            report.checked_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(
                    _issue(
                        f"Invalid JSON: {exc}",
                        row_number,
                    )
                )
                continue

            report.errors.extend(validate_jsonl_row(row, schema_id, row_number))

    report.valid = len(report.errors) == 0
    return report
