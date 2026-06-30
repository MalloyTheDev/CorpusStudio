import json
from pathlib import Path
from typing import Any

from corpus_studio.schemas.registry import load_builtin_schema
from corpus_studio.validators.results import ValidationIssue, ValidationReport


def validate_example_fields(
    row: dict[str, Any],
    schema_id: str,
    row_number: int | None = None,
) -> list[ValidationIssue]:
    schema = load_builtin_schema(schema_id)
    issues: list[ValidationIssue] = []

    for field in schema.fields:
        if field.required and field.name not in row:
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"Missing required field: {field.name}",
                    row_number=row_number,
                    field=field.name,
                )
            )
            continue

        value = row.get(field.name)
        if field.required and (value is None or value == ""):
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"Required field is empty: {field.name}",
                    row_number=row_number,
                    field=field.name,
                )
            )

    return issues


def validate_jsonl_file(path: Path, schema_id: str) -> ValidationReport:
    report = ValidationReport(valid=True, schema_id=schema_id)

    with path.open("r", encoding="utf-8") as f:
        for row_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            report.checked_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(
                    ValidationIssue(
                        level="error",
                        message=f"Invalid JSON: {exc}",
                        row_number=row_number,
                    )
                )
                continue

            report.errors.extend(validate_example_fields(row, schema_id, row_number))

    report.valid = len(report.errors) == 0
    return report
