from pathlib import Path

from pydantic import BaseModel, Field

from corpus_studio.importers.jsonl_importer import iter_jsonl
from corpus_studio.validators.basic_validator import validate_jsonl_row
from corpus_studio.validators.results import ValidationIssue


class ImportFailure(BaseModel):
    row_number: int
    raw_preview: str
    errors: list[ValidationIssue] = Field(default_factory=list)


class ImportPreviewReport(BaseModel):
    valid: bool
    schema_id: str
    path: str
    total_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    failed_rows: list[ImportFailure] = Field(default_factory=list)


def preview_jsonl_import(path: Path, schema_id: str) -> ImportPreviewReport:
    report = ImportPreviewReport(
        valid=True,
        schema_id=schema_id,
        path=str(path),
    )

    for parsed in iter_jsonl(path):
        row_number = parsed.line_number
        report.total_rows += 1
        errors: list[ValidationIssue]

        if parsed.error is not None:
            errors = [
                ValidationIssue(
                    level="error",
                    message=parsed.error,
                    row_number=row_number,
                )
            ]
        else:
            errors = validate_jsonl_row(parsed.value, schema_id, row_number)

        if errors:
            report.failed_rows.append(
                ImportFailure(
                    row_number=row_number,
                    raw_preview=parsed.raw.strip()[:240],
                    errors=errors,
                )
            )
            continue

        report.accepted_rows += 1

    report.rejected_rows = len(report.failed_rows)
    report.valid = report.rejected_rows == 0
    return report
