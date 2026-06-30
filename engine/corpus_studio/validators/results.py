from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    level: str
    message: str
    row_number: int | None = None
    field: str | None = None


class ValidationReport(BaseModel):
    valid: bool
    schema_id: str
    checked_rows: int = 0
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
