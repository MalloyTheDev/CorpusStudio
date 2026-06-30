from pydantic import BaseModel


class QualityReport(BaseModel):
    example_count: int
    empty_row_count: int
    duplicate_exact_count: int


def build_basic_quality_report(rows: list[dict]) -> QualityReport:
    seen = set()
    duplicate_count = 0
    empty_count = 0

    for row in rows:
        normalized = str(sorted(row.items()))
        if normalized in seen:
            duplicate_count += 1
        seen.add(normalized)

        if not any(str(value).strip() for value in row.values()):
            empty_count += 1

    return QualityReport(
        example_count=len(rows),
        empty_row_count=empty_count,
        duplicate_exact_count=duplicate_count,
    )
