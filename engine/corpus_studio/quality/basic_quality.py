import json
import re
from typing import Any

from pydantic import BaseModel, Field


LOW_INFORMATION_TOKEN_THRESHOLD = 5
SYNTHETIC_REPEATED_OPENING_THRESHOLD = 3
SYNTHETIC_REPEATED_CLOSING_THRESHOLD = 3
SYNTHETIC_WARNING_LIMIT = 20
GENERIC_SYNTHETIC_PHRASES = (
    "as an ai language model",
    "certainly here is",
    "in conclusion",
    "it is important to note",
    "sure here is",
)


class SyntheticPatternIssue(BaseModel):
    kind: str
    severity: str
    message: str
    row_numbers: list[int] = Field(default_factory=list)
    suggestion: str


class QualityReport(BaseModel):
    example_count: int
    empty_row_count: int
    duplicate_exact_count: int
    duplicate_normalized_count: int
    low_information_count: int
    low_information_token_threshold: int = LOW_INFORMATION_TOKEN_THRESHOLD
    synthetic_pattern_count: int = 0
    synthetic_pattern_warnings: list[str] = Field(default_factory=list)
    synthetic_pattern_issues: list[SyntheticPatternIssue] = Field(default_factory=list)


def build_basic_quality_report(rows: list[dict]) -> QualityReport:
    exact_seen = set()
    normalized_seen = set()
    exact_duplicate_count = 0
    normalized_duplicate_count = 0
    empty_count = 0
    low_information_count = 0
    synthetic_pattern_issues = _synthetic_pattern_issues(rows)
    synthetic_pattern_warnings = [issue.message for issue in synthetic_pattern_issues]

    for row in rows:
        exact_signature = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
        if exact_signature in exact_seen:
            exact_duplicate_count += 1
        exact_seen.add(exact_signature)

        normalized_signature = _normalized_text_signature(row)
        if normalized_signature and normalized_signature in normalized_seen:
            normalized_duplicate_count += 1
        normalized_seen.add(normalized_signature)

        if not any(str(value).strip() for value in row.values()):
            empty_count += 1

        token_count = len(_tokenize_text_values(row))
        if 0 < token_count < LOW_INFORMATION_TOKEN_THRESHOLD:
            low_information_count += 1

    return QualityReport(
        example_count=len(rows),
        empty_row_count=empty_count,
        duplicate_exact_count=exact_duplicate_count,
        duplicate_normalized_count=normalized_duplicate_count,
        low_information_count=low_information_count,
        synthetic_pattern_count=len(synthetic_pattern_issues),
        synthetic_pattern_warnings=synthetic_pattern_warnings,
        synthetic_pattern_issues=synthetic_pattern_issues,
    )


def _normalized_text_signature(value: Any) -> str:
    return " ".join(_tokenize_text_values(value))


def _tokenize_text_values(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9_]+", " ".join(_collect_text_values(value)).lower())


def _collect_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        collected: list[str] = []
        for item in value.values():
            collected.extend(_collect_text_values(item))
        return collected

    if isinstance(value, list):
        collected = []
        for item in value:
            collected.extend(_collect_text_values(item))
        return collected

    if value is None:
        return []

    return [str(value)]


def _synthetic_pattern_issues(rows: list[dict]) -> list[SyntheticPatternIssue]:
    issues: list[SyntheticPatternIssue] = []
    row_texts = [
        (row_number, _normalized_text_signature(row))
        for row_number, row in enumerate(rows, start=1)
    ]
    row_texts = [(row_number, text) for row_number, text in row_texts if text]

    phrase_hits: dict[str, list[int]] = {phrase: [] for phrase in GENERIC_SYNTHETIC_PHRASES}
    opening_rows: dict[str, list[int]] = {}
    closing_rows: dict[str, list[int]] = {}

    for row_number, text in row_texts:
        for phrase in GENERIC_SYNTHETIC_PHRASES:
            if phrase in text:
                phrase_hits[phrase].append(row_number)

        tokens = text.split()
        if len(tokens) >= 8:
            opening = " ".join(tokens[:5])
            closing = " ".join(tokens[-5:])
            opening_rows.setdefault(opening, []).append(row_number)
            closing_rows.setdefault(closing, []).append(row_number)

    for phrase, row_numbers in phrase_hits.items():
        if row_numbers:
            issues.append(
                SyntheticPatternIssue(
                    kind="generic_phrase",
                    severity="medium",
                    message=(
                        f"generic synthetic phrase '{phrase}' appears in row(s): "
                        f"{_format_row_numbers(row_numbers)}."
                    ),
                    row_numbers=row_numbers,
                    suggestion=(
                        "Rewrite these rows with domain-specific phrasing, concrete details, "
                        "and no boilerplate assistant wording."
                    ),
                )
            )

    for opening, row_numbers in sorted(
        opening_rows.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        if len(row_numbers) >= SYNTHETIC_REPEATED_OPENING_THRESHOLD:
            issues.append(
                SyntheticPatternIssue(
                    kind="repeated_opening",
                    severity=_severity_for_repetition(len(row_numbers), rows),
                    message=(
                        f"repeated opening '{opening}' appears in row(s): "
                        f"{_format_row_numbers(row_numbers)}."
                    ),
                    row_numbers=row_numbers,
                    suggestion=(
                        "Vary the prompt setup and first sentence; add different contexts, "
                        "constraints, or user intents before accepting these rows."
                    ),
                )
            )

    for closing, row_numbers in sorted(
        closing_rows.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        if len(row_numbers) >= SYNTHETIC_REPEATED_CLOSING_THRESHOLD:
            issues.append(
                SyntheticPatternIssue(
                    kind="repeated_closing",
                    severity=_severity_for_repetition(len(row_numbers), rows),
                    message=(
                        f"repeated closing '{closing}' appears in row(s): "
                        f"{_format_row_numbers(row_numbers)}."
                    ),
                    row_numbers=row_numbers,
                    suggestion=(
                        "Rewrite endings so outputs resolve the task in distinct ways; remove "
                        "template-like final sentences."
                    ),
                )
            )

    return issues[:SYNTHETIC_WARNING_LIMIT]


def _severity_for_repetition(repetition_count: int, rows: list[dict]) -> str:
    if not rows:
        return "low"

    ratio = repetition_count / len(rows)
    if repetition_count >= 8 or ratio >= 0.5:
        return "high"

    if repetition_count >= 5 or ratio >= 0.25:
        return "medium"

    return "low"


def _format_row_numbers(row_numbers: list[int]) -> str:
    preview = ", ".join(str(row_number) for row_number in row_numbers[:8])
    if len(row_numbers) > 8:
        return f"{preview}, +{len(row_numbers) - 8} more"

    return preview
