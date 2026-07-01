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
    pattern: str = ""


class SyntheticPatternCluster(BaseModel):
    """A group of near-duplicate synthetic patterns of the same kind."""

    kind: str
    label: str
    severity: str
    member_count: int
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
    synthetic_pattern_clusters: list[SyntheticPatternCluster] = Field(default_factory=list)


def build_basic_quality_report(rows: list[dict]) -> QualityReport:
    exact_seen = set()
    normalized_seen = set()
    exact_duplicate_count = 0
    normalized_duplicate_count = 0
    empty_count = 0
    low_information_count = 0
    synthetic_pattern_issues = _synthetic_pattern_issues(rows)
    synthetic_pattern_warnings = [issue.message for issue in synthetic_pattern_issues]
    synthetic_pattern_clusters = cluster_synthetic_pattern_issues(synthetic_pattern_issues)

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
        synthetic_pattern_clusters=synthetic_pattern_clusters,
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
                    pattern=phrase,
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
                    pattern=opening,
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
                    pattern=closing,
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


_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
SYNTHETIC_CLUSTER_SIMILARITY = 0.5


def _pattern_tokens(pattern: str) -> set[str]:
    return set(pattern.split())


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _max_severity(severities: list[str]) -> str:
    best = "low"
    for severity in severities:
        if _SEVERITY_ORDER.get(severity, 0) > _SEVERITY_ORDER.get(best, 0):
            best = severity
    return best


def cluster_synthetic_pattern_issues(
    issues: list[SyntheticPatternIssue],
    similarity_threshold: float = SYNTHETIC_CLUSTER_SIMILARITY,
) -> list[SyntheticPatternCluster]:
    """Merge same-kind synthetic issues whose patterns overlap into clusters.

    Near-duplicate templates (e.g. openings that differ by a word or two) are
    grouped by token-set Jaccard similarity so a family of related issues shows
    up as one cluster instead of many fragmented warnings.
    """
    clusters: list[SyntheticPatternCluster] = []

    for kind in dict.fromkeys(issue.kind for issue in issues):
        kind_issues = [issue for issue in issues if issue.kind == kind]
        merged = [False] * len(kind_issues)

        for index, issue in enumerate(kind_issues):
            if merged[index]:
                continue

            members = [issue]
            merged[index] = True
            tokens = _pattern_tokens(issue.pattern)

            for other_index in range(index + 1, len(kind_issues)):
                if merged[other_index]:
                    continue
                other = kind_issues[other_index]
                if _token_jaccard(tokens, _pattern_tokens(other.pattern)) >= similarity_threshold:
                    members.append(other)
                    merged[other_index] = True

            row_numbers = sorted({row for member in members for row in member.row_numbers})
            clusters.append(
                SyntheticPatternCluster(
                    kind=kind,
                    label=issue.pattern or issue.message,
                    severity=_max_severity([member.severity for member in members]),
                    member_count=len(members),
                    row_numbers=row_numbers,
                    suggestion=issue.suggestion,
                )
            )

    clusters.sort(key=lambda cluster: (-len(cluster.row_numbers), cluster.kind, cluster.label))
    return clusters
