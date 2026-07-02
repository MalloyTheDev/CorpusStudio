import json
import re
import unicodedata
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


class PiiFinding(BaseModel):
    """A family of likely PII / secret matches found across the dataset."""

    kind: str
    severity: str
    match_count: int
    row_numbers: list[int] = Field(default_factory=list)
    sample: str
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
    pii_finding_count: int = 0
    pii_findings: list[PiiFinding] = Field(default_factory=list)


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
    pii_findings = _detect_pii(rows)

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
        pii_finding_count=len(pii_findings),
        pii_findings=pii_findings,
    )


def _normalized_text_signature(value: Any) -> str:
    return " ".join(_tokenize_text_values(value))


# Public alias so split-leakage detection can reuse the same normalization.
normalized_text_signature = _normalized_text_signature


# CJK / kana / Hangul scripts have no spaces between words, so each such
# character is treated as its own token; other word characters group into runs.
# This keeps near-duplicate signatures and low-information counts meaningful for
# non-Latin text while preserving ASCII tokenization exactly.
_CJK_RANGES = (
    "぀-ヿ"  # Hiragana + Katakana
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "豈-﫿"  # CJK Compatibility Ideographs
    "가-힯"  # Hangul syllables
    "ｦ-ﾟ"  # Half-width Katakana
)
_TOKEN_RE = re.compile(rf"[{_CJK_RANGES}]|[^\W{_CJK_RANGES}]+", re.UNICODE)


def _tokenize_text_values(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKC", " ".join(_collect_text_values(value))).lower()
    return _TOKEN_RE.findall(text)


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


# --- PII / secret leak detection -------------------------------------------
# High-precision patterns only: the goal is to catch obvious secrets/PII that
# would be harmful to ship into training data, not to be an exhaustive scanner.
_PII_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PII_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_PII_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
)
_PII_API_KEY_RE = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b|\bxox[baprs]-[A-Za-z0-9-]{10,}\b|\bghp_[A-Za-z0-9]{20,}\b"
)
_PII_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\b")
_PII_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PII_CC_CANDIDATE_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")

_PII_PATTERNS = (
    ("private_key", "high", _PII_PRIVATE_KEY_RE, "Remove private key material before using this data."),
    ("aws_access_key", "high", _PII_AWS_KEY_RE, "Remove AWS access keys before using this data."),
    ("api_key", "high", _PII_API_KEY_RE, "Remove API keys / access tokens before using this data."),
    ("jwt", "high", _PII_JWT_RE, "Remove JSON Web Tokens before using this data."),
    ("email", "medium", _PII_EMAIL_RE, "Redact or anonymize email addresses if they are not intended to be public."),
    ("ssn", "medium", _PII_SSN_RE, "Redact Social Security numbers before using this data."),
)


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = (len(digits) - 1) % 2
    for index, char in enumerate(digits):
        value = ord(char) - 48
        # Double every second digit counting from the right (excluding the
        # rightmost check digit): that is the digits whose left index parity
        # differs from the check digit's.
        if index % 2 != parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _mask_secret(text: str) -> str:
    text = text.strip()
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}{'*' * (len(text) - 4)}{text[-2:]}"


def _detect_pii(rows: list[dict]) -> list["PiiFinding"]:
    per_kind: dict[str, dict[str, Any]] = {}

    def _record(kind: str, severity: str, suggestion: str, row_number: int, count: int, sample: str) -> None:
        entry = per_kind.setdefault(
            kind,
            {"severity": severity, "rows": set(), "count": 0, "sample": "", "suggestion": suggestion},
        )
        entry["rows"].add(row_number)
        entry["count"] += count
        if not entry["sample"]:
            entry["sample"] = _mask_secret(sample)

    for row_number, row in enumerate(rows, start=1):
        text = " ".join(_collect_text_values(row))
        if not text:
            continue

        for kind, severity, regex, suggestion in _PII_PATTERNS:
            matches = regex.findall(text)
            if matches:
                _record(kind, severity, suggestion, row_number, len(matches), matches[0])

        credit_cards = [
            digits
            for candidate in _PII_CC_CANDIDATE_RE.findall(text)
            for digits in [re.sub(r"[ -]", "", candidate)]
            if 13 <= len(digits) <= 19 and _luhn_valid(digits)
        ]
        if credit_cards:
            _record(
                "credit_card",
                "high",
                "Remove or tokenize payment card numbers before using this data.",
                row_number,
                len(credit_cards),
                credit_cards[0],
            )

    findings = [
        PiiFinding(
            kind=kind,
            severity=entry["severity"],
            match_count=entry["count"],
            row_numbers=sorted(entry["rows"]),
            sample=entry["sample"],
            suggestion=entry["suggestion"],
        )
        for kind, entry in per_kind.items()
    ]
    findings.sort(key=lambda finding: (-_SEVERITY_ORDER.get(finding.severity, 0), -finding.match_count, finding.kind))
    return findings
