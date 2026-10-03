import json
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any, Final

from pydantic import BaseModel, Field

from corpus_studio.quality.applicability import (
    QualityApplicability,
    free_text_leaf_paths,
    measure_applicability,
)
from corpus_studio.quality.text import collect_text_values, tokenize_text_values
from corpus_studio.schemas.base import DatasetSchema
from corpus_studio.tokenization.estimate import estimate_tokens


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


class TokenLengthOutlier(BaseModel):
    """A row whose estimated token count is an unusually high outlier."""

    row_number: int
    token_count: int


class CategoryImbalance(BaseModel):
    """A low-cardinality field where one value dominates the dataset."""

    field: str
    dominant_value: str
    dominant_count: int
    total: int
    share: float
    distinct_values: int


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
    token_length_threshold: int = 0
    token_length_outlier_count: int = 0
    token_length_outliers: list[TokenLengthOutlier] = Field(default_factory=list)
    category_imbalances: list[CategoryImbalance] = Field(default_factory=list)
    # Which signals above could actually run on this dataset's shape, and how much of its content
    # they read. ``None`` means the applicability was never assessed (a report assembled by hand
    # from counts, with no rows to measure) - not that every signal applied.
    applicability: QualityApplicability | None = None


def build_basic_quality_report(
    rows: list[dict],
    schema: DatasetSchema | None = None,
) -> QualityReport:
    """Build the quality report, skipping the signals ``schema``'s shape does not support.

    A signal the shape cannot support is NOT computed: its count stays 0 and its name is listed in
    ``applicability.not_applicable_signals``, so a reader can never mistake "nothing to read" for
    "nothing to fix". Without a schema every signal runs and the applicability is reported
    ``unmeasured`` - see :mod:`corpus_studio.quality.applicability` for why the roles cannot be
    inferred from the values.
    """

    applicability = measure_applicability(rows, schema)
    exact_seen = set()
    normalized_seen = set()
    exact_duplicate_count = 0
    normalized_duplicate_count = 0
    empty_count = 0
    low_information_count = 0
    check_normalized = applicability.applies("normalized_duplicates")
    check_low_information = applicability.applies("low_information")

    if applicability.applies("synthetic_patterns"):
        all_synthetic_pattern_issues = _synthetic_pattern_issues(rows, schema)
    else:
        all_synthetic_pattern_issues = []
    synthetic_pattern_count = len(all_synthetic_pattern_issues)  # true total, before display cap
    synthetic_pattern_issues = all_synthetic_pattern_issues[:SYNTHETIC_WARNING_LIMIT]
    synthetic_pattern_warnings = [issue.message for issue in synthetic_pattern_issues]
    synthetic_pattern_clusters = cluster_synthetic_pattern_issues(synthetic_pattern_issues)
    pii_findings = _detect_pii(rows)
    if applicability.applies("token_length_outliers"):
        token_length_threshold, token_length_outliers = _token_length_outliers(rows)
    else:
        token_length_threshold, token_length_outliers = 0, []
    category_imbalances = (
        _category_imbalances(rows) if applicability.applies("category_imbalance") else []
    )

    for row in rows:
        exact_signature = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
        if exact_signature in exact_seen:
            exact_duplicate_count += 1
        exact_seen.add(exact_signature)

        if check_normalized:
            normalized_signature = _normalized_text_signature(row)
            if normalized_signature and normalized_signature in normalized_seen:
                normalized_duplicate_count += 1
            normalized_seen.add(normalized_signature)

        if not any(str(value).strip() for value in row.values()):
            empty_count += 1

        if check_low_information:
            token_count = len(_tokenize_text_values(row))
            if 0 < token_count < LOW_INFORMATION_TOKEN_THRESHOLD:
                low_information_count += 1

    return QualityReport(
        example_count=len(rows),
        empty_row_count=empty_count,
        duplicate_exact_count=exact_duplicate_count,
        duplicate_normalized_count=normalized_duplicate_count,
        low_information_count=low_information_count,
        synthetic_pattern_count=synthetic_pattern_count,
        synthetic_pattern_warnings=synthetic_pattern_warnings,
        synthetic_pattern_issues=synthetic_pattern_issues,
        synthetic_pattern_clusters=synthetic_pattern_clusters,
        pii_finding_count=len(pii_findings),
        pii_findings=pii_findings,
        token_length_threshold=token_length_threshold,
        token_length_outlier_count=len(token_length_outliers),
        token_length_outliers=token_length_outliers,
        category_imbalances=category_imbalances,
        applicability=applicability,
    )


# Unit separator — cannot appear in tokenized text, so it safely delimits fields.
_FIELD_SEP = "\x1f"


def _normalized_text_signature(value: Any) -> str:
    if isinstance(value, dict):
        # Field-aware: each key's token stream is scoped by the key and joined by a reserved
        # separator, so two rows with the same combined text but a different field structure
        # do NOT collide (which caused false near-dup drops and false split-leakage blocks).
        return _FIELD_SEP.join(
            f"{key}={' '.join(_tokenize_text_values(sub))}"
            for key, sub in sorted(value.items(), key=lambda item: item[0])
        )
    return " ".join(_tokenize_text_values(value))


# Public alias so split-leakage detection can reuse the same normalization.
normalized_text_signature = _normalized_text_signature


# Tokenization lives in quality.text so the applicability assessment can share it without a
# circular import. These module-private aliases keep the call sites below unchanged.
_tokenize_text_values = tokenize_text_values
_collect_text_values = collect_text_values


# A field whose every value is one whitespace-free string containing a path separator is a path,
# a URL, or a similar located identifier: its leading segments are shared BY CONSTRUCTION, so a
# shared "opening" across rows says nothing about templated prose.
_PATH_SHAPED_VALUE = re.compile(r"\A\S*[/\\]\S*\Z")


def _leaf_values(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every scalar inside ``value`` as ``(leaf_path, scalar)``. List elements collapse onto one
    index-free path (``tags[]``): every element of a list is the same field."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_values(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for item in value:
            yield from _leaf_values(item, f"{prefix}[]")
    else:
        yield prefix, value


def structurally_repeating_leaves(rows: list[dict]) -> frozenset[str]:
    """Leaf paths whose repetition is structural, so templating detection must not read them.

    Two measured cases, both independent of any schema:

    * **constant** - one distinct value wherever the field appears. A corpus-wide provenance tag
      (``source_dataset``, ``source_version``, ``source_url``), a fixed license, a lane, a policy
      note: these are REQUIRED to be identical across a lane, so reading their repetition as
      "templated repetition" inverts the verdict - it flags the metadata for being correct.
    * **path-shaped** - see ``_PATH_SHAPED_VALUE``.

    Constancy only carries information across at least two rows, so a shorter dataset excludes
    nothing (it also cannot reach the repetition thresholds).
    """
    if len(rows) < 2:
        return frozenset()

    distinct: dict[str, set[str]] = {}
    path_shaped: dict[str, bool] = {}
    for row in rows:
        for path, value in _leaf_values(row):
            distinct.setdefault(path, set()).add(repr(value))
            shaped = isinstance(value, str) and bool(_PATH_SHAPED_VALUE.match(value))
            path_shaped[path] = path_shaped.get(path, True) and shaped

    constant = {path for path, values in distinct.items() if len(values) == 1}
    return frozenset(constant | {path for path, shaped in path_shaped.items() if shaped})


# Returned in place of a dropped SCALAR so its container can omit it; never a data value.
_DROPPED: Final[object] = object()


def _without_leaves(value: Any, excluded: frozenset[str], prefix: str = "") -> Any:
    """``value`` with every leaf in ``excluded`` removed, containers otherwise intact.

    The walk mirrors :func:`_leaf_values` exactly: only a SCALAR is dropped, and only when its own
    leaf path is excluded. A container is always recursed into, never removed wholesale, because a
    field that is ``null`` in some rows and a populated list in others occupies BOTH paths
    (``meta.mustInclude`` and ``meta.mustInclude[]``) - dropping the key on the scalar path would
    silently discard the list's contents too.
    """
    if isinstance(value, dict):
        pruned_pairs = (
            (key, _without_leaves(item, excluded, f"{prefix}.{key}" if prefix else key))
            for key, item in value.items()
        )
        return {key: item for key, item in pruned_pairs if item is not _DROPPED}
    if isinstance(value, list):
        pruned = [_without_leaves(item, excluded, f"{prefix}[]") for item in value]
        return [item for item in pruned if item is not _DROPPED]
    return _DROPPED if prefix in excluded else value


def _synthetic_pattern_text(row: Any, excluded: frozenset[str] = frozenset()) -> str:
    """Flat text for synthetic-pattern detection, minus the fields whose repetition is structural.

    ``excluded`` drops the leaves :func:`structurally_repeating_leaves` measured as required to
    repeat. For a CHAT row this additionally DROPS system-role message(s): an identical shared
    system prompt across chat rows is intentional (the dataset's system instruction), not a
    synthetic 'repeated opening/closing' - flagging it is a false positive. It is the same rule one
    level deeper than a leaf path reaches, since a role decides which message is boilerplate. The
    user/assistant turns are still checked. Non-chat rows (and chat rows with no system message) are
    otherwise unchanged."""
    if isinstance(row, dict):
        row = _without_leaves(row, excluded)
        messages = row.get("messages")
        if isinstance(messages, list) and any(
            isinstance(message, dict) and message.get("role") == "system" for message in messages
        ):
            without_system = {key: value for key, value in row.items() if key != "messages"}
            without_system["messages"] = [
                message
                for message in messages
                if not (isinstance(message, dict) and message.get("role") == "system")
            ]
            return " ".join(_tokenize_text_values(without_system))
    return " ".join(_tokenize_text_values(row))


def _is_within(path: str, prefixes: frozenset[str]) -> bool:
    """Whether ``path`` IS one of ``prefixes`` or sits underneath one (``messages[].content``
    under ``messages``)."""
    return any(
        path == prefix or path.startswith(f"{prefix}.") or path.startswith(f"{prefix}[")
        for prefix in prefixes
    )


def _only_free_text(value: Any, prose: frozenset[str], prefix: str = "") -> Any:
    """``value`` keeping only the scalars that sit within a declared prose leaf."""
    if isinstance(value, dict):
        pruned_pairs = (
            (key, _only_free_text(item, prose, f"{prefix}.{key}" if prefix else key))
            for key, item in value.items()
        )
        return {key: item for key, item in pruned_pairs if item is not _DROPPED}
    if isinstance(value, list):
        pruned = [_only_free_text(item, prose, f"{prefix}[]") for item in value]
        return [item for item in pruned if item is not _DROPPED]
    return value if _is_within(prefix, prose) else _DROPPED


def _synthetic_pattern_issues(
    rows: list[dict],
    schema: DatasetSchema | None = None,
) -> list[SyntheticPatternIssue]:
    issues: list[SyntheticPatternIssue] = []
    # Synthetic-pattern detection reads the flat surface text (openings/closings/phrases),
    # NOT the field-aware dedup signature - field prefixes/separators would corrupt n-grams.
    #
    # Which fields it reads depends on what is known about the shape, and the two rules differ
    # because only one of them can tell metadata from content:
    #   * WITH a schema, only the leaves it declares as PROSE. That is the sharper rule: a
    #     provenance tag or a path is excluded for being non-prose, while a prose field that is
    #     identical in every row stays in - an identical assistant answer across rows is the
    #     archetype of templated data, not structural repetition.
    #   * WITHOUT one, every leaf except those measured as structurally repeating. Constancy
    #     cannot distinguish a corpus-wide license from a copy-pasted answer, so this fallback
    #     can miss the latter; that is why a report with no measured applicability and no
    #     findings withholds its grade rather than publishing a pass.
    # The shared chat system prompt is dropped either way (see _synthetic_pattern_text).
    if schema is not None:
        prose = free_text_leaf_paths(schema)
        prepared: list[Any] = [_only_free_text(row, prose) for row in rows]
        excluded: frozenset[str] = frozenset()
    else:
        prepared = list(rows)
        excluded = structurally_repeating_leaves(rows)
    row_texts = [
        (row_number, _synthetic_pattern_text(row, excluded))
        for row_number, row in enumerate(prepared, start=1)
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

    # Return the full set; the caller reports the true count and caps only the displayed
    # sample (capping here would make synthetic_pattern_count under-report at >20 patterns).
    return issues


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
# PEM / armor labels of private-key blocks (PKCS#1 RSA, SEC1 EC, DSA, OpenSSH, PKCS#8 plain and
# ENCRYPTED, and PGP, whose armor label ends in " BLOCK").
_PEM_PRIVATE_KEY_LABEL = r"(?:(?:RSA|EC|DSA|OPENSSH|ENCRYPTED|PGP) )?PRIVATE KEY(?: BLOCK)?"
# One match spans the WHOLE key, because redaction masks exactly what matches (in the same joined
# row text _detect_pii scans, so a key split across fields is one match): a header-only match would
# mask the BEGIN line and leave the key body and END line in the deliverable.
#   1. A BEGIN marker through the first END marker; with no END marker, through the end of the text
#      (an unterminated block is still key material, so it must not survive by losing its footer).
#   2. An END marker not consumed by branch 1, i.e. key material whose BEGIN marker is not in the
#      text (a key cut off before its header, or a header masked by an earlier header-only
#      redaction). Detecting it keeps the gate from passing on that leftover body; redaction masks
#      the body before it.
# Linear time: both branches start with a literal "-----", and branch 1's `.*` fallback consumes the
# rest of the text whenever no END marker follows, so a BEGIN never rescans text another BEGIN has
# already scanned. There are no capture groups, so findall() returns whole matches.
_PII_PRIVATE_KEY_RE = re.compile(
    rf"-----BEGIN {_PEM_PRIVATE_KEY_LABEL}-----(?:.*?-----END {_PEM_PRIVATE_KEY_LABEL}-----|.*)"
    rf"|-----END {_PEM_PRIVATE_KEY_LABEL}-----",
    re.DOTALL,
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


def _looks_like_payment_card(digits: str) -> bool:
    """Whether a Luhn-valid digit run also matches a known card brand's IIN prefix AND that
    brand's length class. Requiring this (not just Luhn + length) stops Luhn-valid non-cards —
    IMEIs, order/invoice numbers, GS1 codes — from false-positiving as payment cards."""

    n = len(digits)
    head4 = int(digits[:4]) if n >= 4 else 0
    if digits[0] == "4" and n in (13, 16, 19):  # Visa
        return True
    if n == 16 and (digits[:2] in {"51", "52", "53", "54", "55"} or 2221 <= head4 <= 2720):  # Mastercard
        return True
    if n == 15 and digits[:2] in {"34", "37"}:  # American Express
        return True
    if n in (16, 19) and (  # Discover
        digits[:4] == "6011" or digits[:2] == "65" or (n >= 3 and 644 <= int(digits[:3]) <= 649)
    ):
        return True
    if n == 14 and (digits[:3] in {"300", "301", "302", "303", "304", "305"} or digits[:2] in {"36", "38", "39"}):
        return True  # Diners Club
    if 16 <= n <= 19 and 3528 <= head4 <= 3589:  # JCB
        return True
    # UnionPay (IIN 62) and Maestro (50, 56-59) are constrained to 16-19 digits so they can
    # NEVER match a 15-digit IMEI — the length gap is what keeps the IMEI false positive out
    # while still catching these real card brands (Maestro/Discover 65 and UnionPay 62 overlap,
    # so 65 stays with Discover above; only the 50/56-59 Maestro range is added here).
    if 16 <= n <= 19 and digits[:2] == "62":  # UnionPay
        return True
    if 16 <= n <= 19 and digits[:2] in {"50", "56", "57", "58", "59"}:  # Maestro
        return True
    return False


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
                sample = matches[0]
                if kind == "private_key":
                    # A private-key match is the whole block (or the rest of an unterminated one), so
                    # sample only its "-----BEGIN/END <label>-----" marker: _mask_secret keeps the
                    # edge characters, which must never be key body, and the sample stays short.
                    sample = sample[: sample.index("-----", len("-----")) + len("-----")]
                _record(kind, severity, suggestion, row_number, len(matches), sample)

        credit_cards = [
            digits
            for candidate in _PII_CC_CANDIDATE_RE.findall(text)
            for digits in [re.sub(r"[ -]", "", candidate)]
            if 13 <= len(digits) <= 19 and _luhn_valid(digits) and _looks_like_payment_card(digits)
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


# --- token-length outliers & category imbalance ----------------------------
TOKEN_LENGTH_MIN_ROWS = 8
TOKEN_LENGTH_OUTLIER_LIMIT = 50
CATEGORY_IMBALANCE_MIN_ROWS = 10
CATEGORY_IMBALANCE_SHARE = 0.8
CATEGORY_IMBALANCE_MAX_DISTINCT = 20
CATEGORY_IMBALANCE_LIMIT = 20


def _percentile(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _token_length_outliers(rows: list[dict]) -> tuple[int, list[TokenLengthOutlier]]:
    counts: list[tuple[int, int]] = []
    for row_number, row in enumerate(rows, start=1):
        text = " ".join(_collect_text_values(row))
        if not text.strip():
            continue
        counts.append((row_number, estimate_tokens(text)))

    if len(counts) < TOKEN_LENGTH_MIN_ROWS:
        return 0, []

    values = sorted(count for _, count in counts)
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    threshold = q3 + 1.5 * (q3 - q1)

    outliers = [
        TokenLengthOutlier(row_number=row_number, token_count=count)
        for row_number, count in counts
        if count > threshold and count > q3
    ]
    outliers.sort(key=lambda outlier: -outlier.token_count)
    return round(threshold), outliers[:TOKEN_LENGTH_OUTLIER_LIMIT]


def _category_imbalances(rows: list[dict]) -> list[CategoryImbalance]:
    if len(rows) < CATEGORY_IMBALANCE_MIN_ROWS:
        return []

    scalar_values: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, (str, int, float, bool)):
                scalar_values.setdefault(key, []).append(str(value))

    total_rows = len(rows)
    max_distinct = min(CATEGORY_IMBALANCE_MAX_DISTINCT, total_rows // 2)
    findings: list[CategoryImbalance] = []

    for field, values in scalar_values.items():
        # Field must be present (as a scalar) in most rows to be a category.
        if len(values) < total_rows * 0.8:
            continue
        counter = Counter(values)
        distinct = len(counter)
        if distinct < 2 or distinct > max_distinct:
            continue
        dominant_value, dominant_count = counter.most_common(1)[0]
        # Share is relative to the whole dataset, not just rows where the field is present —
        # otherwise a field present in 80% of rows and dominant within that subset over-reports
        # (e.g. dominant in 79/80 present rows = 0.99 present-share but only 0.79 dataset-share).
        share = dominant_count / total_rows
        if share >= CATEGORY_IMBALANCE_SHARE:
            findings.append(
                CategoryImbalance(
                    field=field,
                    dominant_value=dominant_value[:80],
                    dominant_count=dominant_count,
                    total=total_rows,
                    share=round(share, 3),
                    distinct_values=distinct,
                )
            )

    findings.sort(key=lambda finding: -finding.share)
    return findings[:CATEGORY_IMBALANCE_LIMIT]
