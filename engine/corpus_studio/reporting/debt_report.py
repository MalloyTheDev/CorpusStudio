"""Dataset Debt ledger — a prioritized, normalized view of a dataset's outstanding
quality problems (v1.1).

Turns the flat counts of a :class:`QualityReport` into a ranked "pay this down
first" ledger with one honest health grade. It adds **no** new detection — every
item derives from ``build_basic_quality_report``. The value is what the raw report
does not give: **normalization** (rates per dataset size, so "8 duplicates" is read
as 40% of 20 rows vs 0.008% of 100k), **cross-category prioritization** (severity
ranking, so there is a clear #1 fix), and a single **documented grade** — answering
"is this dataset train-ready, and if not, what do I fix first?".

Severity is coarse and rule-based, never a fake-precise number. Rates are used only
where a rate is meaningful; the **secret/PII class is PRESENCE-based, never
normalized by rate** — a single leaked credential is critical no matter how large
the dataset.

Severity rules (documented, per category):
- empty_rows / low_information: rate > 0.10 → high, > 0.02 → moderate, > 0 → low.
- exact / normalized duplicates: rate > 0.05 → high, > 0.01 → moderate, > 0 → low.
- secrets (high-severity PII): present → **critical** (presence, not rate).
- personal_data (medium-severity PII): present → **high** (presence, not rate).
- synthetic_patterns: max issue severity mapped high→high, medium→moderate, low→low.
- token_length_outliers: advisory — rate > 0.10 → moderate, > 0 → low (capped).
- category_imbalance (worst field by dominant share): share > 0.90 → high,
  > 0.75 → moderate, > 0.50 → low.

Grade rule: F if any critical; else D if any high; else C if any moderate; else B
if any low; else A (no items). An empty dataset is ``N/A`` ("no rows to assess"),
never grade A.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from corpus_studio.quality.basic_quality import QualityReport

NONE = "none"
LOW = "low"
MODERATE = "moderate"
HIGH = "high"
CRITICAL = "critical"

_SEVERITY_ORDER = {NONE: 0, LOW: 1, MODERATE: 2, HIGH: 3, CRITICAL: 4}


class DebtItem(BaseModel):
    category: str
    severity: str  # low | moderate | high | critical (never 'none' once emitted)
    count: int
    # None where a rate is meaningless (secret/PII presence, imbalance share).
    rate: float | None = None
    message: str
    remediation: str


class DebtReport(BaseModel):
    example_count: int
    has_data: bool  # False for 0 rows -> "no rows to assess", NOT grade A
    grade: str      # A | B | C | D | F, or 'N/A' when not has_data
    items: list[DebtItem] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.has_data and not self.items


def _rate_severity(rate: float, moderate_above: float, high_above: float) -> str:
    if rate > high_above:
        return HIGH
    if rate > moderate_above:
        return MODERATE
    if rate > 0:
        return LOW
    return NONE


def _max_synthetic_severity(severities: list[str]) -> str:
    mapping = {"high": HIGH, "medium": MODERATE, "warn": MODERATE, "low": LOW}
    best = LOW
    for raw in severities:
        mapped = mapping.get(raw, LOW)
        if _SEVERITY_ORDER[mapped] > _SEVERITY_ORDER[best]:
            best = mapped
    return best


def build_debt_report(quality: QualityReport) -> DebtReport:
    """Aggregate a quality report into a ranked, graded debt ledger (pure)."""

    total = quality.example_count
    if total <= 0:
        return DebtReport(example_count=max(total, 0), has_data=False, grade="N/A", items=[])

    items: list[DebtItem] = []

    def _rate(count: int) -> float:
        return count / total

    # --- rate-based row-quality debts ---------------------------------------
    empty_sev = _rate_severity(_rate(quality.empty_row_count), 0.02, 0.10)
    if empty_sev != NONE:
        items.append(DebtItem(
            category="empty_rows", severity=empty_sev, count=quality.empty_row_count,
            rate=_rate(quality.empty_row_count),
            message=f"{quality.empty_row_count} empty row(s).",
            remediation="Remove or fill empty rows before export.",
        ))

    exact_sev = _rate_severity(_rate(quality.duplicate_exact_count), 0.01, 0.05)
    if exact_sev != NONE:
        items.append(DebtItem(
            category="exact_duplicates", severity=exact_sev, count=quality.duplicate_exact_count,
            rate=_rate(quality.duplicate_exact_count),
            message=f"{quality.duplicate_exact_count} exact-duplicate row(s).",
            remediation="Export with --dedupe to drop exact duplicates.",
        ))

    norm_sev = _rate_severity(_rate(quality.duplicate_normalized_count), 0.01, 0.05)
    if norm_sev != NONE:
        items.append(DebtItem(
            category="normalized_duplicates", severity=norm_sev,
            count=quality.duplicate_normalized_count,
            rate=_rate(quality.duplicate_normalized_count),
            message=f"{quality.duplicate_normalized_count} near-duplicate (normalized) row(s).",
            remediation="Export with --dedupe (normalized) or review near-duplicates.",
        ))

    low_info_sev = _rate_severity(_rate(quality.low_information_count), 0.02, 0.10)
    if low_info_sev != NONE:
        items.append(DebtItem(
            category="low_information", severity=low_info_sev, count=quality.low_information_count,
            rate=_rate(quality.low_information_count),
            message=f"{quality.low_information_count} low-information row(s) "
                    f"(< {quality.low_information_token_threshold} tokens).",
            remediation="Export with --drop-low-information, or edit sparse rows.",
        ))

    # --- PII / secrets: PRESENCE-based, never normalized by rate ------------
    high_pii = [f for f in quality.pii_findings if f.severity == "high"]
    medium_pii = [f for f in quality.pii_findings if f.severity == "medium"]
    if high_pii:
        items.append(DebtItem(
            category="secrets", severity=CRITICAL, count=len(high_pii), rate=None,
            message=f"{len(high_pii)} high-severity secret finding(s) (keys/tokens/JWTs).",
            remediation="Redact or remove secrets before training; never ship credentials.",
        ))
    elif medium_pii:
        items.append(DebtItem(
            category="personal_data", severity=HIGH, count=len(medium_pii), rate=None,
            message=f"{len(medium_pii)} personal-data finding(s) (emails/SSNs).",
            remediation="Redact or anonymize personal data before training.",
        ))

    # --- synthetic patterns (presence + max issue severity) -----------------
    if quality.synthetic_pattern_count > 0:
        synth_sev = _max_synthetic_severity(
            [issue.severity for issue in quality.synthetic_pattern_issues]
        )
        items.append(DebtItem(
            category="synthetic_patterns", severity=synth_sev,
            count=quality.synthetic_pattern_count, rate=None,
            message=f"{quality.synthetic_pattern_count} synthetic-pattern issue(s) "
                    "(templated/repetitive rows).",
            remediation="Diversify AI-generated rows; reduce templated repetition.",
        ))

    # --- token-length outliers (advisory, capped at moderate) ---------------
    outlier_rate = _rate(quality.token_length_outlier_count)
    if outlier_rate > 0:
        items.append(DebtItem(
            category="token_length_outliers",
            severity=MODERATE if outlier_rate > 0.10 else LOW,
            count=quality.token_length_outlier_count, rate=outlier_rate,
            message=f"{quality.token_length_outlier_count} token-length outlier row(s).",
            remediation="Review unusually long or short rows.",
        ))

    # --- category imbalance (worst field by dominant share) -----------------
    if quality.category_imbalances:
        worst = max(quality.category_imbalances, key=lambda c: c.share)
        if worst.share > 0.90:
            imbalance_sev = HIGH
        elif worst.share > 0.75:
            imbalance_sev = MODERATE
        elif worst.share > 0.50:
            imbalance_sev = LOW
        else:
            imbalance_sev = NONE
        if imbalance_sev != NONE:
            pct = round(worst.share * 100, 1)
            items.append(DebtItem(
                category="category_imbalance", severity=imbalance_sev,
                count=worst.distinct_values, rate=None,
                message=f"Field '{worst.field}' is {pct}% '{worst.dominant_value}' "
                        f"({worst.dominant_count}/{worst.total}); {worst.distinct_values} distinct value(s).",
                remediation=f"Add examples for under-represented values of '{worst.field}'.",
            ))

    # Highest severity first; then higher rate (None -> 0); then category for stability.
    items.sort(key=lambda item: (-_SEVERITY_ORDER[item.severity], -(item.rate or 0.0), item.category))
    return DebtReport(example_count=total, has_data=True, grade=_grade(items), items=items)


def _grade(items: list[DebtItem]) -> str:
    severities = {item.severity for item in items}
    if CRITICAL in severities:
        return "F"
    if HIGH in severities:
        return "D"
    if MODERATE in severities:
        return "C"
    if LOW in severities:
        return "B"
    return "A"


def _safe(text: Any) -> str:
    collapsed = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text))
    return re.sub(r"\s+", " ", collapsed).strip()


def _measure(item: DebtItem) -> str:
    if item.rate is not None:
        return f"{item.rate * 100:.1f}% ({item.count})"
    return f"count {item.count}"


def render_debt_report_markdown(report: DebtReport) -> str:
    lines = [f"# Dataset Debt — Grade {report.grade}", ""]
    if not report.has_data:
        lines.append("No rows to assess.")
        return "\n".join(lines)
    if not report.items:
        lines.append("No debt detected — grade A. The dataset is clean by the current checks.")
        return "\n".join(lines)

    counts: dict[str, int] = {}
    for item in report.items:
        counts[item.severity] = counts.get(item.severity, 0) + 1
    breakdown = ", ".join(
        f"{counts[sev]} {sev}" for sev in (CRITICAL, HIGH, MODERATE, LOW) if sev in counts
    )
    lines.append(
        f"{len(report.items)} debt item(s): {breakdown}. Pay down the highest severity first."
    )
    lines.append("")
    for item in report.items:
        lines.append(
            f"- **[{item.severity.upper()}]** {_safe(item.category)} — {_safe(item.message)} "
            f"({_measure(item)}). Fix: {_safe(item.remediation)}"
        )
    return "\n".join(lines)
