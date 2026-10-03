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
if any low; else A (no items).

A grade is WITHHELD (``grade is None``, with ``grade_reason`` saying why) rather than guessed
whenever the ledger cannot stand behind one. Three cases:
- an empty dataset has no rows to assess, and is never grade A;
- a dataset whose shape does not support the text signals (an object-detection corpus of
  normalized boxes, a numeric table) has no meaningful letter: the signals that produce a grade
  for a text corpus could not run, so A would read as "clean" and D as "broken" on the strength
  of heuristics that read nothing. Those signals are listed in ``not_assessed`` and contribute
  no debt items;
- no debt found AND the applicability was never measured (no schema). A clean bill of health is a
  POSITIVE claim: finding nothing is not the same as finding nothing wrong, and without a schema
  it is not known whether the signals could read the data at all. Items that DID fire are their
  own evidence, so an unmeasured B/C/D/F still stands.
Applicability comes from ``QualityReport.applicability``; see
:mod:`corpus_studio.quality.applicability`.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from corpus_studio.quality.applicability import QualityApplicability
from corpus_studio.quality.basic_quality import QualityReport

NONE = "none"
LOW = "low"
MODERATE = "moderate"
HIGH = "high"
CRITICAL = "critical"

_SEVERITY_ORDER = {NONE: 0, LOW: 1, MODERATE: 2, HIGH: 3, CRITICAL: 4}

# The signals a letter grade for a text corpus actually rests on. Where a dataset's shape
# supports none of them, the remaining any-shape signals (empty rows, exact duplicates, secrets)
# are too thin a basis for one health verdict, so the grade is withheld instead of guessed.
_GRADE_BEARING_SIGNALS = frozenset(
    {"normalized_duplicates", "low_information", "synthetic_patterns", "token_length_outliers"}
)


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
    # A | B | C | D | F, or None when no grade can honestly be given (see the module docstring).
    grade: str | None
    # Why the grade is withheld; "" exactly when a grade WAS given.
    grade_reason: str = ""
    items: list[DebtItem] = Field(default_factory=list)
    # Signals this dataset's shape does not support, so they ran no check at all. Their absence
    # from ``items`` is "not measured", never "measured clean".
    not_assessed: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Assessed and free of debt. A withheld grade is never clean: nothing was assessed."""
        return self.has_data and self.grade is not None and not self.items

    @property
    def graded(self) -> bool:
        return self.grade is not None


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
        return DebtReport(
            example_count=max(total, 0),
            has_data=False,
            grade=None,
            grade_reason="no rows to assess",
            items=[],
        )

    applicability = quality.applicability
    not_assessed = list(applicability.not_applicable_signals) if applicability else []

    def _assessed(signal: str) -> bool:
        return signal not in not_assessed

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
    if norm_sev != NONE and _assessed("normalized_duplicates"):
        items.append(DebtItem(
            category="normalized_duplicates", severity=norm_sev,
            count=quality.duplicate_normalized_count,
            rate=_rate(quality.duplicate_normalized_count),
            message=f"{quality.duplicate_normalized_count} near-duplicate (normalized) row(s).",
            remediation="Export with --dedupe (normalized) or review near-duplicates.",
        ))

    low_info_sev = _rate_severity(_rate(quality.low_information_count), 0.02, 0.10)
    if low_info_sev != NONE and _assessed("low_information"):
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
    if quality.synthetic_pattern_count > 0 and _assessed("synthetic_patterns"):
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
    if outlier_rate > 0 and _assessed("token_length_outliers"):
        items.append(DebtItem(
            category="token_length_outliers",
            severity=MODERATE if outlier_rate > 0.10 else LOW,
            count=quality.token_length_outlier_count, rate=outlier_rate,
            message=f"{quality.token_length_outlier_count} token-length outlier row(s).",
            remediation="Review unusually long or short rows.",
        ))

    # --- category imbalance (worst field by dominant share) -----------------
    if quality.category_imbalances and _assessed("category_imbalance"):
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
    withheld = _withheld_reason(applicability, items)
    return DebtReport(
        example_count=total,
        has_data=True,
        grade=None if withheld else _grade(items),
        grade_reason=withheld,
        items=items,
        not_assessed=sorted(not_assessed),
    )


def _withheld_reason(
    applicability: QualityApplicability | None,
    items: list[DebtItem],
) -> str:
    """Why no letter grade can be given, or "" when one can.

    Two withholding rules, both about not making a claim the ledger cannot support:

    * a MEASURED applicability saying the grade-bearing text signals could not run. Those are the
      signals a text corpus's letter rests on, so without them a letter reports the heuristics'
      silence as health - in either direction, an A reading as clean or a D as broken.
    * an UNMEASURED applicability (no schema) with NO debt items. A clean bill of health is a
      POSITIVE claim, and it needs to be known that the checks could read this data at all;
      finding nothing is not the same as finding nothing wrong. Debt items that DID fire are
      their own evidence, so an unmeasured B/C/D/F still stands and is not withheld.
    """
    if applicability is None or not applicability.measured:
        if items:
            return ""
        return (
            "no debt was found, but whether these text signals can read this data shape was not assessed "
            "(no schema supplied), and a signal that cannot read a field reports nothing rather than "
            "nothing wrong: pass --schema (with --project-dir for a project-local schema) to grade it"
        )
    ungraded = sorted(set(applicability.not_applicable_signals) & _GRADE_BEARING_SIGNALS)
    if not ungraded:
        return ""
    share = applicability.assessed_content_share
    measured = "unmeasured" if share is None else f"{share * 100:.1f}%"
    return (
        f"no applicable text signals for this schema: {', '.join(ungraded)} could not assess "
        f"this data shape (free text covers {measured} of the measured content, floor "
        f"{applicability.min_assessed_content_share * 100:.0f}%), so a letter grade would "
        f"report their silence as health"
    )


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
    heading = f"Grade {report.grade}" if report.graded else "Grade withheld"
    lines = [f"# Dataset Debt - {heading}", ""]
    if not report.has_data:
        lines.append("No rows to assess.")
        return "\n".join(lines)
    if not report.graded:
        lines.append(f"No grade: {_safe(report.grade_reason)}.")
        lines.extend(_not_assessed_lines(report))
        if report.items:
            lines.append("")
            lines.append("The signals that DO apply to this shape found:")
            lines.extend(_item_lines(report))
        return "\n".join(lines)
    if not report.items:
        lines.append("No debt detected - grade A. The dataset is clean by the current checks.")
        lines.extend(_not_assessed_lines(report))
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
    lines.extend(_item_lines(report))
    lines.extend(_not_assessed_lines(report))
    return "\n".join(lines)


def _item_lines(report: DebtReport) -> list[str]:
    return [
        f"- **[{item.severity.upper()}]** {_safe(item.category)} - {_safe(item.message)} "
        f"({_measure(item)}). Fix: {_safe(item.remediation)}"
        for item in report.items
    ]


def _not_assessed_lines(report: DebtReport) -> list[str]:
    """A named list of the signals that ran no check, so a short ledger is never read as a clean
    bill of health for checks that never happened."""
    if not report.not_assessed:
        return []
    names = ", ".join(_safe(signal) for signal in report.not_assessed)
    return ["", f"Not assessed on this shape: {names}."]
