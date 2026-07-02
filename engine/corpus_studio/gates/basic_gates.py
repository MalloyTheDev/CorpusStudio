"""Basic gates wired to the existing validation/quality/leakage/eval logic.

Each gate is a pure function from an already-computed input to a GateResult, so
gates are easy to test and reuse. The runner computes the inputs.
"""

from __future__ import annotations

from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.gates.models import GateResult, GateScope, GateStatus, GateThresholds
from corpus_studio.quality.basic_quality import QualityReport
from corpus_studio.splitters.leakage import SplitLeakageReport
from corpus_studio.validators.results import ValidationReport


def input_present_gate(
    row_count: int, scope: GateScope, block_when_empty: bool = False
) -> GateResult:
    """Flag an empty input so it cannot pass every other gate silently."""

    if row_count > 0:
        status = GateStatus.PASS
    else:
        status = GateStatus.BLOCK if block_when_empty else GateStatus.WARN
    return GateResult(
        gate_id="input_present",
        name="Input present",
        scope=scope,
        status=status,
        observed=f"{row_count} row(s)",
        expected=">= 1 row",
        message="Input has rows." if row_count else "Input is empty.",
        repair=None if row_count else "Add or import rows before gating.",
    )


def schema_gate(report: ValidationReport, scope: GateScope = GateScope.DATASET) -> GateResult:
    error_count = len(report.errors)
    failing_rows = len(
        {issue.row_number for issue in report.errors if issue.row_number is not None}
    )
    status = GateStatus.PASS if report.valid else GateStatus.BLOCK
    affected = sorted(
        {str(issue.row_number) for issue in report.errors if issue.row_number is not None}
    )
    return GateResult(
        gate_id="schema",
        name="Schema validation",
        scope=scope,
        status=status,
        observed=f"{error_count} validation error(s) across {report.checked_rows} row(s)",
        expected="0 validation errors",
        affected=affected,
        message="All rows validate against the schema."
        if report.valid
        else f"{failing_rows} row(s) fail schema validation.",
        repair=None if report.valid else "Fix or remove invalid rows before continuing.",
    )


def quality_gate(
    report: QualityReport,
    thresholds: GateThresholds,
    scope: GateScope = GateScope.DATASET,
) -> GateResult:
    statuses = [GateStatus.PASS]
    messages: list[str] = []
    if report.duplicate_exact_count > thresholds.max_exact_duplicates:
        statuses.append(
            GateStatus.BLOCK if thresholds.block_exact_duplicates else GateStatus.WARN
        )
        messages.append(f"{report.duplicate_exact_count} exact duplicate row(s)")
    if report.duplicate_normalized_count > thresholds.max_normalized_duplicates:
        statuses.append(GateStatus.WARN)
        messages.append(f"{report.duplicate_normalized_count} near-duplicate row(s)")
    if report.low_information_count > thresholds.max_low_information:
        statuses.append(GateStatus.WARN)
        messages.append(f"{report.low_information_count} low-information row(s)")
    if report.synthetic_pattern_count >= thresholds.warn_synthetic_pattern_issues:
        statuses.append(GateStatus.WARN)
        messages.append(f"{report.synthetic_pattern_count} synthetic-pattern issue(s)")

    from corpus_studio.gates.models import worst_status

    status = worst_status(statuses)
    return GateResult(
        gate_id="quality",
        name="Quality thresholds",
        scope=scope,
        status=status,
        observed="; ".join(messages) if messages else "no quality issues over threshold",
        expected=(
            f"<= {thresholds.max_exact_duplicates} exact dup, "
            f"<= {thresholds.max_normalized_duplicates} near-dup, "
            f"<= {thresholds.max_low_information} low-info"
        ),
        message="Quality within thresholds."
        if status == GateStatus.PASS
        else "; ".join(messages) + ".",
        repair=None
        if status == GateStatus.PASS
        else "Dedupe and rewrite flagged rows, or export with the cleaning pass.",
    )


def leakage_gate(report: SplitLeakageReport, scope: GateScope = GateScope.SPLIT) -> GateResult:
    leaked = report.rows_shared_across_splits
    status = GateStatus.BLOCK if leaked > 0 else GateStatus.PASS
    # SplitLeakage carries no row ids, so keep `affected` empty (row-id typed,
    # like the other gates) and surface sample text in the message instead.
    samples = "; ".join(leak.sample for leak in report.leaks[:2])
    message = (
        "No leakage across splits."
        if status == GateStatus.PASS
        else f"{leaked} duplicate/near-duplicate row(s) leak across splits"
        + (f" (e.g. {samples})" if samples else "")
        + "."
    )
    return GateResult(
        gate_id="leakage",
        name="Train/validation/test leakage",
        scope=scope,
        status=status,
        observed=f"{leaked} row(s) shared across splits in {report.leaked_group_count} group(s)",
        expected="0 rows shared across splits",
        affected=[],
        message=message,
        repair=None
        if status == GateStatus.PASS
        else "Deduplicate before splitting so copies do not span train and test.",
    )


def pii_gate(
    report: QualityReport,
    thresholds: GateThresholds,
    scope: GateScope = GateScope.DATASET,
) -> GateResult:
    high = [finding for finding in report.pii_findings if finding.severity == "high"]
    medium = [finding for finding in report.pii_findings if finding.severity == "medium"]

    if high and thresholds.block_on_high_severity_pii:
        status = GateStatus.BLOCK
    elif medium and thresholds.warn_on_medium_severity_pii:
        status = GateStatus.WARN
    elif high or medium:
        status = GateStatus.WARN
    else:
        status = GateStatus.PASS

    kinds = sorted({finding.kind for finding in report.pii_findings})
    affected = sorted(
        {str(row) for finding in report.pii_findings for row in finding.row_numbers}
    )
    return GateResult(
        gate_id="pii",
        name="PII / secret leakage",
        scope=scope,
        status=status,
        observed=f"{len(high)} high, {len(medium)} medium finding(s): {', '.join(kinds) or 'none'}",
        expected="no secrets; no high-severity PII",
        affected=affected,
        message="No PII/secret findings."
        if status == GateStatus.PASS
        else f"Found {', '.join(kinds)}.",
        repair=None
        if status == GateStatus.PASS
        else "Remove keys/tokens and redact personal data before continuing.",
    )


def artifact_integrity_gate(
    integrity: str, scope: GateScope = GateScope.MODEL_ARTIFACT
) -> GateResult:
    """Block promotion when the weights changed or vanished since evaluation."""

    ok = integrity == "ok"
    return GateResult(
        gate_id="integrity",
        name="Artifact integrity",
        scope=scope,
        status=GateStatus.PASS if ok else GateStatus.BLOCK,
        observed=f"integrity={integrity}",
        expected="integrity=ok (weights unchanged since evaluation)",
        message="Artifact weights are intact."
        if ok
        else f"Artifact weights are {integrity} (changed or gone since evaluation); do not promote.",
        repair=None if ok else "Re-evaluate the current weights, then re-register and re-gate.",
    )


def regression_gate(
    before: EvaluationReport | None,
    after: EvaluationReport | None,
    thresholds: GateThresholds,
    provenance_ok: bool,
    scope: GateScope = GateScope.TRAINING_RUN,
) -> GateResult:
    """Block when the trained model regressed vs the baseline.

    Trust depends on provenance: if the after-eval targeted the base model (or no
    model id was linked), the comparison is not trustworthy and the gate WARNs
    with 'unverified linkage' rather than claiming a pass/block.
    """

    if before is None or after is None:
        return GateResult(
            gate_id="regression",
            name="Training regression",
            scope=scope,
            status=GateStatus.WARN,
            observed="before and/or after evaluation is missing or could not be read",
            expected="a baseline eval and an eval of the trained model, both linked and readable",
            message="Cannot gate regression: a before/after evaluation is not linked, "
            "or the linked report could not be read.",
            repair="Evaluate the base model and the trained model and link both; "
            "if a report is already linked, repair or regenerate the unreadable file.",
        )

    delta = round(after.average_score - before.average_score, 2)
    observed = (
        f"after {after.average_score:.1f} vs before {before.average_score:.1f} (Δ{delta:+.1f})"
    )

    regressed = delta < -thresholds.max_regression_score_drop

    if not provenance_ok:
        # An unverified comparison that is DOWN is at best equal-or-worse, so a
        # real drop still blocks; an unverified improvement can't be trusted (warn).
        if regressed:
            return GateResult(
                gate_id="regression",
                name="Training regression",
                scope=scope,
                status=GateStatus.BLOCK,
                observed=observed,
                expected="the after-eval must target the trained model, not the base model",
                message=f"Trained model regressed (score dropped {abs(delta):.1f}) AND the linkage "
                "is unverified — the after-eval may target the base model; do not promote.",
                repair="Evaluate the trained adapter (not the base model), re-link, and re-check.",
            )
        return GateResult(
            gate_id="regression",
            name="Training regression",
            scope=scope,
            status=GateStatus.WARN,
            observed=observed,
            expected="the after-eval must target the trained model, not the base model",
            message="Unverified linkage: the after-eval appears to target the base model "
            "(or its model id is missing), so this before/after comparison is not trustworthy.",
            repair="Evaluate the trained adapter (not the base model) and re-link the after-eval.",
        )

    if regressed:
        return GateResult(
            gate_id="regression",
            name="Training regression",
            scope=scope,
            status=GateStatus.BLOCK,
            observed=observed,
            expected=f"after >= before - {thresholds.max_regression_score_drop:.1f}",
            message=f"Trained model regressed: average score dropped {abs(delta):.1f} "
            f"(tolerance {thresholds.max_regression_score_drop:.1f}).",
            repair="Keep the base model, or retrain/adjust before promoting this run.",
        )

    trend = "improved" if delta > 0 else "held within tolerance"
    return GateResult(
        gate_id="regression",
        name="Training regression",
        scope=scope,
        status=GateStatus.PASS,
        observed=observed,
        expected=f"after >= before - {thresholds.max_regression_score_drop:.1f}",
        message=f"No regression: average score {trend} (Δ{delta:+.1f}).",
    )


def eval_score_gate(
    report: EvaluationReport,
    thresholds: GateThresholds,
    scope: GateScope = GateScope.EVALUATION_REPORT,
) -> GateResult:
    tested = report.examples_tested
    pass_rate = (tested - report.failed_examples) / tested if tested else 0.0
    below_score = report.average_score < thresholds.min_eval_average_score
    below_pass = pass_rate < thresholds.min_eval_pass_rate
    status = GateStatus.BLOCK if (below_score or below_pass) else GateStatus.PASS
    return GateResult(
        gate_id="eval_score",
        name="Evaluation score",
        scope=scope,
        status=status,
        observed=f"avg {report.average_score:.1f}, pass rate {pass_rate:.0%} over {tested} example(s)",
        expected=(
            f"avg >= {thresholds.min_eval_average_score:.0f} and "
            f"pass rate >= {thresholds.min_eval_pass_rate:.0%}"
        ),
        message="Evaluation meets thresholds."
        if status == GateStatus.PASS
        else "Evaluation below score/pass-rate thresholds.",
        repair=None
        if status == GateStatus.PASS
        else "Improve or filter failing examples, or retrain, before promoting this model.",
    )
