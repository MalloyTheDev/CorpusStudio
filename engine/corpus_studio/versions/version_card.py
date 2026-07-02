"""Dataset version card — a live projection of a dataset version (never stored).

Rendered on demand from the version record plus the current state of everything
it links to (dataset fingerprint, source runs, artifacts, eval/gate reports), so
it can never drift from the underlying state. Honesty flags lead: if the live
dataset has drifted from the recorded fingerprint, or a linked artifact/report
is gone, the card says so *first* rather than presenting stale lineage as fact.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from corpus_studio.versions.version_registry import (
    DatasetVersionRecord,
    integrity_from_fingerprints,
)


def _safe(text: Any) -> str:
    """Neutralize newlines/control chars so an untrusted field (label, id, path)
    cannot inject extra Markdown lines/blockquotes into the card."""

    collapsed = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text))
    return re.sub(r"\s+", " ", collapsed).strip()


class VersionRunLink(BaseModel):
    run_id: str
    present: bool
    status: str = ""
    base_model: str = ""


class VersionArtifactLink(BaseModel):
    artifact_id: str
    present: bool
    status: str = ""
    integrity: str = ""


class DatasetVersionCard(BaseModel):
    version_id: str
    label: str = ""
    trigger: str = ""
    created_at: str = ""
    updated_at: str = ""
    row_count: int = 0
    content_fingerprint: str | None = None
    fingerprint_algo: str = ""
    row_signature_kind: str = ""
    current_integrity: str = ""  # matches | drifted | unreadable
    runs: list[VersionRunLink] = Field(default_factory=list)
    artifacts: list[VersionArtifactLink] = Field(default_factory=list)
    eval_report_linked: bool = False
    eval_report_present: bool = False
    eval_average_score: float | None = None
    gate_report_linked: bool = False
    gate_report_present: bool = False
    gate_overall_status: str | None = None
    notes: str = ""
    warnings: list[str] = Field(default_factory=list)


def build_version_card(
    record: DatasetVersionRecord,
    *,
    current_fingerprint: str | None,
    runs_by_id: dict[str, Any] | None = None,
    artifacts_by_id: dict[str, tuple[Any, str]] | None = None,
    load_eval_report: Callable[[str], Any] | None = None,
    load_gate_report: Callable[[str], Any] | None = None,
) -> DatasetVersionCard:
    """Assemble a version card by resolving every link LIVE (pure — no disk I/O).

    Callers resolve the inputs and inject them: ``current_fingerprint`` is the
    dataset hashed now; ``runs_by_id`` maps a linked run_id → its run record;
    ``artifacts_by_id`` maps a linked artifact_id → ``(record, integrity)``;
    ``load_eval_report`` / ``load_gate_report`` load a report from a path (may
    return None). Anything missing becomes an honesty flag, not a crash.
    """

    runs_by_id = runs_by_id or {}
    artifacts_by_id = artifacts_by_id or {}
    warnings: list[str] = []

    integrity = integrity_from_fingerprints(record.content_fingerprint, current_fingerprint)
    if integrity == "drifted":
        warnings.append(
            "The dataset has changed since this version was recorded, so its linked "
            "runs, artifacts, and evaluations may describe a different dataset state."
        )
    elif integrity == "unreadable":
        warnings.append(
            "Cannot verify the current dataset against this version "
            "(dataset missing/unreadable, or no fingerprint was recorded)."
        )

    runs: list[VersionRunLink] = []
    for run_id in record.source_run_ids:
        run = runs_by_id.get(run_id)
        if run is None:
            runs.append(VersionRunLink(run_id=run_id, present=False))
            warnings.append(f"Linked training run '{_safe(run_id)}' was not found.")
        else:
            runs.append(
                VersionRunLink(
                    run_id=run_id,
                    present=True,
                    status=str(getattr(run, "status", "") or ""),
                    base_model=str(getattr(run, "base_model", "") or ""),
                )
            )

    artifacts: list[VersionArtifactLink] = []
    for artifact_id in record.artifact_ids:
        resolved = artifacts_by_id.get(artifact_id)
        if resolved is None:
            artifacts.append(VersionArtifactLink(artifact_id=artifact_id, present=False))
            warnings.append(f"Linked model artifact '{_safe(artifact_id)}' was not found.")
        else:
            artifact, artifact_integrity = resolved
            artifacts.append(
                VersionArtifactLink(
                    artifact_id=artifact_id,
                    present=True,
                    status=str(getattr(artifact, "status", "") or ""),
                    integrity=str(artifact_integrity or ""),
                )
            )
            if artifact_integrity and artifact_integrity != "ok":
                warnings.append(
                    f"Linked artifact '{_safe(artifact_id)}' integrity is "
                    f"{_safe(artifact_integrity)} (weights changed or gone)."
                )

    eval_linked = record.eval_report_path is not None
    eval_present = False
    eval_score: float | None = None
    if eval_linked:
        report = load_eval_report(record.eval_report_path) if load_eval_report else None
        if report is None:
            warnings.append("Linked evaluation report is missing or unreadable.")
        else:
            eval_present = True
            score = getattr(report, "average_score", None)
            eval_score = float(score) if score is not None else None

    gate_linked = record.gate_report_path is not None
    gate_present = False
    gate_status: str | None = None
    if gate_linked:
        report = load_gate_report(record.gate_report_path) if load_gate_report else None
        if report is None:
            warnings.append("Linked gate report is missing or unreadable.")
        else:
            gate_present = True
            status = getattr(report, "overall_status", None)
            gate_status = getattr(status, "value", None) or (str(status) if status is not None else None)

    return DatasetVersionCard(
        version_id=record.version_id,
        label=record.label,
        trigger=record.trigger,
        created_at=record.created_at,
        updated_at=record.updated_at,
        row_count=record.row_count,
        content_fingerprint=record.content_fingerprint,
        fingerprint_algo=record.fingerprint_algo,
        row_signature_kind=record.row_signature_kind,
        current_integrity=integrity,
        runs=runs,
        artifacts=artifacts,
        eval_report_linked=eval_linked,
        eval_report_present=eval_present,
        eval_average_score=eval_score,
        gate_report_linked=gate_linked,
        gate_report_present=gate_present,
        gate_overall_status=gate_status,
        notes=record.notes,
        warnings=warnings,
    )


def _short_fp(fingerprint: str | None) -> str:
    if not fingerprint:
        return "—"
    return fingerprint[:12] + "…" if len(fingerprint) > 12 else fingerprint


def render_version_card_markdown(card: DatasetVersionCard) -> str:
    lines = [
        f"# Dataset Version Card — {_safe(card.version_id)}",
        "",
        f"- **Label**: {_safe(card.label) or '(none)'}",
        f"- **Trigger**: {_safe(card.trigger) or '(unspecified)'}",
        f"- **Rows**: {card.row_count}",
        f"- **Fingerprint**: {_safe(card.fingerprint_algo)}/{_safe(card.row_signature_kind)} "
        f"{_short_fp(card.content_fingerprint)}",
        f"- **Current integrity**: {card.current_integrity}",
    ]

    # Warnings first, so a drifted/missing-link card never leads with stale lineage.
    for warning in card.warnings:
        lines += ["", f"> ⚠ {_safe(warning)}"]

    lines += ["", "## Lineage", ""]

    if card.runs:
        lines.append(f"- **Source runs**: {len(card.runs)}")
        for run in card.runs:
            if run.present:
                base = f", base {_safe(run.base_model)}" if run.base_model else ""
                lines.append(f"  - {_safe(run.run_id)} — {_safe(run.status) or 'unknown'}{base}")
            else:
                lines.append(f"  - {_safe(run.run_id)} — not found")
    else:
        lines.append("- **Source runs**: (none)")

    if card.artifacts:
        lines.append(f"- **Artifacts**: {len(card.artifacts)}")
        for artifact in card.artifacts:
            if artifact.present:
                lines.append(
                    f"  - {_safe(artifact.artifact_id)} — {_safe(artifact.status) or 'unknown'}, "
                    f"integrity {_safe(artifact.integrity) or 'unknown'}"
                )
            else:
                lines.append(f"  - {_safe(artifact.artifact_id)} — not found")
    else:
        lines.append("- **Artifacts**: (none)")

    if not card.eval_report_linked:
        lines.append("- **Eval report**: (none)")
    elif card.eval_report_present:
        score = f"{card.eval_average_score:.1f}" if card.eval_average_score is not None else "—"
        lines.append(f"- **Eval report**: average score {score}")
    else:
        lines.append("- **Eval report**: linked but missing")

    if not card.gate_report_linked:
        lines.append("- **Gate report**: (none)")
    elif card.gate_report_present:
        lines.append(f"- **Gate report**: {_safe(card.gate_overall_status) or 'unknown'}")
    else:
        lines.append("- **Gate report**: linked but missing")

    if card.notes:
        lines += ["", f"_Notes: {_safe(card.notes)}_"]
    lines += ["", f"_Recorded {_safe(card.created_at)}, updated {_safe(card.updated_at)}._"]
    return "\n".join(lines)
