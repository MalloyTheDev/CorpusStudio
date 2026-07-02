"""Gate foundation: pass/warn/block decisions over existing engine logic.

A gate decides whether a dataset, split, export, or evaluation report may move
forward. Gates reuse the existing validation, quality, leakage, PII, and
evaluation logic — they only aggregate results against thresholds and produce a
serializable, project-local report.
"""

from corpus_studio.gates.models import (
    GateReport,
    GateResult,
    GateScope,
    GateStatus,
    GateThresholds,
)
from corpus_studio.gates.runner import (
    run_dataset_gates,
    run_evaluation_gate,
    run_export_gates,
    run_split_gate,
)

__all__ = [
    "GateReport",
    "GateResult",
    "GateScope",
    "GateStatus",
    "GateThresholds",
    "run_dataset_gates",
    "run_evaluation_gate",
    "run_export_gates",
    "run_split_gate",
]
