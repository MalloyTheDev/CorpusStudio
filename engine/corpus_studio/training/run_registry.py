"""Durable, project-local training run records (v0.8 Training Run Registry).

The desktop owns the trainer process and writes these records directly (no
subprocess on the crash path). The engine owns the schema + storage helpers +
headless listing, and provides crash reconciliation: a run left in ``running``
is *unconfirmed* — a reader that finds its pid dead reconciles it to
``interrupted`` rather than trusting a status the writer never got to finalize.

Records are per-run inspectable JSON under ``training_runs/`` because status is
mutable (a JSONL append log is wrong for mutable state). ``run_id`` is
timestamp-prefixed so listing is chronological without an index file.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

# ResumeLineage is a torch-free platform contract; it must be a real (not TYPE_CHECKING) import because
# it is a pydantic field type resolved at class-definition time.
from corpus_studio.platform.contracts import ResumeLineage
from corpus_studio.training.provenance import RunProvenance

if TYPE_CHECKING:
    from corpus_studio.platform.contracts import RunPlan

RUN_REGISTRY_DIRNAME = "training_runs"

PREPARED = "prepared"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"

RUN_STATUSES = frozenset({PREPARED, RUNNING, SUCCEEDED, FAILED, CANCELLED, INTERRUPTED})
TERMINAL_STATUSES = frozenset({SUCCEEDED, FAILED, CANCELLED, INTERRUPTED})


class TrainingRunRecord(BaseModel):
    run_id: str
    created_at: str
    updated_at: str
    status: str = PREPARED
    target: str = ""
    base_model: str = ""
    config_path: str = ""
    output_dir: str = ""
    argv: list[str] = Field(default_factory=list)
    pid: int | None = None
    # Process identity so a recycled pid is not mistaken for a live run.
    process_started_at: str | None = None
    exit_code: int | None = None
    checkpoints: list[str] = Field(default_factory=list)
    before_eval_path: str | None = None
    after_eval_path: str | None = None
    # Provenance: the model/adapter the after-eval targeted. The regression gate
    # (v0.8.1) must not trust a before/after comparison whose after-eval did not
    # run against the model this run produced.
    after_eval_model: str | None = None
    # Back-link to the dataset version (v1.0) captured for this run, if any. A
    # tolerant default so pre-v1.0 records without the field still load as None.
    source_snapshot_id: str | None = None
    # Reproducibility manifest (dataset fingerprint / config hash / engine+platform)
    # captured at run start. Tolerant default so older records load as None.
    provenance: RunProvenance | None = None
    # Exact resume lineage (#440/#486) when this run was prepared as a resume of a parent checkpoint:
    # the parent run/checkpoint identity + the optimizer step it continues from. Tolerant default so an
    # ordinary (non-resumed) run loads as None.
    resume_lineage: ResumeLineage | None = None
    notes: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_unconfirmed(self) -> bool:
        """A ``running`` record whose liveness a reader must confirm via pid."""

        return self.status == RUNNING


_VALID_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _slug(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("_") or "run"


def pid_alive(pid: int | None) -> bool:
    """Best-effort process liveness. POSIX probes via signal 0; on Windows
    (where ``os.kill(pid, 0)`` would terminate the process) it conservatively
    returns True — the desktop reconciles Windows runs via the OS process table.
    """

    if pid is None:
        return False
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return True  # unknown; do not flip on ambiguity


def mint_run_id(timestamp_compact: str, suffix: str) -> str:
    """Chronologically-sortable id, e.g. '20260702T183000-ab12cd'."""

    return f"{timestamp_compact}-{suffix}"


def registry_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / RUN_REGISTRY_DIRNAME


def record_path(project_dir: Path | str, run_id: str) -> Path:
    return registry_dir(project_dir) / f"{_slug(run_id)}.json"


def save_run_record(project_dir: Path | str, record: TrainingRunRecord) -> Path:
    """Atomically write a run record (temp + replace).

    ``run_id`` must match ``[A-Za-z0-9._-]+`` so the slugged filename is injective
    (distinct ids can never collapse to the same file and silently overwrite).
    """

    if not _VALID_RUN_ID.match(record.run_id):
        raise ValueError(
            f"Invalid run_id '{record.run_id}': must match [A-Za-z0-9._-]+."
        )

    directory = registry_dir(project_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(record.run_id)}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_run_record(path: Path | str) -> TrainingRunRecord:
    return TrainingRunRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))


def prepare_resumed_run(
    project_dir: Path | str,
    plan: RunPlan,
    checkpoint_dir: Path | str,
    *,
    resumed_run_id: str,
    now: str,
) -> TrainingRunRecord:
    """Admit a checkpoint as a compatible resume source for ``plan``, then persist a PREPARED run record
    for a FRESH resumed run carrying the verified :class:`ResumeLineage`.

    This is the control-plane *preparation* half of resume (#440/#486): it verifies + records the
    resume but never executes it - the worker trainer consuming the resume request is a separate, gated
    slice. Fails closed via :func:`admit_resume` on any partial / corrupt / incomplete / externally
    changed / incompatible checkpoint, or if ``resumed_run_id`` reuses the parent's run id; no record is
    written unless the checkpoint fully admits.
    """

    # admit_resume is torch-free; imported lazily so importing this storage module stays light.
    from corpus_studio.platform.checkpoint import admit_resume  # noqa: PLC0415

    lineage = admit_resume(plan, checkpoint_dir, resumed_run_id=resumed_run_id)
    # admit_resume (via bound_identities_from_plan) already fails closed unless resolved_execution is
    # present, so this asserts the guaranteed invariant (narrowing it for the type checker) rather than
    # silently writing an empty base_model that would hide a misconfiguration.
    execution = plan.resolved_execution
    assert execution is not None
    record = TrainingRunRecord(
        run_id=resumed_run_id,
        created_at=now,
        updated_at=now,
        status=PREPARED,
        base_model=execution.inputs.model.ref.id,
        resume_lineage=lineage,
        notes="resume-prepared; worker resume execution is a separate gated slice",
    )
    save_run_record(project_dir, record)
    return record


def list_run_records(project_dir: Path | str) -> list[TrainingRunRecord]:
    """All records, newest first (run_id is chronological). Unreadable files skipped."""

    directory = registry_dir(project_dir)
    if not directory.exists():
        return []
    records: list[TrainingRunRecord] = []
    for path in directory.glob("*.json"):
        try:
            records.append(load_run_record(path))
        except Exception:  # noqa: BLE001 - a corrupt record must not break listing.
            continue
    records.sort(key=lambda record: record.run_id, reverse=True)
    return records


def validate_transition(old_status: str, new_status: str) -> None:
    """Reject only truly-impossible transitions (terminal -> different status).

    Messy real transitions (running -> failed without an exit code, etc.) are
    allowed; only leaving a terminal state is forbidden.
    """

    if new_status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status '{new_status}'.")
    if old_status in TERMINAL_STATUSES and new_status != old_status:
        raise ValueError(f"Run is already {old_status}; cannot change to {new_status}.")


def reconcile_running_records(
    records: list[TrainingRunRecord],
    is_alive: Callable[[int], bool],
    updated_at: str,
) -> list[TrainingRunRecord]:
    """Flip any ``running`` record whose pid is no longer alive to ``interrupted``.

    ``is_alive`` is injected (the desktop checks the OS process table); a record
    with no pid but ``running`` status is also treated as interrupted.
    """

    reconciled: list[TrainingRunRecord] = []
    for record in records:
        if record.status == RUNNING and (record.pid is None or not is_alive(record.pid)):
            reconciled.append(
                record.model_copy(
                    update={
                        "status": INTERRUPTED,
                        "updated_at": updated_at,
                        "notes": (record.notes + " " if record.notes else "")
                        + "reconciled: process not alive on load",
                    }
                )
            )
        else:
            reconciled.append(record)
    return reconciled
