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

from pydantic import BaseModel, Field

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
