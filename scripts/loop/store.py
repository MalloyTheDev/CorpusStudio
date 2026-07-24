"""Durable persistence for :class:`LoopState` (controller slice 2a).

The controller kernel is a pure in-memory state machine; this module is its durable side - it turns a
:class:`LoopState` into a deterministic on-disk JSON document and back, so a bounded loop survives across
observe cycles (and, later, across processes/sessions) instead of living only in one turn's memory.

Discipline (same as the kernel and the assurance plane):
  * stdlib-only; imports nothing from ``corpus_studio`` and no torch.
  * FAIL-CLOSED reads: a malformed / wrong-typed / bad-enum document raises :class:`LoopStateError`
    (never a silent default or a half-built state). Enum fields are reconstructed and validated - a
    string ``current_phase`` that is not a real :class:`Phase` is a hard error, not a silent RECEIVE_GOAL.
  * DETERMINISTIC writes: ``sort_keys=True`` + ``ensure_ascii=True`` so the same state always serialises
    to the same bytes (diff-friendly, and a stable input to any later digest).
  * ATOMIC writes: write a sibling temp file then ``os.replace`` so a crash mid-write cannot leave a
    truncated state file (single-writer, like the assurance plane's records).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from loop.controller import Phase, LoopState
from loop.locking import FileLock, LockError

SCHEMA_VERSION = 1
# NOTE: a CAS save (save_cas) adds envelope metadata (state_revision / state_digest / previous_state_digest
# / writer_id). from_dict IGNORES unknown keys, so the deterministic single-writer `save` and a versioned
# `save_cas` interoperate: load() reads either, dropping the metadata.

# The LoopState fields carried verbatim as JSON-native values (lists/dicts/str/None). current_phase is
# handled separately (enum <-> str); everything else round-trips as-is.
_PASSTHROUGH_FIELDS = (
    "goal", "goal_id", "success_criteria", "task_graph", "active_agents", "observations",
    "hypotheses", "failed_approaches", "budgets", "assurance_records", "review_state",
    "blockers", "termination_reason",
)
_LIST_FIELDS = frozenset({
    "success_criteria", "task_graph", "active_agents", "observations", "hypotheses",
    "failed_approaches", "assurance_records", "blockers",
})
_DICT_FIELDS = frozenset({"budgets", "review_state"})


class LoopStateError(Exception):
    """A durable LoopState document is malformed or has a bad type/enum (fail-closed)."""


class ConcurrentStateWrite(LoopStateError):
    """A CAS save found the on-disk state at a different revision than the writer loaded - a concurrent
    writer moved it. Fail closed rather than clobber the other writer's update (reload, then re-apply)."""


def to_dict(state: LoopState) -> dict[str, Any]:
    """Serialise a LoopState to a JSON-native dict (the Phase enum becomes its string value)."""
    data: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "current_phase": state.current_phase.value}
    for name in _PASSTHROUGH_FIELDS:
        # Deep-copy the mutable containers so the emitted dict is INDEPENDENT of the live state (a
        # round-trip must not alias the source's lists/dicts - matching the JSON path's semantics).
        data[name] = copy.deepcopy(getattr(state, name))
    return data


def from_dict(data: Any) -> LoopState:
    """Reconstruct a LoopState from a parsed document, validating types and the phase enum. Fail-closed:
    any structural or type problem raises :class:`LoopStateError` rather than defaulting silently."""
    if not isinstance(data, dict):
        raise LoopStateError(f"loop state document is not an object (got {type(data).__name__})")

    version = data.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool) or version > SCHEMA_VERSION:
        # A future / non-int schema version cannot be faithfully represented by this build - fail closed
        # rather than load it and silently drop the fields this version does not know about.
        raise LoopStateError(f"unsupported loop-state schema_version {version!r} (this build supports <= {SCHEMA_VERSION})")

    phase_value = data.get("current_phase", Phase.RECEIVE_GOAL.value)
    try:
        current_phase = Phase(phase_value)
    except ValueError as exc:
        raise LoopStateError(f"invalid current_phase {phase_value!r}") from exc

    kwargs: dict[str, Any] = {"current_phase": current_phase}
    for name in _PASSTHROUGH_FIELDS:
        if name not in data:
            continue  # absent -> the dataclass default_factory / default applies
        value = data[name]
        if name in _LIST_FIELDS and not isinstance(value, list):
            raise LoopStateError(f"field {name!r} must be a list (got {type(value).__name__})")
        if name in _DICT_FIELDS and not isinstance(value, dict):
            raise LoopStateError(f"field {name!r} must be an object (got {type(value).__name__})")
        if name == "goal" and not isinstance(value, str):
            raise LoopStateError(f"field 'goal' must be a string (got {type(value).__name__})")
        if name == "termination_reason" and value is not None and not isinstance(value, str):
            raise LoopStateError("field 'termination_reason' must be a string or null")
        # Deep-copy so the reconstructed state owns independent containers, regardless of the input
        # dict's provenance (a caller may keep + mutate the dict it passed in).
        kwargs[name] = copy.deepcopy(value)
    return LoopState(**kwargs)


def dumps(state: LoopState) -> str:
    """Deterministic JSON text for a LoopState (sorted keys, ASCII) - stable bytes for the same state."""
    return json.dumps(to_dict(state), sort_keys=True, ensure_ascii=True, indent=2) + "\n"


def loads(text: str) -> LoopState:
    """Parse JSON text into a LoopState, fail-closed on invalid JSON or a bad document."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:  # RecursionError: deeply nested -> fail closed, not exit-1
        raise LoopStateError(f"loop state is not valid JSON: {exc}") from exc
    return from_dict(data)


def save(state: LoopState, path: Path) -> None:
    """Atomically write the state to ``path`` (temp sibling + os.replace); creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(dumps(state), encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX + Windows


def load(path: Path) -> LoopState:
    """Read a state file, fail-closed on a missing/unreadable/invalid document."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LoopStateError(f"loop state file could not be read ({path}): {exc}") from exc
    return loads(text)


# --------------------------------------------------------------------------- optimistic concurrency (CAS)
# `save` above is single-writer safe (atomic replace, deterministic bytes). A CONCURRENT writer (a
# multi-process runtime) needs stale-writer detection: read the revision, and only write if it has not
# moved. state_revision / save_cas add that WITHOUT changing the single-writer path.


def _content_digest(state: LoopState) -> str:
    """A sha256 over the DETERMINISTIC state content (the `dumps` bytes, excluding save-metadata) - it
    identifies the state, so a writer can tell whether the state changed under it."""
    return "sha256:" + hashlib.sha256(dumps(state).encode("utf-8")).hexdigest()


def _read_doc(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def state_revision(path: Path) -> int:
    """The persisted ``state_revision`` of a state file (0 if absent / unversioned / unreadable). This is
    the value a concurrent writer passes back to :func:`save_cas` as ``expected_revision``."""
    doc = _read_doc(path)
    rev = doc.get("state_revision") if doc is not None else None
    return rev if isinstance(rev, int) and not isinstance(rev, bool) and rev >= 0 else 0


def save_cas(state: LoopState, path: Path, expected_revision: int, *, writer_id: str = "") -> int:
    """COMPARE-AND-SWAP save for a CONCURRENT writer. Under a cross-process :class:`FileLock`, verify the
    on-disk state is still at ``expected_revision`` (what this writer loaded); if it moved, raise
    :class:`ConcurrentStateWrite` (a stale writer must not clobber the other's update - reload + re-apply).
    On success, atomically write the state at ``expected_revision + 1`` with ``state_digest``,
    ``previous_state_digest`` and ``writer_id``, and return the new revision. The lock makes the whole
    read-check-write atomic across processes, so two writers can never both pass the check and lose an
    update. The single-writer :func:`save` is unchanged (and remains deterministic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(path):
            current = state_revision(path)
            if current != expected_revision:
                raise ConcurrentStateWrite(
                    f"state at {path} is revision {current}, not the expected {expected_revision} "
                    "(a concurrent writer moved it); reload before writing")
            prior = _read_doc(path)
            previous_digest = prior.get("state_digest") if prior is not None else None
            doc = to_dict(state)
            doc["state_revision"] = expected_revision + 1
            doc["writer_id"] = writer_id
            doc["previous_state_digest"] = previous_digest if isinstance(previous_digest, str) else None
            doc["state_digest"] = _content_digest(state)
            tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
            tmp.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
                           encoding="utf-8")
            os.replace(tmp, path)
            return expected_revision + 1
    except LockError as exc:
        raise ConcurrentStateWrite(f"could not lock state {path} for a CAS write: {exc}") from exc
