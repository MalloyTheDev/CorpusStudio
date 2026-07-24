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


class CorruptStateFile(LoopStateError):
    """A state file EXISTS but is unreadable / not valid JSON / wrong-shaped. Distinct from an ABSENT file
    (which is a normal first-write): a CAS writer must NOT silently overwrite corruption as if it were a
    fresh file (revision 0), nor misreport it as a concurrent-writer conflict."""


_ABSENT = object()  # sentinel returned by _read_doc for a genuinely non-existent file (vs corrupt -> None)


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


def _atomic_write(path: Path, text: str) -> None:
    """Durably + atomically write ``text`` to ``path``. A temp sibling is written and ``fsync``'d (so its
    bytes are on disk), then ``os.replace``'d (atomic - no torn/truncated file), then the parent directory
    is ``fsync``'d (so the rename itself survives a power-loss crash rather than silently rolling back to
    the prior revision). Directory fsync is best-effort (some platforms disallow it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    try:
        os.write(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)  # atomic on POSIX + Windows
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return  # cannot open the directory to fsync it (e.g. Windows) - the replace is still atomic
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def save(state: LoopState, path: Path) -> None:
    """Durably + atomically write the state to ``path`` (fsync'd temp + os.replace + dir fsync); creates
    parent dirs. A crash cannot leave a torn/truncated file, nor silently roll back the last transition."""
    _atomic_write(path, dumps(state))


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


def _read_doc(path: Path) -> Any:
    """Return the parsed dict, the ``_ABSENT`` sentinel (the file does not exist - a normal first-write),
    or ``None`` (the file EXISTS but is unreadable / not valid JSON / wrong-shaped - i.e. CORRUPT). The
    absent-vs-corrupt distinction is what keeps a CAS writer from overwriting corruption as a fresh file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _ABSENT
    except (OSError, UnicodeDecodeError):
        return None  # present but unreadable -> corrupt, NOT absent
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None  # present but not valid JSON -> corrupt
    return data if isinstance(data, dict) else None  # present but wrong shape -> corrupt


def state_revision(path: Path) -> int:
    """The persisted ``state_revision`` of a state file: 0 for an ABSENT file (a first write is revision 1),
    or the stored revision. A present-but-CORRUPT file raises :class:`CorruptStateFile` (a CAS writer must
    not silently treat corruption as revision 0 and overwrite it, nor misreport it as a concurrent write).
    This is the value a concurrent writer passes back to :func:`save_cas` as ``expected_revision``."""
    doc = _read_doc(path)
    if doc is _ABSENT:
        return 0
    if not isinstance(doc, dict):
        raise CorruptStateFile(f"state file {path} exists but is unreadable/corrupt; refusing to write over it")
    rev = doc.get("state_revision")
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
            previous_digest = prior.get("state_digest") if isinstance(prior, dict) else None
            doc = to_dict(state)
            doc["state_revision"] = expected_revision + 1
            doc["writer_id"] = writer_id
            doc["previous_state_digest"] = previous_digest if isinstance(previous_digest, str) else None
            doc["state_digest"] = _content_digest(state)
            _atomic_write(path, json.dumps(doc, sort_keys=True, ensure_ascii=True, indent=2) + "\n")
            return expected_revision + 1
    except LockError as exc:
        raise ConcurrentStateWrite(f"could not lock state {path} for a CAS write: {exc}") from exc
