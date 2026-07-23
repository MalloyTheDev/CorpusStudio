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

import json
import os
from pathlib import Path
from typing import Any

from loop.controller import Phase, LoopState

SCHEMA_VERSION = 1

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


def to_dict(state: LoopState) -> dict[str, Any]:
    """Serialise a LoopState to a JSON-native dict (the Phase enum becomes its string value)."""
    data: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "current_phase": state.current_phase.value}
    for name in _PASSTHROUGH_FIELDS:
        data[name] = getattr(state, name)
    return data


def from_dict(data: Any) -> LoopState:
    """Reconstruct a LoopState from a parsed document, validating types and the phase enum. Fail-closed:
    any structural or type problem raises :class:`LoopStateError` rather than defaulting silently."""
    if not isinstance(data, dict):
        raise LoopStateError(f"loop state document is not an object (got {type(data).__name__})")

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
        kwargs[name] = value
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
