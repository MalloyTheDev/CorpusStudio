"""Tests for durable LoopState persistence (scripts/loop/store.py, controller slice 2a).

Pins the round-trip fidelity, the FAIL-CLOSED reads (bad enum / type / JSON), deterministic bytes, and
atomic save/load - so a persisted loop resumes faithfully and a corrupt file never silently degrades.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import LoopState, Observation, Phase, apply  # noqa: E402
from loop.store import LoopStateError, dumps, from_dict, load, loads, save, to_dict  # noqa: E402


def _mid_loop_state() -> LoopState:
    # A non-trivial state: drive a couple of transitions so lists/budget are populated.
    state = LoopState(goal="ship slice 2", goal_id="g-1", success_criteria=["gate green"],
                      current_phase=Phase.DIAGNOSE)
    apply(state, Observation.TEST_REGRESSION, fingerprint="sha256:abc", note="first try")
    state.hypotheses.append({"id": "h1", "text": "off-by-one"})
    return state


def test_round_trip_preserves_every_field_including_the_phase_enum() -> None:
    state = _mid_loop_state()
    restored = from_dict(to_dict(state))
    assert restored.current_phase is state.current_phase  # reconstructed to the Phase enum, not a str
    assert isinstance(restored.current_phase, Phase)
    for name in ("goal", "goal_id", "success_criteria", "observations", "failed_approaches",
                 "budgets", "hypotheses", "termination_reason"):
        assert getattr(restored, name) == getattr(state, name), name


def test_dumps_is_deterministic() -> None:
    state = _mid_loop_state()
    assert dumps(state) == dumps(state)
    assert dumps(state) == dumps(from_dict(to_dict(state)))  # round-trip is byte-stable


def test_from_dict_fails_closed_on_bad_phase() -> None:
    with pytest.raises(LoopStateError, match="current_phase"):
        from_dict({"current_phase": "NOT_A_PHASE"})


def test_from_dict_fails_closed_on_non_object_and_bad_types() -> None:
    with pytest.raises(LoopStateError, match="not an object"):
        from_dict(["not", "a", "dict"])
    with pytest.raises(LoopStateError, match="must be a list"):
        from_dict({"current_phase": "EXECUTE", "observations": "should-be-a-list"})
    with pytest.raises(LoopStateError, match="must be an object"):
        from_dict({"current_phase": "EXECUTE", "budgets": []})
    with pytest.raises(LoopStateError, match="termination_reason"):
        from_dict({"current_phase": "STOPPED", "termination_reason": 123})


def test_loads_fails_closed_on_invalid_json() -> None:
    with pytest.raises(LoopStateError, match="valid JSON"):
        loads("{not json")
    with pytest.raises(LoopStateError):
        loads("[" * 20000 + "]" * 20000)  # deeply nested -> RecursionError -> fail closed, not a traceback


def test_absent_fields_take_dataclass_defaults() -> None:
    state = from_dict({"current_phase": "RECON"})
    assert state.current_phase is Phase.RECON
    assert state.observations == [] and state.budgets.get("max_attempts") == 20


def test_save_load_round_trip_and_atomic(tmp_path: Path) -> None:
    state = _mid_loop_state()
    path = tmp_path / "loops" / "state.json"
    save(state, path)  # creates parent dirs
    assert path.is_file()
    # no leftover temp files from the atomic write
    assert list(path.parent.glob("*.tmp-*")) == []
    restored = load(path)
    assert restored.current_phase is state.current_phase
    assert restored.observations == state.observations


def test_load_fails_closed_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LoopStateError, match="could not be read"):
        load(tmp_path / "nope.json")


def test_schema_version_is_emitted() -> None:
    assert to_dict(LoopState())["schema_version"] == 1
