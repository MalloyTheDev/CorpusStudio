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
from loop.store import (  # noqa: E402
    ConcurrentStateWrite,
    CorruptStateFile,
    LoopStateError,
    dumps,
    from_dict,
    load,
    loads,
    save,
    save_cas,
    state_revision,
    to_dict,
)


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


def test_round_trip_is_independent_not_aliased() -> None:
    # A to_dict/from_dict round-trip must own INDEPENDENT containers - the driver mutates state in place,
    # so an aliased snapshot/restored copy would be silently corrupted.
    s = _mid_loop_state()
    r = from_dict(to_dict(s))
    assert r.observations is not s.observations and r.budgets is not s.budgets
    r.budgets["total_attempts"] = 999
    r.observations.append({"x": 1})
    assert s.budgets["total_attempts"] != 999
    assert len(s.observations) != len(r.observations)


def test_from_dict_rejects_a_future_or_non_int_schema_version() -> None:
    with pytest.raises(LoopStateError, match="schema_version"):
        from_dict({"schema_version": 999, "current_phase": "EXECUTE"})  # newer than this build
    with pytest.raises(LoopStateError, match="schema_version"):
        from_dict({"schema_version": "x", "current_phase": "EXECUTE"})  # not an int


# --------------------------------------------------------------------------- optimistic concurrency (#12)


def test_save_cas_increments_the_revision_and_records_metadata(tmp_path: Path) -> None:
    p = tmp_path / "loop.json"
    assert state_revision(p) == 0  # absent -> 0
    r1 = save_cas(_mid_loop_state(), p, expected_revision=0, writer_id="w1")
    assert r1 == 1 and state_revision(p) == 1
    import json
    doc = json.loads(p.read_text())
    assert doc["state_revision"] == 1 and doc["writer_id"] == "w1"
    assert doc["state_digest"].startswith("sha256:") and doc["previous_state_digest"] is None
    # a second write from the up-to-date writer chains the previous digest
    r2 = save_cas(_mid_loop_state(), p, expected_revision=1, writer_id="w1")
    assert r2 == 2 and json.loads(p.read_text())["previous_state_digest"] == doc["state_digest"]


def test_save_cas_rejects_a_stale_writer(tmp_path: Path) -> None:
    p = tmp_path / "loop.json"
    save_cas(_mid_loop_state(), p, expected_revision=0)          # -> revision 1
    with pytest.raises(ConcurrentStateWrite, match="not the expected"):
        save_cas(_mid_loop_state(), p, expected_revision=0)      # a writer that still thinks it is 0
    assert state_revision(p) == 1                                # the on-disk state was NOT clobbered


def test_load_ignores_cas_metadata(tmp_path: Path) -> None:
    p = tmp_path / "loop.json"
    s = _mid_loop_state()
    save_cas(s, p, expected_revision=0, writer_id="w1")
    restored = load(p)  # the metadata keys are not LoopState content
    assert restored.goal == s.goal and restored.current_phase is s.current_phase
    assert restored.observations == s.observations


def test_state_revision_distinguishes_absent_from_corrupt(tmp_path: Path) -> None:
    # Hardening (rev-0): an ABSENT file is a normal first-write (revision 0); a present-but-CORRUPT file
    # must NOT masquerade as revision 0 (which would let a CAS writer silently overwrite the corruption).
    assert state_revision(tmp_path / "none.json") == 0  # absent -> first write
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(CorruptStateFile):
        state_revision(corrupt)


def test_save_cas_refuses_to_overwrite_a_corrupt_file(tmp_path: Path) -> None:
    # A CAS writer over a corrupt file fails closed (CorruptStateFile) rather than clobbering it as a
    # fresh revision-0 write or misreporting a concurrent conflict.
    path = tmp_path / "s.json"
    path.write_text("garbage{", encoding="utf-8")
    with pytest.raises(CorruptStateFile):
        save_cas(_mid_loop_state(), path, expected_revision=0, writer_id="w1")
