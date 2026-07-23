"""Tests for multi-goal campaign orchestration (scripts/loop/campaign.py, last L8 piece).

Pins running a queue of goals (each its own loop), dependency skipping, stop-on-escalate, the shared
cross-goal learning ledger, per-goal state isolation, and fail-closed validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.campaign import CampaignError, Goal, run_campaign  # noqa: E402
from loop.completeness import Criterion, CriterionKind  # noqa: E402
from loop.controller import Observation  # noqa: E402
from loop.orchestrate import LoopContext  # noqa: E402


def _cs():
    steps = [{"name": n, "passed": True, "exit_code": 0, "timed_out": False} for n in ("ruff", "mypy", "pytest")]
    rec = {"verify": {"record_digest": "sha256:v", "payload": {
        "gate_passed": True, "gate_steps": steps, "fired_obligations": [], "change_set_fingerprint": "cs:x"}},
        "changeset": {"payload": {"changed_paths": []}}, "impact": {"payload": {"fired_obligations": []}},
        "doclint": {"finding_count": 0}}
    return lambda _r, *a: (0, json.dumps(rec.get(a[0] if a else "", {})), "")


def _gh():
    snapshot = json.dumps({"headRefOid": "sha1", "statusCheckRollup": [{"name": "pytest", "bucket": "pass"}]})
    return lambda *a: (0, "merged", "") if len(a) >= 2 and a[1] == "merge" else (0, snapshot, "")


def _ctx(*, dangerous: bool = False, ledger_path: Path | None = None) -> LoopContext:
    return LoopContext(repo_root=REPO_ROOT, executor=lambda _s, _d: Observation.SUCCESS,
                       reviewer=lambda _s: [],
                       critic=lambda _s: [Criterion("c", "done", kind=CriterionKind.DETERMINISTIC,
                                                    met=True, evidence="sha256:v")],
                       gh_runner=_gh(), pr_ref="1", dangerous=dangerous, run_cs_assure=_cs(),
                       ledger_path=ledger_path)


def _goals(*ids: str) -> list[Goal]:
    return [Goal(goal=f"do {i}", goal_id=i) for i in ids]


def _boom(_goal: Goal) -> LoopContext:
    raise RuntimeError("factory blew up")


def test_campaign_runs_a_queue_of_goals_to_finalize() -> None:
    outcomes = run_campaign(_goals("g1", "g2", "g3"), _ctx())
    assert [o.goal_id for o in outcomes] == ["g1", "g2", "g3"]
    assert all(o.finalized and o.final_phase == "FINALIZE" for o in outcomes)


def test_dependency_skips_when_an_upstream_goal_did_not_finalize() -> None:
    # g1 escalates (dangerous); g2 depends on g1 -> skipped, not run blindly.
    goals = [Goal("risky", "g1"), Goal("needs g1", "g2", depends_on=["g1"])]
    outcomes = run_campaign(goals, _ctx(dangerous=True), stop_on_escalate=False)
    by_id = {o.goal_id: o for o in outcomes}
    assert by_id["g1"].final_phase == "ESCALATED" and not by_id["g1"].finalized
    assert by_id["g2"].final_phase == "SKIPPED"


def test_stop_on_escalate_halts_the_campaign() -> None:
    outcomes = run_campaign(_goals("g1", "g2", "g3"), _ctx(dangerous=True), stop_on_escalate=True)
    by = {o.goal_id: o.final_phase for o in outcomes}
    assert by["g1"] == "ESCALATED" and by["g2"] == "SKIPPED" and by["g3"] == "SKIPPED"  # g2/g3 not run


def test_topological_scheduling_runs_dependency_before_dependent_regardless_of_order() -> None:
    # gB is listed BEFORE its dependency gA; topological scheduling must run gA first, then gB - never
    # skip gB just because it appeared before its prerequisite in the input.
    goals = [Goal("depends on A", "gB", depends_on=["gA"]), Goal("prereq", "gA")]
    outcomes = {o.goal_id: o for o in run_campaign(goals, _ctx())}
    assert outcomes["gA"].finalized and outcomes["gB"].finalized


def test_shared_ledger_records_each_goal(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"goal": "prior", "failed_approaches": ["sha256:old"]}]))
    run_campaign(_goals("g1", "g2"), _ctx(ledger_path=ledger))
    entries = json.loads(ledger.read_text())
    recorded = {e.get("goal_id") for e in entries}
    assert "g1" in recorded and "g2" in recorded  # both goals recorded to the shared ledger


def test_per_goal_state_isolation(tmp_path: Path) -> None:
    run_campaign(_goals("g1", "g2"), _ctx(), store_dir=tmp_path)
    assert (tmp_path / "g1.json").is_file() and (tmp_path / "g2.json").is_file()  # separate state files


def test_context_for_factory_isolates_each_goal(tmp_path: Path) -> None:
    # With a context_for factory, each goal gets its OWN context (here: its own state file in its own
    # per-goal directory) - the isolation seam a runtime fills with a per-goal branch/worktree/PR.
    from dataclasses import replace as _replace
    seen: list[str] = []

    def context_for(goal: Goal) -> LoopContext:
        seen.append(goal.goal_id)
        goal_dir = tmp_path / goal.goal_id
        goal_dir.mkdir()
        return _replace(_ctx(), store_path=goal_dir / "state.json")

    outcomes = run_campaign(_goals("g1", "g2"), context_for=context_for)
    assert all(o.finalized for o in outcomes) and seen == ["g1", "g2"]
    assert (tmp_path / "g1" / "state.json").is_file() and (tmp_path / "g2" / "state.json").is_file()


def test_a_misbehaving_factory_fails_closed() -> None:
    with pytest.raises(CampaignError, match="must return a LoopContext"):
        run_campaign(_goals("g1"), context_for=lambda _g: "not a context")  # type: ignore[arg-type,return-value]
    with pytest.raises(CampaignError, match="raised RuntimeError"):
        run_campaign(_goals("g1"), context_for=_boom)


def test_needs_a_ctx_or_a_factory() -> None:
    with pytest.raises(CampaignError, match="base ctx or a context_for"):
        run_campaign(_goals("g1"))  # neither given


def test_a_finished_goal_is_resumed_not_rerun(tmp_path: Path) -> None:
    # A goal whose state file already shows FINALIZE is RESUMED (reported), never re-executed - the executor
    # must not be called, and the ledger must not double-record it.
    from loop.store import save
    from loop.controller import LoopState, Phase
    done = LoopState(goal="do g1", goal_id="g1", current_phase=Phase.FINALIZE)
    save(done, tmp_path / "g1.json")
    ledger = tmp_path / "ledger.json"

    def explode(_s: LoopState, _d: object) -> Observation:
        raise AssertionError("a resumed, already-finished goal must not be executed")

    ctx = LoopContext(repo_root=REPO_ROOT, executor=explode, reviewer=lambda _s: [],
                      critic=lambda _s: [Criterion("c", "done", kind=CriterionKind.DETERMINISTIC,
                                                   met=True, evidence="sha256:v")],
                      gh_runner=_gh(), pr_ref="1", run_cs_assure=_cs(), ledger_path=ledger)
    outcomes = run_campaign([Goal("do g1", "g1")], ctx, store_dir=tmp_path)
    assert outcomes[0].finalized  # reported as finished
    assert not ledger.exists()  # a resumed-terminal goal is not re-recorded to the ledger


def test_validation_fails_closed() -> None:
    with pytest.raises(CampaignError, match="duplicate"):
        run_campaign([Goal("a", "g1"), Goal("b", "g1")], _ctx())
    with pytest.raises(CampaignError, match="unknown goal"):
        run_campaign([Goal("a", "g1", depends_on=["ghost"])], _ctx())
    with pytest.raises(CampaignError, match="unsafe goal_id"):
        run_campaign([Goal("a", "")], _ctx())  # empty id
    with pytest.raises(CampaignError, match="cycle"):
        run_campaign([Goal("a", "g1", depends_on=["g2"]), Goal("b", "g2", depends_on=["g1"])], _ctx())


def test_unsafe_goal_id_is_rejected_before_it_can_escape_the_store_dir(tmp_path: Path) -> None:
    # A goal_id names a per-goal state file; a traversal id must be refused, never written outside store_dir.
    for bad in ("../../etc/pwned", "a/b", "..", "with space", "x" * 65):
        with pytest.raises(CampaignError, match="unsafe goal_id"):
            run_campaign([Goal("x", bad)], _ctx(), store_dir=tmp_path)
