"""Tests for the cs_loop interactive CLI (scripts/cs_loop.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CS_LOOP = REPO_ROOT / "scripts" / "cs_loop.py"


def _run(state: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(CS_LOOP), "--state", str(state), *args],
                          capture_output=True, text=True)


def test_init_next_status_flow(tmp_path: Path) -> None:
    state = tmp_path / "loop.json"
    init = _run(state, "init", "--goal", "add a scorer")
    assert init.returncode == 0 and state.is_file()
    assert json.loads(init.stdout)["goal"] == "add a scorer"

    directive = json.loads(_run(state, "next").stdout)
    assert directive["phase"] == "RECEIVE_GOAL" and directive["terminal"] is False

    status = json.loads(_run(state, "status").stdout)
    assert status["phase"] == "RECEIVE_GOAL" and status["observations"] == 0


def test_init_refuses_to_clobber_without_force(tmp_path: Path) -> None:
    state = tmp_path / "loop.json"
    assert _run(state, "init", "--goal", "g").returncode == 0
    again = _run(state, "init", "--goal", "g2")
    assert again.returncode == 2 and again.stderr.startswith("cs_loop:")
    assert _run(state, "init", "--goal", "g2", "--force").returncode == 0  # --force overwrites


def test_status_fails_closed_on_missing_state(tmp_path: Path) -> None:
    proc = _run(tmp_path / "nope.json", "status")
    assert proc.returncode == 2 and proc.stderr.startswith("cs_loop:") and "Traceback" not in proc.stderr


def test_observe_refuses_on_a_terminal_loop(tmp_path: Path) -> None:
    # Drive the state to terminal by hand, then observe must refuse (fail-closed), not run the gate.
    state = tmp_path / "loop.json"
    _run(state, "init", "--goal", "g")
    data = json.loads(state.read_text())
    data["current_phase"] = "FINALIZE"
    data["termination_reason"] = "done"
    state.write_text(json.dumps(data))
    proc = _run(state, "observe")
    assert proc.returncode == 2 and "terminal" in proc.stderr


# --------------------------------------------------------------------------- runtime commands (PR3)

_ADAPTER = """
import json
from loop.orchestrate import LoopContext
from loop.controller import Observation
from loop.completeness import Criterion, CriterionKind

def build_context(repo_root, base):
    steps = [{"name": n, "passed": True, "exit_code": 0, "timed_out": False} for n in ("ruff", "mypy", "pytest")]
    rec = {"verify": {"record_digest": "sha256:v", "payload": {"gate_passed": True, "gate_steps": steps,
           "fired_obligations": [], "change_set_fingerprint": "cs:x"}},
           "changeset": {"payload": {"changed_paths": []}}, "impact": {"payload": {"fired_obligations": []}},
           "doclint": {"finding_count": 0}}
    def gh(*a):
        return (0, "merged", "") if len(a) >= 2 and a[1] == "merge" else (0, json.dumps([{"name": "pytest", "bucket": "pass"}]), "")
    # DETERMINISTIC criterion citing the verify record the OBSERVE step seals -> autonomous finalize.
    return LoopContext(repo_root=repo_root, executor=lambda s, d: Observation.SUCCESS, reviewer=lambda s: [],
                       critic=lambda s: [Criterion("c", "done", kind=CriterionKind.DETERMINISTIC,
                                                   met=True, evidence="sha256:v")], gh_runner=gh, pr_ref="1",
                       run_cs_assure=lambda r, *a: (0, json.dumps(rec.get(a[0] if a else "", {})), ""))
"""


def _adapter(tmp_path: Path) -> Path:
    p = tmp_path / "adapters.py"
    p.write_text(_ADAPTER)
    return p


def test_run_drives_the_integrated_loop_to_finalize(tmp_path: Path) -> None:
    state = tmp_path / "loop.json"
    _run(state, "init", "--goal", "ship it")
    out = _run(state, "run", "--adapters", str(_adapter(tmp_path)), "--repo-root", str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["phase"] == "FINALIZE" and result["terminal"] is True


def test_campaign_runs_a_dependency_ordered_backlog(tmp_path: Path) -> None:
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps([{"goal": "b", "goal_id": "g2", "depends_on": ["g1"]},
                                 {"goal": "a", "goal_id": "g1"}]))  # b before a: topo must still run both
    out = _run(tmp_path / "unused.json", "campaign", "--adapters", str(_adapter(tmp_path)),
               "--goals", str(goals), "--repo-root", str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert all(o["finalized"] for o in json.loads(out.stdout)["outcomes"])


def test_inspect_pause_run_refused_resume_abort(tmp_path: Path) -> None:
    state = tmp_path / "loop.json"
    _run(state, "init", "--goal", "g")
    assert json.loads(_run(state, "inspect").stdout)["goal"] == "g"
    assert json.loads(_run(state, "pause").stdout)["paused"] is True
    refused = _run(state, "run", "--adapters", str(_adapter(tmp_path)), "--repo-root", str(REPO_ROOT))
    assert refused.returncode == 2 and "paused" in refused.stderr  # run refuses while paused
    assert json.loads(_run(state, "resume").stdout)["paused"] is False
    assert json.loads(_run(state, "abort", "--reason", "stop").stdout)["phase"] == "STOPPED"


def test_abort_clears_a_pending_pause(tmp_path: Path) -> None:
    # A paused-then-aborted loop must be unambiguously terminal, not "both STOPPED and paused".
    state = tmp_path / "loop.json"
    _run(state, "init", "--goal", "g")
    _run(state, "pause")
    _run(state, "abort", "--reason", "stop")
    assert "paused" not in json.loads(_run(state, "inspect").stdout)["review_state"]


def test_campaign_rejects_a_non_list_depends_on(tmp_path: Path) -> None:
    # depends_on: "g1" must NOT be coerced to ['g','1'] - a mistyped config fails fast, clearly.
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps([{"goal": "a", "goal_id": "g1", "depends_on": "g0"}]))
    proc = _run(tmp_path / "unused.json", "campaign", "--adapters", str(_adapter(tmp_path)),
                "--goals", str(goals), "--repo-root", str(REPO_ROOT))
    assert proc.returncode == 2 and "depends_on" in proc.stderr and "Traceback" not in proc.stderr


def test_campaign_rejects_a_non_object_goal_entry(tmp_path: Path) -> None:
    # A non-object entry is a config error - fail closed, never silently drop the goal.
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps([{"goal": "a", "goal_id": "g1"}, "oops"]))
    proc = _run(tmp_path / "unused.json", "campaign", "--adapters", str(_adapter(tmp_path)),
                "--goals", str(goals), "--repo-root", str(REPO_ROOT))
    assert proc.returncode == 2 and "must be an object" in proc.stderr


def test_run_refuses_an_unloadable_adapter(tmp_path: Path) -> None:
    state = tmp_path / "loop.json"
    _run(state, "init", "--goal", "g")
    bad = tmp_path / "bad_adapter.py"
    bad.write_text("this is not valid python :(\n")
    proc = _run(state, "run", "--adapters", str(bad), "--repo-root", str(REPO_ROOT))
    assert proc.returncode == 2 and "adapter module" in proc.stderr and "Traceback" not in proc.stderr


def test_authorize_unescalates_a_blocked_loop(tmp_path: Path) -> None:
    state = tmp_path / "loop.json"
    _run(state, "init", "--goal", "g")
    data = json.loads(state.read_text())
    data["current_phase"] = "ESCALATED"
    data["termination_reason"] = "blocked"
    state.write_text(json.dumps(data))
    out = json.loads(_run(state, "authorize", "--grant", "merge-ok", "--note", "reviewed").stdout)
    assert out["unescalated"] is True and out["phase"] == "DIAGNOSE"
