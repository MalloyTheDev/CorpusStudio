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
