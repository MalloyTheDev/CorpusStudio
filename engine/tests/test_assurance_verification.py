"""Tests for the workspace-verification engine (Phase 8: scripts/assurance/verification.py).

Proves the fail-closed gate loader, that cs_assure RUNS the declared gate and reports the REAL exit
codes, the exit-code contract (green=0, red=1, cannot-evaluate=2), that a green result never
overclaims beyond WORKSPACE_GATE, and that the fired obligations ride along for the completion
record. All gates here are FAKE fast argv steps - the tests never invoke the real 50s gate (which
would recurse pytest-in-pytest).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CS_ASSURE = SCRIPTS_DIR / "cs_assure.py"
REAL_POLICY_TEXT = (SCRIPTS_DIR / "assurance" / "policy" / "obligations.json").read_text("utf-8")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from assurance.verification import (  # noqa: E402
    GateError,
    GateStep,
    LoadedGate,
    build_verification_record,
    load_gate,
    parse_gate,
    run_gate,
)
from assurance.records import verify_record  # noqa: E402


# --------------------------------------------------------------------------- helpers


def _step(name: str, exit_code: int, expected: int = 0) -> dict:
    return {"name": name, "argv": [sys.executable, "-c", f"import sys;sys.exit({exit_code})"],
            "cwd": ".", "expected_exit": expected}


def _gate(*steps: dict) -> dict:
    return {"schema_version": 1, "description": "test gate", "steps": list(steps)}


def _repo(tmp_path: Path, gate: dict) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "assurance" / "policy").mkdir(parents=True)
    for a in (("init", "-q", "-b", "main"), ("config", "user.email", "a@b.c"),
              ("config", "user.name", "t")):
        subprocess.run(["git", "-C", str(repo), *a], check=True)
    (repo / "scripts" / "assurance" / "policy" / "obligations.json").write_text(REAL_POLICY_TEXT)
    (repo / "scripts" / "assurance" / "policy" / "gate.json").write_text(json.dumps(gate))
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed"], check=True)
    return repo


def run_cli(start_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CS_ASSURE), *args], cwd=str(start_dir), capture_output=True, text=True
    )


# --------------------------------------------------------------------------- gate spec (fail-closed)


@pytest.mark.parametrize(
    "spec, needle",
    [
        ({"schema_version": 2, "steps": [_step("a", 0)]}, "schema_version"),
        ({"schema_version": 1, "steps": []}, "non-empty 'steps'"),
        (_gate(_step("a", 0), _step("a", 0)), "more than once"),
        ({"schema_version": 1, "steps": [{"name": "a", "argv": [], "cwd": ".", "expected_exit": 0}]}, "non-empty list"),
        ({"schema_version": 1, "steps": [{"name": "a", "argv": ["x"], "cwd": "/abs", "expected_exit": 0}]}, "repo-relative"),
        ({"schema_version": 1, "steps": [{"name": "a", "argv": ["x"], "cwd": "../x", "expected_exit": 0}]}, "'..'"),
        ({"schema_version": 1, "steps": [{"name": "a", "argv": ["x"], "cwd": ".", "expected_exit": True}]}, "0..255"),
        ({"schema_version": 1, "steps": [{"name": "a", "argv": ["x"], "cwd": ".", "expected_exit": -11}]}, "0..255"),
        ({"schema_version": 1, "steps": [{"name": "a", "argv": ["x"], "cwd": ".", "expected_exit": 256}]}, "0..255"),
        ({"schema_version": 1, "steps": [{"name": "a", "argv": ["x"], "cwd": "."}]}, "key mismatch"),
        ({"schema_version": 1, "steps": [{"name": "", "argv": ["x"], "cwd": ".", "expected_exit": 0}]}, "empty/non-string name"),
    ],
)
def test_parse_gate_fails_closed(spec: dict, needle: str) -> None:
    with pytest.raises(GateError, match=re.escape(needle)):
        parse_gate(spec)


def test_load_gate_reads_the_real_workspace_gate() -> None:
    gate = load_gate(REPO_ROOT)
    assert [s.name for s in gate.steps] == ["ruff", "mypy", "pytest"]
    assert gate.digest.startswith("sha256:")


def test_load_gate_missing_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="could not be read"):
        load_gate(tmp_path)


# --------------------------------------------------------------------------- run_gate


def test_run_gate_reports_real_exit_codes() -> None:
    gate = LoadedGate(
        steps=(GateStep("ok", (sys.executable, "-c", "import sys;sys.exit(0)"), ".", 0),
               GateStep("bad", (sys.executable, "-c", "import sys;sys.exit(2)"), ".", 0)),
        digest="sha256:x", schema_version=1, relpath="fake",
    )
    results = {r.name: r for r in run_gate(REPO_ROOT, gate)}
    assert results["ok"].passed and results["ok"].exit_code == 0
    assert not results["bad"].passed and results["bad"].exit_code == 2


def test_run_gate_unlaunchable_step_fails_closed() -> None:
    gate = LoadedGate(
        steps=(GateStep("missing", ("/no/such/binary_xyz",), ".", 0),),
        digest="sha256:x", schema_version=1, relpath="fake",
    )
    with pytest.raises(GateError, match="could not be launched"):
        run_gate(REPO_ROOT, gate)


def test_run_gate_embedded_null_argv_fails_closed() -> None:
    # An embedded NUL (reachable via gate.json "") makes subprocess raise ValueError, which is
    # neither OSError nor TimeoutExpired - it must still fail CLOSED, never escape as a traceback.
    gate = LoadedGate(
        steps=(GateStep("nul", ("bad\x00arg",), ".", 0),),
        digest="sha256:x", schema_version=1, relpath="fake",
    )
    with pytest.raises(GateError, match="could not be launched or run"):
        run_gate(REPO_ROOT, gate)


def test_run_gate_timeout_is_not_a_pass() -> None:
    gate = LoadedGate(
        steps=(GateStep("slow", (sys.executable, "-c", "import time;time.sleep(30)"), ".", 0),),
        digest="sha256:x", schema_version=1, relpath="fake",
    )
    result = run_gate(REPO_ROOT, gate, timeout=1)[0]
    assert result.timed_out is True and result.passed is False


# --------------------------------------------------------------------------- verification record


def test_verify_green_gate(tmp_path: Path) -> None:
    record = build_verification_record(start_dir=_repo(tmp_path, _gate(_step("a", 0), _step("b", 0))), base_ref="HEAD")
    assert record["record_type"] == "workspace_verification"
    assert record["payload"]["gate_passed"] is True
    assert record["payload"]["gate_passed_count"] == 2
    assert record["payload"]["completion_level"] == "WORKSPACE_GATE"
    assert verify_record(record)


def test_verify_red_gate(tmp_path: Path) -> None:
    record = build_verification_record(start_dir=_repo(tmp_path, _gate(_step("a", 0), _step("b", 1))), base_ref="HEAD")
    assert record["payload"]["gate_passed"] is False
    assert record["payload"]["gate_passed_count"] == 1
    steps = {s["name"]: s for s in record["payload"]["gate_steps"]}
    assert steps["b"]["passed"] is False and steps["b"]["exit_code"] == 1


def test_verify_lists_fired_obligations_without_asserting_discharge(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _gate(_step("a", 0)))
    (repo / "engine" / "corpus_studio" / "platform").mkdir(parents=True)
    (repo / "engine" / "corpus_studio" / "platform" / "worker.py").write_text("x\n")
    record = build_verification_record(start_dir=repo, base_ref="HEAD")
    fired = {f["id"]: f for f in record["payload"]["fired_obligations"]}
    assert "worker-closure" in fired
    # the record only LISTS what fired (a summary); it never carries a "discharged" flag.
    assert set(fired["worker-closure"]) == {"id", "severity", "obligation", "source", "trigger_path_count"}


def test_verify_record_is_deterministic(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _gate(_step("a", 0)))
    r1 = build_verification_record(start_dir=repo, base_ref="HEAD")
    r2 = build_verification_record(start_dir=repo, base_ref="HEAD")
    assert r1["record_digest"] == r2["record_digest"]  # no timestamps sealed


# --------------------------------------------------------------------------- CLI (exit contract)


def test_cli_verify_green_exit_0(tmp_path: Path) -> None:
    proc = run_cli(_repo(tmp_path, _gate(_step("a", 0))), "verify", "--base", "HEAD")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["payload"]["gate_passed"] is True


def test_cli_verify_red_exit_1_still_emits_record(tmp_path: Path) -> None:
    proc = run_cli(_repo(tmp_path, _gate(_step("a", 1))), "verify", "--base", "HEAD")
    assert proc.returncode == 1  # red gate is a not-clean result, NOT a fail-closed refusal
    assert json.loads(proc.stdout)["payload"]["gate_passed"] is False  # evidence still emitted


def test_cli_verify_malformed_gate_exit_2(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _gate(_step("a", 0)))
    (repo / "scripts" / "assurance" / "policy" / "gate.json").write_text("{bad json")
    proc = run_cli(repo, "verify", "--base", "HEAD")
    assert proc.returncode == 2 and "GateError" in proc.stderr and not proc.stdout.strip()


def test_cli_verify_unlaunchable_step_is_fail_closed_not_red(tmp_path: Path) -> None:
    repo = _repo(tmp_path, _gate({"name": "x", "argv": ["/no/such/bin_xyz"], "cwd": ".", "expected_exit": 0}))
    proc = run_cli(repo, "verify", "--base", "HEAD")
    assert proc.returncode == 2 and "GateError" in proc.stderr  # cannot evaluate != red gate


def test_cli_verify_embedded_null_argv_is_fail_closed_not_red(tmp_path: Path) -> None:
    # gate.json can carry a NUL via a JSON \u0000 escape; verify must fail closed (exit 2), never exit 1 (red) or crash.
    repo = _repo(tmp_path, _gate({"name": "nul", "argv": ["a\x00b"], "cwd": ".", "expected_exit": 0}))
    proc = run_cli(repo, "verify", "--base", "HEAD")
    assert proc.returncode == 2 and "GateError" in proc.stderr and not proc.stdout.strip()
