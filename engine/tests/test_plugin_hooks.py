"""Behavioural tests for the CorpusStudio Claude-Code plugin hooks (.claude/hooks/*).

The hooks are fail-safe infrastructure: the Stop reminder must NEVER trap a session except on an
explicit, current FINALIZE_REQUESTED state, and the advisory classifier must never crash or block.
These run the hooks as real subprocesses (the way the harness invokes them) so a regression in the
allow/block logic, the fixed state path, the None-input guard, or the boundary match is caught.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FINALIZE_HOOK = REPO_ROOT / ".claude" / "hooks" / "finalize_reminder.py"
ADVISORY_HOOK = REPO_ROOT / ".claude" / "hooks" / "advisory_classify.py"

pytestmark = pytest.mark.skipif(
    not FINALIZE_HOOK.exists() or not ADVISORY_HOOK.exists(),
    reason="plugin hooks are not present on this checkout",
)


def _run(hook: Path, payload: object, cwd: Path, *, raw: str | None = None) -> subprocess.CompletedProcess[str]:
    text = raw if raw is not None else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(hook)], input=text, cwd=str(cwd), capture_output=True, text=True
    )


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for args in (("init", "-q", "-b", "main"), ("config", "user.email", "a@b.c"),
                 ("config", "user.name", "t")):
        subprocess.run(["git", "-C", str(path), *args], check=True)
    return path


def _write_slice_state(repo: Path, state: dict) -> None:
    rel = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--git-path",
         "corpusstudio-assurance/current-slice.json"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    path = Path(rel) if Path(rel).is_absolute() else repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def _blocks(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode != 0 or not proc.stdout.strip():
        return False
    try:
        return json.loads(proc.stdout).get("decision") == "block"
    except ValueError:
        return False


# --------------------------------------------------------------------------- finalize Stop hook


def test_finalize_allows_when_no_state(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "r")
    proc = _run(FINALIZE_HOOK, {"cwd": str(repo)}, repo)
    assert proc.returncode == 0 and not _blocks(proc)


def test_finalize_blocks_only_on_finalize_requested_without_stop_reason(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "r")
    _write_slice_state(repo, {"phase": "FINALIZE_REQUESTED", "stop_reason": None})
    proc = _run(FINALIZE_HOOK, {"cwd": str(repo)}, repo)
    assert proc.returncode == 0 and _blocks(proc)


def test_finalize_allows_other_phase(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "r")
    _write_slice_state(repo, {"phase": "ACT", "stop_reason": None})
    assert not _blocks(_run(FINALIZE_HOOK, {"cwd": str(repo)}, repo))


def test_finalize_allows_when_stop_reason_set(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "r")
    _write_slice_state(repo, {"phase": "FINALIZE_REQUESTED", "stop_reason": "done"})
    assert not _blocks(_run(FINALIZE_HOOK, {"cwd": str(repo)}, repo))


def test_finalize_present_but_falsy_stop_reason_does_not_trap(tmp_path: Path) -> None:
    # H2: resolution is tested by PRESENCE (`is not None`), so a present-but-empty stop_reason
    # releases the session instead of re-trapping it.
    repo = _init_repo(tmp_path / "r")
    _write_slice_state(repo, {"phase": "FINALIZE_REQUESTED", "stop_reason": ""})
    assert not _blocks(_run(FINALIZE_HOOK, {"cwd": str(repo)}, repo))


def test_finalize_allows_when_stop_hook_active(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "r")
    _write_slice_state(repo, {"phase": "FINALIZE_REQUESTED", "stop_reason": None})
    proc = _run(FINALIZE_HOOK, {"cwd": str(repo), "stop_hook_active": True}, repo)
    assert not _blocks(proc)  # re-entrant block-cap guard


def test_finalize_allows_on_malformed_stdin(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "r")
    proc = _run(FINALIZE_HOOK, None, repo, raw="not json{{{")
    assert proc.returncode == 0 and not _blocks(proc)


# --------------------------------------------------------------------------- advisory classifier


def _advise(payload: object, cwd: Path, *, raw: str | None = None) -> subprocess.CompletedProcess[str]:
    return _run(ADVISORY_HOOK, payload, cwd, raw=raw)


def test_advisory_flags_contract_edit(tmp_path: Path) -> None:
    proc = _advise({"tool_input": {"file_path": "engine/corpus_studio/platform/contracts.py"}}, tmp_path)
    assert proc.returncode == 0 and "contract-count" in proc.stderr


def test_advisory_boundary_match_rejects_lookalike_path(tmp_path: Path) -> None:
    # "training/trainer.py" must NOT match "myproj_training/trainer.py" (segment boundary).
    proc = _advise({"tool_input": {"file_path": "myproj_training/trainer.py"}}, tmp_path)
    assert proc.returncode == 0 and proc.stderr.strip() == ""


def test_advisory_boundary_match_accepts_real_worker_path(tmp_path: Path) -> None:
    proc = _advise({"tool_input": {"file_path": "engine/corpus_studio/training/trainer.py"}}, tmp_path)
    assert proc.returncode == 0 and "worker-closure" in proc.stderr


def test_advisory_survives_null_tool_input(tmp_path: Path) -> None:
    proc = _advise({"tool_input": None}, tmp_path)
    assert proc.returncode == 0 and proc.stderr.strip() == ""


def test_advisory_sanitizes_control_chars(tmp_path: Path) -> None:
    proc = _advise({"tool_input": {"file_path": "scripts/assurance/x\n\x1b[31mEVIL.py"}}, tmp_path)
    assert proc.returncode == 0
    assert "\x1b" not in proc.stderr and "\n[31m" not in proc.stderr  # no raw ANSI / newline injection


def test_advisory_ignores_ordinary_path(tmp_path: Path) -> None:
    proc = _advise({"tool_input": {"file_path": "README.md"}}, tmp_path)
    assert proc.returncode == 0 and proc.stderr.strip() == ""
