"""Phase 7.1 - the write-capable single-agent adapter (scripts/loop_adapters/single_agent_write.py).

Pins the WRITE path end-to-end AND its safety: the agent's sealed diff is applied in an ISOLATED worktree
(never the main tree), committed on a fresh branch, pushed, and a PR is opened - while the developer's
working tree is left pristine, no merge ever happens, and any failure (a diff that won't apply, a drifted
apply) fails closed. A local bare remote makes ``git push`` work offline; ``gh`` is faked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import loop_adapters.single_agent_write as saw  # noqa: E402
from loop.controller import LoopState, Phase  # noqa: E402
from loop.orchestrate import run_loop  # noqa: E402

_DIFF = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"


class _StubAgent:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response if response is not None else {"unified_diff": _DIFF, "rationale": "make it new"}

    def propose(self, request: dict) -> dict:
        return self.response


def _cs_assure_green():
    import json
    steps = [{"name": n, "passed": True, "exit_code": 0, "timed_out": False} for n in ("ruff", "mypy", "pytest")]
    rec = {
        "verify": {"record_type": "workspace_verification", "schema_version": 2, "record_digest": "sha256:v",
                   "payload": {"gate_passed": True, "gate_steps": steps, "workspace_stable": True,
                               "fired_obligations": [], "change_set_fingerprint": "cs:x"}},
        "changeset": {"payload": {"changed_paths": []}},
        "impact": {"payload": {"fired_obligations": [], "base_policy_available": True, "change_set_fingerprint": "cs:x"}},
        "doclint": {"finding_count": 0},
    }
    return lambda _r, *a: (0, json.dumps(rec.get(a[0] if a else "", {})), "")


def _g(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def _repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True, capture_output=True)
    root = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True, capture_output=True)
    _g(root, "config", "user.email", "a@b.c")
    _g(root, "config", "user.name", "t")
    (root / "README.md").write_text("old\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-q", "-m", "base")
    _g(root, "remote", "add", "origin", str(remote))
    _g(root, "push", "-q", "-u", "origin", "main")
    return root, remote


def _fake_gh(calls: list):
    def run(*argv: str) -> tuple[int, str, str]:
        calls.append(tuple(argv))
        if tuple(argv[:2]) == ("pr", "create"):
            return (0, "https://example/pull/1\n", "")
        return (0, "", "")
    return run


def _build(tmp_path: Path, root: Path, agent: _StubAgent, calls: list):
    return saw.build_context(root, "main", agent_client=agent, proposals_dir=tmp_path / "prop",
                             worktrees_dir=tmp_path / "wt", gh_runner=_fake_gh(calls),
                             run_cs_assure=_cs_assure_green())


# --------------------------------------------------------------------------- the write path + isolation


def test_declares_the_write_capability_and_escalates_the_merge() -> None:
    ctx = saw.build_context(REPO_ROOT, "main", agent_client=_StubAgent(), gh_runner=lambda *a: (0, "", ""),
                            proposals_dir=REPO_ROOT / ".t", worktrees_dir=REPO_ROOT / ".t")
    assert ctx.capabilities == frozenset({"write"})  # the capability gate refuses without --allow-capabilities write
    assert ctx.dangerous is True                     # the merge gate escalates - a human merges, never the loop


def test_write_run_applies_in_a_worktree_pushes_a_branch_opens_a_pr_and_leaves_main_untouched(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    state = LoopState(goal="tidy the readme", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), calls))

    # the loop ESCALATED (a human reviews + merges the PR) and a PR was opened
    assert state.current_phase is Phase.ESCALATED
    assert ("pr", "create") in [c[:2] for c in calls] and ("pr", "merge") not in [c[:2] for c in calls]
    ref = state.review_state["agent_proposals"][0]
    assert ref["branch"] == "cs-agent/g1" and ref["pr"] == "https://example/pull/1"
    assert ref["changed_paths"] == ["README.md"]

    # the write landed on a PUSHED branch in the remote, carrying the applied change...
    assert "cs-agent/g1" in _g(remote, "branch", "--list", "cs-agent/g1").stdout
    assert _g(remote, "show", "cs-agent/g1:README.md").stdout == "new\n"
    # ...but the developer's MAIN working tree is pristine, and no worktree is left behind
    assert _g(root, "status", "--porcelain").stdout == ""
    assert (root / "README.md").read_text() == "old\n"
    assert "cs-agent" not in _g(root, "worktree", "list").stdout


def test_a_diff_that_does_not_apply_fails_closed_and_writes_nothing(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    bad = _StubAgent({"unified_diff": "--- a/README.md\n+++ b/README.md\n@@ -9 +9 @@\n-nope\n+x\n", "rationale": "r"})
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, bad, []))
    # git apply rejects the bogus hunk -> WriteAdapterError -> escalate; nothing pushed, main pristine, no leftover wt
    assert state.current_phase is Phase.ESCALATED
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""
    assert _g(root, "status", "--porcelain").stdout == "" and "cs-agent" not in _g(root, "worktree", "list").stdout


def test_a_pr_create_failure_fails_closed(tmp_path: Path) -> None:
    root, _remote = _repo_with_remote(tmp_path)

    def gh_refuses(*argv: str) -> tuple[int, str, str]:
        return (1, "", "gh: not authenticated") if argv[:2] == ("pr", "create") else (0, "", "")
    ctx = saw.build_context(root, "main", agent_client=_StubAgent(), proposals_dir=tmp_path / "p",
                            worktrees_dir=tmp_path / "w", gh_runner=gh_refuses, run_cs_assure=_cs_assure_green())
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx)
    assert state.current_phase is Phase.ESCALATED  # a failed PR-create escalates; the worktree is still disposed
    assert "cs-agent" not in _g(root, "worktree", "list").stdout


# --------------------------------------------------------------------------- the gh boundary (no merge)


def test_write_gh_allows_pr_create_but_refuses_merge_and_every_other_mutation(tmp_path: Path) -> None:
    gh = saw.write_gh(tmp_path)
    for refused in (("pr", "merge", "1"), ("pr", "close", "1"), ("pr", "edit", "1"), ("api", "-X", "POST")):
        code, _out, err = gh(*refused)
        assert code == 97 and "refused" in err, refused
    # `pr create` is allowlisted -> not a 97 refusal (it may still fail on env gh auth; that's a real exit)
    assert gh("pr", "create", "--head", "x")[0] != 97


class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_default_worktrees_dir_resolves_the_git_path_and_falls_back(tmp_path, monkeypatch) -> None:
    # inside a repo: the isolated-worktree dir is under the git dir (outside the working tree)...
    monkeypatch.setattr(saw.subprocess, "run",
                        lambda *a, **k: _FakeProc(0, "/abs/git/corpusstudio-loop/worktrees\n"))
    assert saw._default_worktrees_dir(tmp_path) == Path("/abs/git/corpusstudio-loop/worktrees")
    # ...and outside a repo it falls back to a worktree-local path (never inside the working tree implicitly).
    monkeypatch.setattr(saw.subprocess, "run", lambda *a, **k: _FakeProc(128, "", "not a git repo"))
    assert saw._default_worktrees_dir(tmp_path) == tmp_path / ".corpusstudio-loop-worktrees"


def test_git_helper_fails_closed_on_a_nonzero_exit(tmp_path) -> None:
    root, _remote = _repo_with_remote(tmp_path)
    with pytest.raises(saw.WriteAdapterError):
        saw._git(root, "rev-parse", "--verify", "does-not-exist")


def test_branch_suffix_is_sanitized_to_a_safe_ref() -> None:
    # a messy goal id can never yield an invalid git ref (spaces / punctuation / '..' / case / length).
    assert saw._sanitize_branch_suffix("Fix bug #5 (README)!") == "fix-bug-5-readme"
    assert saw._sanitize_branch_suffix("a..b") == "a-b" and ".." not in saw._sanitize_branch_suffix("a..b")
    assert saw._sanitize_branch_suffix("") == "goal" and saw._sanitize_branch_suffix("...--__") == "goal"
    assert len(saw._sanitize_branch_suffix("x" * 100)) <= 40
