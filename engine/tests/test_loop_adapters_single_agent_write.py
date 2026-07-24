"""Phase 7.1 - the write-capable single-agent adapter (scripts/loop_adapters/single_agent_write.py).

Pins the WRITE path end-to-end AND its safety: the agent's sealed diff is applied in an ISOLATED worktree
(never the main tree), committed on a fresh branch, pushed, and a PR is opened - while the developer's
working tree is left pristine, no merge ever happens, and any failure (a diff that won't apply, a drifted
apply) fails closed. A local bare remote makes ``git push`` work offline; ``gh`` is faked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import loop_adapters.single_agent as sa  # noqa: E402
import loop_adapters.single_agent_write as saw  # noqa: E402
from loop.controller import LoopState, Phase  # noqa: E402
from loop.orchestrate import run_loop  # noqa: E402

_DIFF = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"


class _StubAgent:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response if response is not None else {"unified_diff": _DIFF, "rationale": "make it new"}
        self.seen_cwd: str | None = None
        self.cwd_was_dir: bool | None = None

    def propose(self, request: dict) -> dict:
        # Record how the executor CONFINED us: the cwd it handed us, and whether it existed AT CALL TIME
        # (the disposable worktree is removed after we return, so this is the only chance to observe it).
        self.seen_cwd = request.get("_cwd")
        self.cwd_was_dir = bool(self.seen_cwd) and Path(self.seen_cwd).is_dir()
        return self.response


def _green_verify():
    # for the loop's OWN dev-tree OBSERVE/VERIFY (trusted, always green in these tests)
    steps = [{"name": n, "passed": True, "exit_code": 0, "timed_out": False} for n in ("ruff", "mypy", "pytest")]
    return {"record_type": "workspace_verification", "schema_version": 2, "record_digest": "sha256:v",
            "payload": {"gate_passed": True, "gate_steps": steps, "workspace_stable": True,
                        "fired_obligations": [], "change_set_fingerprint": "cs:x"}}


def _assure_runner(*, obligations=()):
    """A fake cs_assure. `impact` drives the STATIC CANDIDATE assurance in the executor (fired obligations
    block/route the publish); `verify`/`doclint`/`changeset` drive the loop's own dev-tree OBSERVE/VERIFY.
    `impact` executes NO code - candidate assurance is deliberately static (no untrusted pytest locally)."""
    rec = {
        "impact": {"record_digest": "sha256:imp",
                   "payload": {"fired_obligations": [{"id": o, "severity": "blocking"} for o in obligations],
                               "base_policy_available": True, "change_set_fingerprint": "cs:x"},
                   "provenance": {"policy_digest": "sha256:p"}},
        "verify": _green_verify(),
        "changeset": {"payload": {"changed_paths": []}},
        "doclint": {"finding_count": 0},
    }
    return lambda _r, *a: (0, json.dumps(rec.get(a[0] if a else "", {})), "")


def _cs_assure_green():
    return _assure_runner()


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


def _build(tmp_path: Path, root: Path, agent: _StubAgent, calls: list, assure=None):
    return saw.build_context(root, "main", agent_client=agent, proposals_dir=tmp_path / "prop",
                             worktrees_dir=tmp_path / "wt", gh_runner=_fake_gh(calls),
                             run_cs_assure=assure or _cs_assure_green())


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


def test_the_agent_is_confined_to_a_disposable_worktree_not_the_repo(tmp_path: Path) -> None:
    root, _remote = _repo_with_remote(tmp_path)
    agent = _StubAgent()
    state = LoopState(goal="tidy", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, agent, []))
    # the propose ran with cwd inside a REAL directory (a worktree), under worktrees_dir, and NOT the repo...
    assert agent.cwd_was_dir is True
    assert agent.seen_cwd is not None
    seen = Path(agent.seen_cwd).resolve()
    assert (tmp_path / "wt").resolve() in seen.parents
    assert seen != root.resolve()
    # ...and that disposable propose worktree is gone afterwards (only the branch worktree ever pushes).
    assert not seen.exists()


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


# ------------------------------------------------------------------ candidate assurance gates the publish (7.1.2)


def _denied_diff(path: str) -> str:
    return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+content\n"


def test_a_human_gated_candidate_escalates_and_publishes_nothing(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    # the candidate's static impact fires a self-modify obligation -> AUTHORIZATION_REQUIRED (a human).
    assure = _assure_runner(obligations=("loop-controller-self-modify",))
    ctx = _build(tmp_path, root, _StubAgent(), calls, assure=assure)
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx, max_steps=25)
    assert state.current_phase is Phase.ESCALATED
    assert ("pr", "create") not in [c[:2] for c in calls]
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""
    ca = state.review_state["candidate_assurance"]
    assert ca["published"] is False and ca["observation"] == "AUTHORIZATION_REQUIRED"


def test_a_worker_touching_candidate_escalates_and_publishes_nothing(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    # the candidate's static impact fires worker-closure -> WORKER_LINEAGE_IMPACT (human-gated workflow).
    ctx = _build(tmp_path, root, _StubAgent(), calls, assure=_assure_runner(obligations=("worker-closure",)))
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx, max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""
    ca = state.review_state["candidate_assurance"]
    assert ca["published"] is False and ca["observation"] == "WORKER_LINEAGE_IMPACT"


def test_a_clear_candidate_publishes_and_records_the_assurance(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    state = LoopState(goal="tidy", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), calls))
    # a clear candidate publishes (branch pushed by refspec from a DETACHED apply worktree) + records it
    assert _g(remote, "show", "cs-agent/g1:README.md").stdout == "new\n"
    ca = state.review_state["candidate_assurance"]
    assert ca["published"] is True and ca["observation"] == "SUCCESS"
    assert "sha256:imp" in state.assurance_records  # the candidate impact digest is on the audit trail


def test_candidate_assurance_targets_the_candidate_worktree_not_the_dev_tree(tmp_path: Path) -> None:
    # the whole point of 7.1.2: `impact` must be run against the CANDIDATE worktree, never the dev tree.
    seen: list = []
    base = _assure_runner()

    def recording(root, *a):
        if a and a[0] == "impact":
            seen.append(str(root))
        return base(root, *a)
    root, _remote = _repo_with_remote(tmp_path)
    ctx = _build(tmp_path, root, _StubAgent(), [], assure=recording)
    run_loop(LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL), ctx)
    assert seen, "candidate impact was never run"
    for observed in seen:
        assert (tmp_path / "wt").resolve() in Path(observed).resolve().parents  # the isolated worktree...
        assert Path(observed).resolve() != root.resolve()                       # ...never the dev tree


def test_a_sensitive_path_is_denied_before_apply(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    # a diff touching a CI-workflow path is refused pre-apply: no apply worktree, no push, no PR.
    agent = _StubAgent({"unified_diff": _denied_diff(".github/workflows/ci.yml"), "rationale": "r"})
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, agent, calls), max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""
    assert "apply-" not in _g(root, "worktree", "list").stdout
    assert state.review_state.get("agent_proposals") in (None, [])


def test_a_secret_in_the_diff_blocks_the_publish(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    secret_diff = "--- a/README.md\n+++ b/README.md\n@@ -1 +1,2 @@\n old\n+AKIAIOSFODNN7EXAMPLE\n"
    agent = _StubAgent({"unified_diff": secret_diff, "rationale": "r"})
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, agent, calls), max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""


def test_classify_sensitive_paths_denies_protected_and_credential_shaped() -> None:
    assert saw._classify_sensitive_paths(["README.md", "docs/x.md"], _DIFF) == []
    for denied in ("scripts/loop/x.py", "scripts/loop_adapters/y.py", "scripts/assurance/z.py",
                   "scripts/cs_assure.py", "scripts/cs_loop.py", ".claude/settings.json",
                   ".github/workflows/ci.yml", "research/paper.tex", "docs/paper/fig.py", ".gitmodules"):
        assert saw._classify_sensitive_paths([denied], _DIFF), denied
    for cred in ("config/.env", ".env.local", "certs/tls.pem", "secrets/app.key", "home/id_rsa"):
        assert saw._classify_sensitive_paths([cred], _DIFF), cred
    # GATE-AFFECTING config is denied so an agent can't neutralize the ruff/mypy/pytest gate (local or CI).
    for cfg in ("engine/tests/conftest.py", "conftest.py", "engine/pyproject.toml", "pytest.ini",
                "setup.cfg", "tox.ini", "ruff.toml", "mypy.ini", ".gitignore"):
        assert saw._classify_sensitive_paths([cfg], _DIFF), cfg
    # structural refusals: a symlink / mode / binary change in the raw diff
    assert saw._classify_sensitive_paths(["a"], "new file mode 120000\n")
    assert saw._classify_sensitive_paths(["a"], "GIT binary patch\n")
    # an over-large change is refused wholesale
    assert saw._classify_sensitive_paths([f"f{i}.py" for i in range(60)], _DIFF)


def test_scan_secrets_flags_added_lines_only() -> None:
    assert saw._scan_secrets(_DIFF) == []                                  # clean
    assert saw._scan_secrets("+AKIAIOSFODNN7EXAMPLE\n")                    # AWS key on an added line
    assert saw._scan_secrets("+ghp_" + "a" * 36 + "\n")                    # GitHub token
    assert saw._scan_secrets('+password = "hunter2-not-a-real-secret"\n')  # hardcoded credential
    assert saw._scan_secrets("-AKIAIOSFODNN7EXAMPLE\n") == []             # a REMOVED line is not flagged
    assert saw._scan_secrets("+++ b/AKIAIOSFODNN7EXAMPLE\n") == []        # the +++ header is not content


def test_the_default_cs_assure_runner_sanitizes_the_environment(tmp_path, monkeypatch) -> None:
    # candidate assurance (and the dev-tree gate) must run cs_assure with NO secrets in the environment.
    monkeypatch.setenv("GITHUB_TOKEN", "should-not-leak")
    captured: dict = {}

    def fake_run(*a, **k):
        captured.update(k)
        return _FakeProc(0, "{}")
    monkeypatch.setattr(saw.subprocess, "run", fake_run)
    saw._sanitized_cs_assure(tmp_path, "impact", "--base", "main")
    assert "GITHUB_TOKEN" not in captured["env"]  # secret-free
    assert captured["cwd"] == str(tmp_path)


def test_assure_candidate_fails_closed_on_a_refused_impact(tmp_path) -> None:
    # an unusable / refused impact record raises (the loop escalates) - never a silent publish.
    with pytest.raises(saw.WriteAdapterError):
        saw._assure_candidate(tmp_path, "main", lambda _r, *a: (2, "", "boom"))
    with pytest.raises(saw.WriteAdapterError):
        saw._assure_candidate(tmp_path, "main", lambda _r, *a: (0, "not json", ""))


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
    # the worktrees-dir resolver is now the SHARED helper in single_agent (write reuses it); it internally
    # calls single_agent.subprocess, so patch THAT module. inside a repo: under the git dir (outside the tree)...
    monkeypatch.setattr(sa.subprocess, "run",
                        lambda *a, **k: _FakeProc(0, "/abs/git/corpusstudio-loop/worktrees\n"))
    assert saw.default_worktrees_dir(tmp_path) == Path("/abs/git/corpusstudio-loop/worktrees")
    # ...and outside a repo it falls back to a worktree-local path (never inside the working tree implicitly).
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(128, "", "not a git repo"))
    assert saw.default_worktrees_dir(tmp_path) == tmp_path / ".corpusstudio-loop-worktrees"


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
