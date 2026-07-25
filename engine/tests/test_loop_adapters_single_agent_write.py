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

# The writable surface (7.1.2) is MODIFY-ONLY under engine/corpus_studio/**/*.py, so the "legit" change
# every happy-path test drives is an in-place edit of a product module.
_TARGET = "engine/corpus_studio/mod.py"
_DIFF = f"--- a/{_TARGET}\n+++ b/{_TARGET}\n@@ -1 +1 @@\n-old = 1\n+new = 2\n"


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


def _assure_runner(*, obligations=(), worker_reachable=()):
    """A fake cs_assure. `impact` + `worker-reachability` drive the STATIC CANDIDATE assurance in the
    executor (a fired obligation, or a changed path inside the worker import closure, blocks/routes the
    publish); `verify`/`doclint`/`changeset` drive the loop's own dev-tree OBSERVE/VERIFY. Neither
    candidate subcommand executes candidate code - assurance is deliberately static."""
    rec = {
        "impact": {"record_digest": "sha256:imp",
                   "payload": {"fired_obligations": [{"id": o, "severity": "blocking"} for o in obligations],
                               "base_policy_available": True, "change_set_fingerprint": "cs:x"},
                   "provenance": {"policy_digest": "sha256:p"}},
        "worker-reachability": {"payload": {"undeclared_reachable": list(worker_reachable),
                                            "worker_roots": [], "added_reachable": [],
                                            "distribution_impacting_paths": []}},
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
    target = root / _TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old = 1\n")
    (root / ".github").mkdir(exist_ok=True)
    (root / ".github" / "dependabot.yml").write_text("version: 2\n")
    (root / "secrets").mkdir(exist_ok=True)
    (root / "secrets" / "deploy.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nAKIAIOSFODNN7EXAMPLE\n")
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
                             run_cs_assure=assure or _cs_assure_green(),
                             ci_attested_safe=True)  # the operator's explicit CI attestation


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
    assert ref["changed_paths"] == [_TARGET]

    # the write landed on a PUSHED branch in the remote, carrying the applied change...
    assert "cs-agent/g1" in _g(remote, "branch", "--list", "cs-agent/g1").stdout
    assert _g(remote, "show", f"cs-agent/g1:{_TARGET}").stdout == "new = 2\n"
    # ...but the developer's MAIN working tree is pristine, and no worktree is left behind
    assert _g(root, "status", "--porcelain").stdout == ""
    assert (root / _TARGET).read_text() == "old = 1\n"
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
                            worktrees_dir=tmp_path / "w", gh_runner=gh_refuses,
                            run_cs_assure=_cs_assure_green(), ci_attested_safe=True)
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
    assert _g(remote, "show", f"cs-agent/g1:{_TARGET}").stdout == "new = 2\n"
    ca = state.review_state["candidate_assurance"]
    assert ca["published"] is True and ca["observation"] == "SUCCESS"
    # the candidate impact digest is an AUDIT field only - deliberately NOT in assurance_records, which
    # feeds the completeness check (see test_the_candidates_own_analysis_is_not_completeness_evidence)
    assert ca["impact_record_digest"] == "sha256:imp"


def test_candidate_assurance_targets_the_candidate_worktree_not_the_dev_tree(tmp_path: Path) -> None:
    # the whole point of 7.1.2: `impact` must ANALYZE the CANDIDATE worktree, never the dev tree. The
    # trusted-tool wrapper passes the candidate via --start-dir, so assert on that argument.
    seen: list = []
    base = _assure_runner()

    def recording(root, *a):
        if a and a[0] == "impact":
            seen.append((str(root), list(a)))
        return base(root, *a)
    root, _remote = _repo_with_remote(tmp_path)
    ctx = _build(tmp_path, root, _StubAgent(), [], assure=recording)
    run_loop(LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL), ctx)
    candidate_runs = [(r, a) for r, a in seen if "--start-dir" in a]
    assert candidate_runs, "candidate impact was never run against a candidate worktree"
    for run_root, argv in candidate_runs:
        target = Path(argv[argv.index("--start-dir") + 1]).resolve()
        assert (tmp_path / "wt").resolve() in target.parents   # analyzes the isolated worktree...
        assert target != root.resolve()                        # ...never the dev tree
        assert Path(run_root).resolve() == root.resolve()      # ...but EXECUTES the trusted dev-tree tool


def test_the_candidates_own_cs_assure_is_never_executed(tmp_path: Path) -> None:
    # HIGH (adversarial review): running the CANDIDATE's scripts/cs_assure.py puts <candidate>/scripts on
    # sys.path[0], so a candidate-added `scripts/argparse.py` would be imported+executed as the operator.
    # The wrapper must always run the TRUSTED dev-tree tool and merely POINT it at the candidate.
    root, _remote = _repo_with_remote(tmp_path)
    calls: list = []
    wrapped = saw._trusted_cs_assure(root, lambda r, *a: calls.append((str(r), list(a))) or (0, "{}", ""))
    candidate = tmp_path / "wt" / "apply-xyz"
    wrapped(candidate, "impact", "--base", "main")
    run_root, argv = calls[-1]
    assert Path(run_root) == root                                    # the TRUSTED tool is what executes
    assert argv[argv.index("--start-dir") + 1] == str(candidate)     # the candidate is only the SUBJECT
    # a dev-tree call is passed through untouched (no redundant --start-dir)
    wrapped(root, "verify", "--base", "main")
    assert calls[-1] == (str(root), ["verify", "--base", "main"])


ATTACKS: dict[str, str] = {
    # every one of these was a VERIFIED hole in the denylist design, or a known execution trick
    "case-variant of a protected path":
        "--- /dev/null\n+++ b/Scripts/argparse.py\n@@ -0,0 +1 @@\n+import os\n",
    "rename-from hides a denied path":
        "diff --git a/.github/dependabot.yml b/engine/corpus_studio/notes.py\n"
        "similarity index 100%\nrename from .github/dependabot.yml\nrename to engine/corpus_studio/notes.py\n",
    "conftest injection (a create)":
        "--- /dev/null\n+++ b/engine/corpus_studio/conftest.py\n@@ -0,0 +1 @@\n+collect_ignore_glob=['*']\n",
    "symlink into the operator filesystem":
        "diff --git a/engine/corpus_studio/l.py b/engine/corpus_studio/l.py\nnew file mode 120000\n"
        "--- /dev/null\n+++ b/engine/corpus_studio/l.py\n@@ -0,0 +1 @@\n+/etc/passwd\n",
    "exec-bit flip":
        "diff --git a/%(t)s b/%(t)s\nold mode 100644\nnew mode 100755\n" % {"t": _TARGET},
    "an edit outside the writable surface":
        "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n",
    "a secret smuggled into an ALLOWED in-place edit":
        "--- a/%(t)s\n+++ b/%(t)s\n@@ -1 +1,2 @@\n old = 1\n+K = 'AKIAIOSFODNN7EXAMPLE'\n" % {"t": _TARGET},
    "an oversized change":
        "--- a/%(t)s\n+++ b/%(t)s\n@@ -1 +1,71 @@\n old = 1\n" % {"t": _TARGET}
        + "".join("+l%d\n" % i for i in range(70)),
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_the_attack_matrix_publishes_nothing(name: str, tmp_path: Path) -> None:
    """Each known attack must publish NOTHING: no PR, no remote branch, and the dev tree left pristine."""
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    agent = _StubAgent({"unified_diff": ATTACKS[name], "rationale": "r"})
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, agent, calls), max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls], name
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == "", name
    assert _g(root, "status", "--porcelain").stdout == "", name
    assert "apply-" not in _g(root, "worktree", "list").stdout, name


def test_agent_prose_is_scanned_and_bounded() -> None:
    # HIGH (4th adversarial pass): the rationale reaches the commit message AND the PR body verbatim, so
    # a secret moved out of the diff and into the prose bypassed every content check. Prose is scanned.
    assert saw._scan_text("Fix a typo.", "the rationale") == []
    assert saw._scan_text("see AKIAIOSFODNN7EXAMPLE", "the rationale")
    assert saw._scan_text("-----BEGIN RSA PRIVATE KEY-----", "the rationale")
    assert saw._scan_text("x" * 9000, "the rationale")            # bounded
    assert saw._scan_text("ok \u202e evil", "the rationale")       # bidi override


def test_a_secret_in_the_rationale_blocks_the_publish(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    agent = _StubAgent({"unified_diff": _DIFF,                      # a perfectly legal diff...
                        "rationale": "Tidy.\n\n-----BEGIN RSA PRIVATE KEY-----\n"})  # ...secret in the prose
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, agent, calls), max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""


def test_publishing_requires_an_explicit_operator_ci_attestation(tmp_path: Path) -> None:
    """Pushing EXECUTES the candidate in CI before review, and this adapter cannot verify someone else's
    CI by parsing it (the previous text-scan failed open five measured ways, and read the dev tree rather
    than the workflows at base). So the operator attests, and the DEFAULT is refuse."""
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    ctx = saw.build_context(root, "main", agent_client=_StubAgent(), proposals_dir=tmp_path / "p",
                            worktrees_dir=tmp_path / "wt", gh_runner=_fake_gh(calls),
                            run_cs_assure=_cs_assure_green())      # <- no attestation
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx, max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]           # nothing published by default
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""
    assert saw._publish_precondition_unmet(False) is not None
    assert saw._publish_precondition_unmet(True) is None


def test_the_writable_surface_is_matched_case_folded_and_path_aware() -> None:
    # fnmatch is UNUSABLE here: fnmatch('engine/tests/test_x/conftest.py','engine/tests/test_*.py') is True
    # (it would admit an auto-executed conftest) and it REJECTS engine/corpus_studio/cli.py. Our compiled
    # matcher gets both right: '*' never crosses '/', '**/' spans directories.
    assert saw._path_is_writable("engine/corpus_studio/cli.py")
    assert saw._path_is_writable("engine/corpus_studio/deep/nested/mod.py")
    for outside in ("README.md", "scripts/loop/x.py", "engine/tests/test_a.py",
                    "engine/corpus_studio/notes.md", "engine/corpus_studio/sub/x.txt"):
        assert not saw._path_is_writable(outside), outside
    # case variants can only SHRINK the surface, never widen it (repo carries core.ignorecase=true)
    assert not saw._path_is_writable("Scripts/argparse.py")
    assert saw._path_is_writable("ENGINE/CORPUS_STUDIO/MOD.PY")  # folds INTO the allowed surface, still gated


def test_compile_pathglob_never_crosses_a_slash() -> None:
    rx = saw._compile_pathglob("engine/tests/test_*.py")
    assert rx.match("engine/tests/test_a.py")
    assert not rx.match("engine/tests/test_x/conftest.py")   # the fnmatch fail-open case
    deep = saw._compile_pathglob("a/**/*.py")
    assert deep.match("a/b.py") and deep.match("a/b/c/d.py") and not deep.match("a/b/c.txt")


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


def test_the_confined_worktree_is_outside_the_repository_and_its_git_dir(tmp_path) -> None:
    # HIGH (5th adversarial pass): this dir used to be <git-dir>/corpusstudio-loop/worktrees, so the
    # untrusted agent's cwd sat THREE levels under the repo's own .git. Appending to .git/config
    # (e.g. core.fsmonitor=/tmp/pwn.sh) gave persistent code execution as the operator with the full
    # un-sanitized environment - measured. The confined checkout must live outside the repo entirely.
    d = saw.default_worktrees_dir(tmp_path)
    repo = tmp_path.resolve()
    assert repo not in d.parents and d != repo          # not inside the working tree...
    assert (repo / ".git") not in d.parents              # ...and emphatically not inside .git
    # a naive walk up from the agent's cwd must not reach the repo or its git dir
    escaped = (d / "propose-x" / ".." / ".." / "..").resolve()
    assert repo not in escaped.parents and escaped != repo and escaped != repo / ".git"
    # distinct repos get distinct roots (no cross-repo collision)
    assert saw.default_worktrees_dir(tmp_path) != saw.default_worktrees_dir(tmp_path / "other")


def test_adapter_git_pins_hooks_and_fsmonitor_off(tmp_path, monkeypatch) -> None:
    # core.hooksPath=/dev/null does NOT suppress core.fsmonitor (measured), and BOTH run with the
    # operator's full environment - so both must be disabled on every invocation, in BOTH adapters.
    seen: list = []
    monkeypatch.setattr(sa.subprocess, "run", lambda a, **k: seen.append(a) or _FakeProc(0, ""))
    sa._git(tmp_path, "status")
    assert "core.hooksPath=/dev/null" in seen[-1] and "core.fsmonitor=" in seen[-1]
    monkeypatch.setattr(saw.subprocess, "run", lambda a, **k: seen.append(a) or _FakeProc(0, ""))
    saw._git(tmp_path, "status")
    assert "core.hooksPath=/dev/null" in seen[-1] and "core.fsmonitor=" in seen[-1]


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


def test_the_worker_import_closure_blocks_the_publish(tmp_path: Path) -> None:
    # HIGH (4th pass): the worker-closure OBLIGATION only flags the 7 DECLARED worker files, but the
    # worker's REAL import closure is far wider (43 undeclared-reachable modules on this repo) and those
    # sit INSIDE the writable surface - so editing one changes worker bytes with the lineage gate silent.
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    assure = _assure_runner(worker_reachable=[_TARGET])   # the candidate's file IS worker-reachable
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), calls, assure=assure), max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]
    assert _g(remote, "branch", "--list", "cs-agent/g1").stdout == ""
    assert state.review_state["candidate_assurance"]["observation"] == "WORKER_LINEAGE_IMPACT"


def test_an_unusable_worker_reachability_record_fails_closed(tmp_path: Path) -> None:
    # an uncomputable closure must never read as "touches no worker code"
    with pytest.raises(saw.WriteAdapterError):
        saw._worker_reachable_paths(tmp_path, "main", lambda _r, *a: (2, "", "refused"))
    with pytest.raises(saw.WriteAdapterError):
        saw._worker_reachable_paths(tmp_path, "main", lambda _r, *a: (0, "not json", ""))


def test_the_candidates_own_analysis_is_not_completeness_evidence(tmp_path: Path) -> None:
    # The candidate's OWN static analysis must NOT enter state.assurance_records: that index feeds the
    # semantic completeness check, and seeding it lets an untrusted executor supply the evidence a later
    # DETERMINISTIC criterion is proven by (the cross-step self-certification vector).
    root, _remote = _repo_with_remote(tmp_path)
    state = LoopState(goal="tidy", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), []))
    assert state.review_state["candidate_assurance"]["impact_record_digest"] == "sha256:imp"  # audited...
    assert "sha256:imp" not in state.assurance_records                                        # ...not evidence


def test_an_allowlist_refusal_is_an_expected_operational_error() -> None:
    # A denied path is this adapter's PRIMARY SAFETY MECHANISM and normal operation - it must escalate
    # labelled as an operational refusal, not as "UNEXPECTED ... (a likely controller bug)" with a traceback.
    from loop.orchestrate import _EXPECTED_DISPATCH_ERRORS
    assert issubclass(saw.WriteAdapterError, _EXPECTED_DISPATCH_ERRORS)
    assert issubclass(saw.AgentError, _EXPECTED_DISPATCH_ERRORS)


def test_a_gh_failure_after_the_push_still_records_the_live_branch(tmp_path: Path) -> None:
    # The push already put an agent-authored branch on the remote (CI may be running on it). A human
    # triaging the escalation must SEE that, not an empty review_state.
    root, remote = _repo_with_remote(tmp_path)

    def gh_fails(*argv: str) -> tuple[int, str, str]:
        return (1, "", "boom") if argv[:2] == ("pr", "create") else (0, "", "")
    ctx = saw.build_context(root, "main", agent_client=_StubAgent(), proposals_dir=tmp_path / "p",
                            worktrees_dir=tmp_path / "wt", gh_runner=gh_fails,
                            run_cs_assure=_cs_assure_green(), ci_attested_safe=True)
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx, max_steps=25)
    assert state.current_phase is Phase.ESCALATED
    assert "cs-agent/g1" in _g(remote, "branch", "--list", "cs-agent/g1").stdout   # the branch IS live...
    ca = state.review_state["candidate_assurance"]                                  # ...and it is recorded
    assert ca["published"] is True and "PR not yet opened" in ca["reason"]


def test_a_pre_existing_credential_shaped_line_does_not_brick_a_file(tmp_path: Path) -> None:
    """HIGH (6th pass): the loose credential heuristic ran over the WHOLE blob, so cli.py - which already
    contains `api_key: Optional[str] = typer.Option(` at main - could NEVER be edited, and could never fix
    the line blocking it. High-confidence token formats still scan the whole blob (that is what catches a
    `copy from`); the loose heuristic now only sees ADDED lines."""
    root, _remote = _repo_with_remote(tmp_path)
    target = root / _TARGET
    target.write_text("api_key: Optional[str] = option(\nold = 1\n")   # pre-existing, credential-SHAPED
    _g(root, "add", "-A")
    _g(root, "commit", "-q", "-m", "pre-existing")
    base = _g(root, "rev-parse", "HEAD").stdout.strip()
    diff = (f"--- a/{_TARGET}\n+++ b/{_TARGET}\n@@ -1,2 +1,3 @@\n"
            " api_key: Optional[str] = option(\n+# a docstring fix\n old = 1\n")
    with saw._apply_worktree(root, base, tmp_path / "wt") as wt:
        saw._git(wt, "apply", "--index", "-", stdin=diff)
        assert saw._classify_candidate_changes(wt, base) == []      # the innocent edit is PUBLISHABLE
    # ...but a real token the candidate ADDS is still refused
    bad = (f"--- a/{_TARGET}\n+++ b/{_TARGET}\n@@ -1,2 +1,3 @@\n"
           " api_key: Optional[str] = option(\n+K = \"gho_" + "a" * 36 + "\"\n old = 1\n")
    with saw._apply_worktree(root, base, tmp_path / "wt") as wt:
        saw._git(wt, "apply", "--index", "-", stdin=bad)
        assert saw._classify_candidate_changes(wt, base)


def test_ordinary_prose_and_code_are_not_flagged_as_secrets() -> None:
    # false positives kill the rung: every one of these is innocent and must pass.
    for ok in ("refactor the task-oriented-configuration helper",
               "fix the disk-space-management-check docstring",
               "api_key: Optional[str] = typer.Option(",
               'password = os.environ["PW"]'):
        assert saw._scan_text(ok, "the rationale") == [], ok
    # ...while real credential shapes still fail closed
    for bad in ("K = \"gho_" + "a" * 36 + "\"", "oauth_token: gho_" + "a" * 36,
                'password = "hunter2hunter2"', "-----BEGIN RSA PRIVATE KEY-----"):
        assert saw._scan_text(bad, "the rationale"), bad


def test_the_cache_dir_must_be_absolute(tmp_path, monkeypatch) -> None:
    # MEDIUM (6th pass): a RELATIVE XDG_CACHE_HOME made `git -C <repo> worktree add <relpath>` resolve
    # against the REPO ROOT, putting the confined agent back inside the tree with the operator's .git a
    # fixed walk away - the pass-5 RCE restored silently. Anything non-absolute is discarded.
    monkeypatch.setenv("XDG_CACHE_HOME", ".cache")
    monkeypatch.setenv("HOME", str(tmp_path))
    d = sa.default_worktrees_dir(tmp_path)
    assert d.is_absolute() and tmp_path.resolve() in d.parents or d.is_absolute()
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert sa.default_worktrees_dir(tmp_path).is_absolute()
