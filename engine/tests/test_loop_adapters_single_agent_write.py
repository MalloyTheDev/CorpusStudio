"""Phase 7.1 - the write-capable single-agent adapter (scripts/loop_adapters/single_agent_write.py).

Pins the WRITE path end-to-end AND its safety: the agent's sealed diff is applied in an ISOLATED worktree
(never the main tree), committed on a fresh branch, pushed, and a PR is opened - while the developer's
working tree is left pristine, no merge ever happens, and any failure (a diff that won't apply, a drifted
apply) fails closed. A local bare remote makes ``git push`` work offline; ``gh`` is faked.
"""

from __future__ import annotations

import json
import os
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
from loop_adapters.target_profile import load_profile  # noqa: E402

_PROFILE = load_profile("corpusstudio")


def _path_is_writable_p(path: str) -> bool:
    """`_path_is_writable` under the CorpusStudio profile (the surface these tests assert)."""
    return saw._path_is_writable(path, _PROFILE)
from loop.controller import LoopState, Phase  # noqa: E402
from loop.orchestrate import run_loop  # noqa: E402

# The writable surface (7.1.2) is MODIFY-ONLY under engine/corpus_studio/**/*.py, so the "legit" change
# every happy-path test drives is an in-place edit of a product module.
_TARGET = "engine/corpus_studio/mod.py"
_FILES = {_TARGET: "new = 2\n"}      # the agent returns WHOLE FILES; git computes the diff
_DIFF = f"--- a/{_TARGET}\n+++ b/{_TARGET}\n@@ -1 +1 @@\n-old = 1\n+new = 2\n"


class _StubAgent:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response if response is not None else {"files": _FILES, "rationale": "make it new"}
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
    def run(root, *a):
        sub = a[0] if a else ""
        body = json.loads(json.dumps(rec.get(sub, {})))     # deep copy per call
        if sub in ("impact", "worker-reachability"):
            # answer about the SUBJECT the adapter asked about. _trusted_cs_assure runs the TRUSTED
            # dev-tree tool and points it at the candidate with --start-dir, so the subject is that dir -
            # not `root`, which is the dev repo.
            argv = list(a)
            subject = argv[argv.index("--start-dir") + 1] if "--start-dir" in argv else str(root)
            head = subprocess.run(["git", "-C", subject, "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            body.setdefault("payload", {})["scope"] = "head"
            body.setdefault("provenance", {})["head_oid"] = head
        return (0, json.dumps(body), "")
    return run


def _cs_assure_green():
    return _assure_runner()


def _g(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # Fixture commits use PINNED dates so a base commit is reproducible across repos. Without this the
    # base oid depends on the wall clock (second resolution), which made the content-addressing test flaky:
    # two fixture repos straddling a second boundary get different bases, hence different candidates.
    env = {**os.environ, "GIT_AUTHOR_DATE": "@0 +0000", "GIT_COMMITTER_DATE": "@0 +0000"}
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True,
                          env=env)


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
    assert ref["branch"].startswith("cs-agent/g1-") and ref["pr"] == "https://example/pull/1"
    assert ref["changed_paths"] == [_TARGET]

    # the write landed on a PUSHED branch in the remote, carrying the applied change...
    assert "cs-agent/g1" in _g(remote, "branch", "--list", "cs-agent/g1-*").stdout
    br = [b.strip("* ") for b in _g(remote, "branch", "--list").stdout.splitlines() if "cs-agent" in b][0]
    assert _g(remote, "show", f"{br}:{_TARGET}").stdout == "new = 2\n"
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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout == ""
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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout == ""
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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout == ""
    ca = state.review_state["candidate_assurance"]
    assert ca["published"] is False and ca["observation"] == "WORKER_LINEAGE_IMPACT"


def test_a_clear_candidate_publishes_and_records_the_assurance(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    state = LoopState(goal="tidy", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), calls))
    # a clear candidate publishes (branch pushed by refspec from a DETACHED apply worktree) + records it
    br = [b.strip("* ") for b in _g(remote, "branch", "--list").stdout.splitlines() if "cs-agent" in b][0]
    assert _g(remote, "show", f"{br}:{_TARGET}").stdout == "new = 2\n"
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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout == "", name
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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout == ""


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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout == ""
    assert saw._publish_precondition_unmet(False) is not None
    assert saw._publish_precondition_unmet(True) is None


def test_the_writable_surface_is_matched_case_folded_and_path_aware() -> None:
    # fnmatch is UNUSABLE here: fnmatch('engine/tests/test_x/conftest.py','engine/tests/test_*.py') is True
    # (it would admit an auto-executed conftest) and it REJECTS engine/corpus_studio/cli.py. Our compiled
    # matcher gets both right: '*' never crosses '/', '**/' spans directories.
    assert _path_is_writable_p("engine/corpus_studio/cli.py")
    assert _path_is_writable_p("engine/corpus_studio/deep/nested/mod.py")
    for outside in ("README.md", "scripts/loop/x.py", "engine/tests/test_a.py",
                    "engine/corpus_studio/notes.md", "engine/corpus_studio/sub/x.txt"):
        assert not _path_is_writable_p(outside), outside
    # case variants can only SHRINK the surface, never widen it (repo carries core.ignorecase=true)
    assert not _path_is_writable_p("Scripts/argparse.py")
    assert _path_is_writable_p("ENGINE/CORPUS_STUDIO/MOD.PY")  # folds INTO the allowed surface, still gated


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
    assert saw._sanitize_branch_suffix("Fix bug #5 (README)!").startswith("fix-bug-5-readme-")
    assert ".." not in saw._sanitize_branch_suffix("a..b")
    assert saw._sanitize_branch_suffix("").startswith("goal-")
    assert saw._sanitize_branch_suffix("...--__").startswith("goal-")
    assert len(saw._sanitize_branch_suffix("x" * 100)) <= 50   # 40-char slug + "-" + 8-hex digest


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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout == ""
    assert state.review_state["candidate_assurance"]["observation"] == "WORKER_LINEAGE_IMPACT"


def test_an_unusable_worker_reachability_record_fails_closed(tmp_path: Path) -> None:
    # an uncomputable closure must never read as "touches no worker code"
    with pytest.raises(saw.WriteAdapterError):
        saw._worker_reachable_paths(tmp_path, "b" * 40, lambda _r, *a: (2, "", "refused"))
    with pytest.raises(saw.WriteAdapterError):
        saw._worker_reachable_paths(tmp_path, "b" * 40, lambda _r, *a: (0, "not json", ""))


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
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout.strip()   # the branch IS live...
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
        tree = saw._git(wt, "write-tree").stdout.strip()
        assert saw._classify_candidate_changes(wt, base, tree, _PROFILE) == []   # the innocent edit is PUBLISHABLE
    # ...but a real token the candidate ADDS is still refused
    bad = (f"--- a/{_TARGET}\n+++ b/{_TARGET}\n@@ -1,2 +1,3 @@\n"
           " api_key: Optional[str] = option(\n+K = \"gho_" + "a" * 36 + "\"\n old = 1\n")
    with saw._apply_worktree(root, base, tmp_path / "wt") as wt:
        saw._git(wt, "apply", "--index", "-", stdin=bad)
        tree = saw._git(wt, "write-tree").stdout.strip()
        assert saw._classify_candidate_changes(wt, base, tree, _PROFILE)


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


# ------------------------------------------------------- per-rule isolation (mutation-resistant, 7th pass)
# A 7th adversarial pass MUTATION-TESTED the suite: disabling the ALLOWLIST gate (`if False:`) left all 36
# tests green, because the only out-of-surface attack case (README.md) is also caught by the basename rule.
# Eleven other guards likewise survived deletion. End-to-end "nothing was published" assertions cannot see
# WHICH rule fired, so each guard now has a case that trips ONLY it.


_CLASSIFY_SEQ = [0]


def _classify_one(tmp_path: Path, path: str, body: str = "x = 1\n",
                  base_body: str = "x = 0\n") -> list[str]:
    """Apply a minimal in-place MODIFY of `path` in a real worktree and return the classifier's reasons."""
    _CLASSIFY_SEQ[0] += 1
    tmp_path = tmp_path / f"c{_CLASSIFY_SEQ[0]}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    root, _remote = _repo_with_remote(tmp_path)
    f = root / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(base_body)
    _g(root, "add", "-A")
    _g(root, "commit", "-q", "-m", "seed")
    base = _g(root, "rev-parse", "HEAD").stdout.strip()
    n_old, n_new = len(base_body.splitlines()), len(body.splitlines())
    diff = (f"--- a/{path}\n+++ b/{path}\n@@ -1,{n_old} +1,{n_new} @@\n"
            + "".join(f"-{ln}\n" for ln in base_body.splitlines())
            + "".join(f"+{ln}\n" for ln in body.splitlines()))
    with saw._apply_worktree(root, base, tmp_path / "wt") as wt:
        saw._git(wt, "apply", "--index", "-", stdin=diff)
        tree = saw._git(wt, "write-tree").stdout.strip()
        return saw._classify_candidate_changes(wt, base, tree, _PROFILE)


def test_rule_allowlist_is_the_only_thing_blocking_an_out_of_surface_path(tmp_path: Path) -> None:
    # `engine/other/mod.py` has a SAFE basename, status M and mode 100644 - every other rule passes - so
    # only the allowlist can refuse it. Disabling the allowlist makes this test fail (the surviving mutant).
    reasons = _classify_one(tmp_path, "engine/other/mod.py")
    assert reasons == ["engine/other/mod.py (outside the writable surface)"]
    assert _classify_one(tmp_path, _TARGET) == []          # the in-surface control publishes


def test_rule_basename_is_the_only_thing_blocking_a_bad_module_name(tmp_path: Path) -> None:
    # inside the allowlist, so only _SAFE_BASENAME can refuse it
    reasons = _classify_one(tmp_path, "engine/corpus_studio/Evil.py")
    assert reasons and "unsafe module basename" in reasons[0]


def test_rule_line_bound_is_the_only_thing_blocking_a_long_line(tmp_path: Path) -> None:
    reasons = _classify_one(tmp_path, _TARGET, body="y = '" + "a" * 500 + "'\n")
    assert reasons and "-byte line" in reasons[0]


def test_rule_invisible_characters_are_the_only_thing_blocking(tmp_path: Path) -> None:
    reasons = _classify_one(tmp_path, _TARGET, body="x = 1  # ​\n")
    assert reasons and "invisible/format character" in reasons[0]


def test_rule_added_line_secret_scan_is_the_only_thing_blocking(tmp_path: Path) -> None:
    reasons = _classify_one(tmp_path, _TARGET, body='password = "hunter2hunter2hunter2"\n')
    assert reasons and "apparent hardcoded credential" in reasons[0]


def test_rule_added_line_scan_sees_content_that_looks_like_a_diff_header(tmp_path: Path) -> None:
    # MEASURED bypass (7th pass): a content line beginning with "++" renders as "+++..." in the diff and
    # was skipped as if it were the `+++ b/<path>` file header, publishing a real credential.
    reasons = _classify_one(tmp_path, _TARGET,
                            body='"""doc\n++password = "hunter2hunter2hunter2"\n"""\n')
    assert reasons and "apparent hardcoded credential" in reasons[0]


def test_rule_high_confidence_blob_scan_is_the_only_thing_blocking(tmp_path: Path) -> None:
    reasons = _classify_one(tmp_path, _TARGET, body='K = "gho_' + "a" * 36 + '"\n')
    assert reasons and "GitHub token" in reasons[0]


def test_branch_names_are_unique_per_goal_even_when_slugs_collide() -> None:
    # MEASURED (7th pass): two goal ids agreeing in the first 40 slug chars mapped to ONE branch, so the
    # second goal could never publish (non-fast-forward) - and a retry after a transient gh failure stuck.
    a = saw._sanitize_branch_suffix("g-engine-corpus-studio-platform-environment_manager.py")
    b = saw._sanitize_branch_suffix("g-engine-corpus-studio-platform-environments.py")
    assert a != b
    assert saw._sanitize_branch_suffix("x") == saw._sanitize_branch_suffix("x")   # deterministic


def test_the_commit_message_does_not_claim_a_human_review_that_has_not_happened(tmp_path: Path) -> None:
    # MEASURED (7th pass): the commit asserted "[single-agent proposal, human-reviewed]" while publishing
    # autonomously - a false provenance claim baked into the artefact a human later merges.
    root, remote = _repo_with_remote(tmp_path)
    run_loop(LoopState(goal="tidy", goal_id="g1", current_phase=Phase.RECEIVE_GOAL),
             _build(tmp_path, root, _StubAgent(), []))
    branch = [b.strip("* ") for b in _g(remote, "branch", "--list").stdout.splitlines() if "cs-agent" in b][0]
    msg = _g(remote, "log", "-1", "--format=%B", branch).stdout
    assert "human-reviewed]" not in msg
    assert "NOT yet human-reviewed" in msg


def _seed(tmp_path: Path, files: dict) -> tuple[Path, str]:
    """A fresh repo containing `files`; returns (root, base_oid)."""
    _CLASSIFY_SEQ[0] += 1
    d = tmp_path / f"s{_CLASSIFY_SEQ[0]}"
    d.mkdir(parents=True, exist_ok=True)
    root, _remote = _repo_with_remote(d)
    for rel, content in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(content if isinstance(content, bytes) else content.encode())
    _g(root, "add", "-A")
    _g(root, "commit", "-q", "-m", "seed")
    return root, _g(root, "rev-parse", "HEAD").stdout.strip()


def _classify_staged(tmp_path: Path, root: Path, base: str, mutate) -> list[str]:
    """Stage a change directly in a candidate worktree and classify it. The classifier reads the INDEX
    (via git plumbing), not a diff, so staging is the faithful way to isolate ONE rule - hand-written
    hunks are brittle and test the patch format rather than the guard."""
    with saw._apply_worktree(root, base, tmp_path / "wtx") as wt:
        mutate(wt)
        saw._git(wt, "add", "-A")
        tree = saw._git(wt, "write-tree").stdout.strip()
        return saw._classify_candidate_changes(wt, base, tree, _PROFILE)


def _write(wt: Path, rel: str, content) -> None:
    f = wt / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content if isinstance(content, bytes) else content.encode())


def test_rule_status_must_be_modify_is_the_only_thing_blocking_a_create(tmp_path: Path) -> None:
    # a CREATE inside the allowlist with a safe basename: allowlist/basename/mode/content all pass, so
    # only the status!=M rule can refuse it. Creates are how conftest injection + stdlib shadowing land.
    root, base = _seed(tmp_path, {_TARGET: "x = 0\n"})
    reasons = _classify_staged(tmp_path, root, base,
                               lambda wt: _write(wt, "engine/corpus_studio/newmod.py", "x = 1\n"))
    assert reasons and "only in-place modification is allowed" in reasons[0], reasons


def test_rule_patch_byte_bound_is_the_only_thing_blocking(tmp_path: Path) -> None:
    # 30 lines of ~340 B: under the 400 B/line bound and the 60-line bound, but over the 8 KiB patch bound.
    root, base = _seed(tmp_path, {_TARGET: "x = 0\n"})
    body = "".join(f"v{i} = '{'a' * 330}'\n" for i in range(30))
    reasons = _classify_staged(tmp_path, root, base, lambda wt: _write(wt, _TARGET, body))
    assert reasons and any("bytes (>" in r for r in reasons), reasons


def test_rule_max_changed_paths_is_the_only_thing_blocking(tmp_path: Path) -> None:
    # three tiny in-surface modifies: every per-file rule passes, so only the path-count bound refuses.
    files = {f"engine/corpus_studio/m{i}.py": "x = 0\n" for i in range(3)}
    root, base = _seed(tmp_path, files)

    def mutate(wt: Path) -> None:
        for rel in files:
            _write(wt, rel, "x = 1\n")
    reasons = _classify_staged(tmp_path, root, base, mutate)
    assert reasons and any("paths (>" in r for r in reasons), reasons


def test_rule_blob_size_cap_is_the_only_thing_blocking(tmp_path: Path) -> None:
    # the RESULTING blob exceeds the 1 MiB read cap while every line stays short; the read-side cap is the
    # only rule that can see it (the per-line and per-patch bounds are computed elsewhere).
    root, base = _seed(tmp_path, {_TARGET: "x = 0\n"})
    big = "".join(f"v{i} = {i}\n" for i in range(140_000))          # ~2 MB, all short lines
    # this guard RAISES from _read_blob rather than appending a reason - both fail closed, and the raise
    # escalates the loop (WriteAdapterError is an EXPECTED operational refusal).
    with pytest.raises(saw.WriteAdapterError, match="oversized blob"):
        _classify_staged(tmp_path, root, base, lambda wt: _write(wt, _TARGET, big))


def test_rule_binary_refusal_is_the_only_thing_blocking(tmp_path: Path) -> None:
    # bytes that are not valid UTF-8: _read_blob returns None and only that branch refuses.
    root, base = _seed(tmp_path, {_TARGET: "x = 0\n"})
    reasons = _classify_staged(tmp_path, root, base,
                               lambda wt: _write(wt, _TARGET, b"x = 1\n\xff\xfe\x00binary\n"))
    assert reasons and any("binary or non-UTF-8" in r for r in reasons), reasons


# ------------------------------------------------------------- independent-review residuals (post-merge)


def test_invisible_characters_in_printable_categories_are_refused() -> None:
    """Category alone was NOT sufficient: U+FE0F (Mn) and U+3164 (Lo) are invisible yet live in printable
    categories AND are valid Python identifier characters, so two names that render identically are
    distinct to the interpreter. The clean exploit is a string literal - a reviewer reads
    `if mode == "strict"` while the literal carries a VS16, so the branch is silently dead code."""
    for cp in (0xFE0F, 0x3164, 0x200B, 0xE0041, 0x00AD, 0x202E, 0x2028):
        assert saw._forbidden_char("x = 1  # a" + chr(cp) + "b"), hex(cp)
    # ...while ordinary non-ASCII source stays legal (77 of 136 files in the surface use an em dash)
    assert saw._forbidden_char('x = 1  # an em dash — and CJK 日') is None


def test_the_added_line_scan_compares_blobs_not_diff_text(tmp_path: Path) -> None:
    """Two prefix-sniffing versions of this scan were defeated: a content line starting '++' renders as
    '+++...' (read as a file header), and anchoring on a preceding '--- ' failed too because a REMOVED
    content line starting '-- ' also renders as '--- '. Both measured, both published a real credential.
    Comparing realized BLOBS has no such surface."""
    root, base = _seed(tmp_path, {_TARGET: '"""doc\n-- note\n"""\n'})
    for body in ('"""doc\n++ password = "sup3rsecretvalue123"\n"""\n',      # the '-- ' predecessor attack
                 '"""doc\n-- note\n++password = "sup3rsecretvalue123"\n"""\n',
                 '"""doc\n-- note\npassword = "sup3rsecretvalue123"\n"""\n'):
        reasons = _classify_staged(tmp_path, root, base, lambda wt, b=body: _write(wt, _TARGET, b))
        assert reasons and "hardcoded credential" in reasons[0], body
    # an unchanged pre-existing line can never block, and a legitimate edit still publishes
    assert _classify_staged(tmp_path, root, base,
                            lambda wt: _write(wt, _TARGET, '"""doc tidied\n-- note\n"""\n')) == []


def test_a_well_formed_but_keyless_worker_reachability_record_fails_closed(tmp_path: Path) -> None:
    """SURVIVING MUTANT found by the independent review: the existing test only covered exit!=0 and
    non-JSON, never a well-formed-JSON payload MISSING the keys - which is the exact shape (schema drift,
    a renamed key, a null) the guard was written for. With it disabled, an empty closure meant the worker
    lineage gate went silent while 50 of the 136 files in the writable surface are worker-reachable."""
    for payload in ({}, {"reachable_undeclared": []}, {"undeclared_reachable": None, "worker_roots": [],
                                                       "added_reachable": [], "distribution_impacting_paths": []}):
        with pytest.raises(saw.WriteAdapterError, match="worker-reachability"):
            saw._worker_reachable_paths(tmp_path, "b" * 40,
                                        lambda _r, *a, p=payload: (0, json.dumps({"payload": p}), ""))
    # a complete record still resolves
    good = {"undeclared_reachable": ["a.py"], "worker_roots": [], "added_reachable": [],
            "distribution_impacting_paths": []}
    assert saw._worker_reachable_paths(
        tmp_path, "main", lambda _r, *a: (0, json.dumps({"payload": good}), "")) == frozenset({"a.py"})


def test_a_blocking_obligation_that_is_not_human_gated_still_blocks(tmp_path: Path) -> None:
    """SURVIVING MUTANT: no test ever fired a BLOCKING obligation that was not ALSO human-gated, so the
    POLICY_BLOCK branch could be deleted with the suite green."""
    def assure(_r, *a):
        if a and a[0] == "impact":
            return (0, json.dumps({"record_digest": "sha256:i", "payload": {
                "scope": "head",
                "fired_obligations": [{"id": "evaluation-honesty", "severity": "blocking"}],
                "base_policy_available": True, "change_set_fingerprint": "cs:x"},
                "provenance": {"head_oid": "c" * 40}}), "")
        return (0, json.dumps({"payload": {"undeclared_reachable": [], "worker_roots": [],
                                           "added_reachable": [], "distribution_impacting_paths": []}}), "")
    observation, reason, _d = saw._assure_candidate(tmp_path, "b" * 40, assure,
                                                    ["engine/corpus_studio/x.py"], "c" * 40,
                                                    _PROFILE)
    assert observation is saw.Observation.POLICY_BLOCK and "evaluation-honesty" in reason


def test_the_added_line_scan_also_covers_newly_added_files(tmp_path: Path) -> None:
    """Sourcery (#707): the blob-compare scan skipped status 'A', so a brand-new file's credentials were
    never seen by the loose heuristic. Not exploitable today - a create is refused outright by modify-only
    - but the heuristic must not silently stop covering new content if that rule is ever relaxed. Asserted
    against the scan DIRECTLY so it cannot pass merely because another gate fired."""
    root, base = _seed(tmp_path, {_TARGET: "x = 0\n"})
    reasons = _classify_staged(
        tmp_path, root, base,
        lambda wt: _write(wt, "engine/corpus_studio/newmod.py", 'password = "sup3rsecretvalue123"\n'))
    assert any("hardcoded credential" in r for r in reasons), reasons   # the scan saw it...
    assert any("only in-place modification" in r for r in reasons), reasons  # ...and modify-only also fired


def test_default_ignorable_lookup_matches_the_declared_ranges() -> None:
    # the bisect fast path must agree with the spec ranges it replaced (Sourcery perf note, #707)
    for lo, hi in saw._DEFAULT_IGNORABLE_RANGES:
        assert saw._is_default_ignorable(chr(lo)) and saw._is_default_ignorable(chr(hi))
        if lo > 0:
            assert not saw._is_default_ignorable(chr(lo - 1)) or any(
                a <= lo - 1 <= b for a, b in saw._DEFAULT_IGNORABLE_RANGES)
    for ordinary in ("a", "—", "日", " ", "\t", "\n"):
        assert not saw._is_default_ignorable(ordinary)


# ------------------------------------------------- closure of the independent review's remaining tail


def test_an_unreadable_blob_fails_closed_rather_than_scanning_empty_content(tmp_path: Path) -> None:
    """`_read_blob` ignored `git cat-file`'s exit status, so an unreadable blob yielded EMPTY content -
    and an empty string passes EVERY content check (secret scan, line bound, character check) on a blob
    that was never actually read. That is fail-OPEN in the middle of the security core."""
    root, _remote = _repo_with_remote(tmp_path)
    # an absent oid fails closed at the SIZE probe...
    with pytest.raises(saw.WriteAdapterError):
        saw._read_blob(root, "0" * 40, "engine/corpus_studio/mod.py")

    # ...and a blob that sizes fine but whose READ fails must also fail closed, not scan "" as clean.
    real = saw.subprocess.run

    def flaky(argv, **kw):
        if "blob" in argv:
            return type("P", (), {"returncode": 128, "stdout": b"", "stderr": b"unreadable"})()
        return real(argv, **kw)
    oid = _g(root, "rev-parse", f"HEAD:{_TARGET}").stdout.strip()
    import unittest.mock as _m
    with _m.patch.object(saw.subprocess, "run", flaky), \
            pytest.raises(saw.WriteAdapterError, match="cannot read blob"):
        saw._read_blob(root, oid, _TARGET)


def test_the_pr_body_declares_it_is_machine_authored() -> None:
    body = saw._pr_body("Tidy a docstring.")
    assert "NOT yet human-reviewed" in body and "Machine-authored" in body
    assert "Tidy a docstring." in body
    # an agent that supplies no rationale still gets the disclosure
    assert "NOT yet human-reviewed" in saw._pr_body("")


def test_the_proposal_is_referenced_even_when_pr_create_fails(tmp_path: Path) -> None:
    """A `gh pr create` failure used to leave a pushed branch on the remote with NO record on the loop
    state, so the human triaging the escalation could not find what had been published."""
    root, remote = _repo_with_remote(tmp_path)

    def gh_fails(*argv: str) -> tuple[int, str, str]:
        return (1, "", "boom") if argv[:2] == ("pr", "create") else (0, "", "")
    ctx = saw.build_context(root, "main", agent_client=_StubAgent(), proposals_dir=tmp_path / "p",
                            worktrees_dir=tmp_path / "wt", gh_runner=gh_fails,
                            run_cs_assure=_cs_assure_green(), ci_attested_safe=True)
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx, max_steps=25)
    assert state.current_phase is Phase.ESCALATED
    refs = state.review_state.get("agent_proposals") or []
    assert refs and refs[0]["branch"].startswith("cs-agent/g1-")     # the live branch IS discoverable
    assert refs[0]["pr"] == ""                                       # ...and honestly has no PR
    # the ref must IDENTIFY the sealed proposal, not just note that something happened
    sealed = json.loads(Path(refs[0]["path"]).read_text())
    assert refs[0]["record_digest"] == sealed["record_digest"]
    assert refs[0]["changed_paths"] == sealed["payload"]["changed_paths"]
    assert _g(remote, "branch", "--list", "cs-agent/g1-*").stdout.strip()


def test_the_candidate_branches_from_the_remote_base_not_the_local_one(tmp_path: Path) -> None:
    """The base was resolved from the LOCAL ref, so an operator whose `main` is ahead of `origin/main` -
    an ordinary state - would have their unpushed commits pushed and PR'd as though the agent wrote them,
    and the PR diff would not be the candidate."""
    root, remote = _repo_with_remote(tmp_path)
    (root / "operator-only.txt").write_text("not pushed\n")           # a local commit NOT on the remote
    _g(root, "add", "-A")
    _g(root, "commit", "-q", "-m", "operator local work")
    run_loop(LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL),
             _build(tmp_path, root, _StubAgent(), []), max_steps=25)
    branch = [b.strip("* ") for b in _g(remote, "branch", "--list").stdout.splitlines() if "cs-agent" in b][0]
    files = _g(remote, "ls-tree", "-r", "--name-only", branch).stdout.split()
    assert "operator-only.txt" not in files, "the operator's unpushed commit was published as the agent's"


# ----------------------------------------------------------- exact candidate identity (phase 7.1.3)


def test_the_published_commit_is_provably_the_assured_object(tmp_path: Path) -> None:
    """The identity chain: the tree that was CLASSIFIED is the tree COMMITTED, the commit ASSURED is the
    commit PUSHED, and all of it is recorded. Previously these were four objects related only by
    'nothing bad happened in between'."""
    root, remote = _repo_with_remote(tmp_path)
    state = LoopState(goal="tidy", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), []))
    ca = state.review_state["candidate_assurance"]
    commit, tree = ca["candidate_commit"], ca["candidate_tree_oid"]
    # the remote carries EXACTLY the assured commit...
    assert ca["remote_oid"] == commit
    branch = ca["branch"]
    assert _g(remote, "rev-parse", branch).stdout.strip() == commit
    # ...whose tree is the classified tree, and whose parent is the base we classified against
    assert _g(remote, "rev-parse", f"{commit}^{{tree}}").stdout.strip() == tree
    assert _g(remote, "rev-list", "--parents", "-n", "1", commit).stdout.split()[1:] == [ca["base_oid"]]
    # and the loop's head-bound merge gate is now bound to it (nothing set this seam before)
    assert state.review_state["agent_proposals"][0]["candidate_commit"] == commit


def test_the_candidate_commit_is_content_addressed(tmp_path: Path) -> None:
    # pinned dates: identical content must yield an identical commit oid, so a resumed/re-run goal
    # reproduces the same object instead of a fresh one each time.
    oids = []
    for i in range(2):
        d = tmp_path / f"r{i}"
        d.mkdir()
        root, _rem = _repo_with_remote(d)
        st = LoopState(goal="tidy", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
        run_loop(st, _build(d, root, _StubAgent(), []))
        oids.append(st.review_state["candidate_assurance"]["candidate_commit"])
    assert oids[0] == oids[1], f"commit oid is not deterministic: {oids}"


def test_assurance_that_assessed_a_different_object_is_refused(tmp_path: Path) -> None:
    """The measured live defect this phase closes: the adapter assessed the MUTABLE WORKING TREE
    (`--scope workspace`, the default) while the COMMIT is what gets pushed. On a candidate whose commit
    edits worker bytes and whose worktree is reverted, workspace reported fired=[] and head reported
    ['worker-closure'] - the gate said clean about a commit that fires a human-gated obligation."""
    # a record about the wrong scope is refused...
    def wrong_scope(_r, *a):
        if a and a[0] == "impact":
            return (0, json.dumps({"record_digest": "sha256:i",
                                   "payload": {"scope": "workspace", "fired_obligations": [],
                                               "base_policy_available": True},
                                   "provenance": {"head_oid": "c" * 40}}), "")
        return (0, json.dumps({"payload": {"undeclared_reachable": [], "worker_roots": [],
                                           "added_reachable": [], "distribution_impacting_paths": []}}), "")
    with pytest.raises(saw.WriteAdapterError, match="scope"):
        saw._assure_candidate(tmp_path, "b" * 40, wrong_scope, [], "c" * 40, _PROFILE)

    # ...and so is a record about a DIFFERENT commit
    def wrong_head(_r, *a):
        if a and a[0] == "impact":
            return (0, json.dumps({"record_digest": "sha256:i",
                                   "payload": {"scope": "head", "fired_obligations": [],
                                               "base_policy_available": True},
                                   "provenance": {"head_oid": "d" * 40}}), "")
        return (0, json.dumps({"payload": {"undeclared_reachable": [], "worker_roots": [],
                                           "added_reachable": [], "distribution_impacting_paths": []}}), "")
    with pytest.raises(saw.WriteAdapterError, match="different object"):
        saw._assure_candidate(tmp_path, "b" * 40, wrong_head, [], "c" * 40, _PROFILE)


def test_the_remote_ref_check_requires_an_exact_single_match(tmp_path: Path) -> None:
    """`git ls-remote <url> <pattern>` matches the ref TAIL, so a decoy `refs/heads/decoy/refs/heads/X`
    also matches and a naive `head -1` reads the DECOY's oid."""
    root, remote = _repo_with_remote(tmp_path)
    head = _g(root, "rev-parse", "HEAD").stdout.strip()
    _g(root, "push", "-q", "origin", f"{head}:refs/heads/decoy/refs/heads/cs-agent/x")
    assert saw._remote_ref_oid(root, "cs-agent/x") == ""        # the decoy is NOT our ref
    _g(root, "push", "-q", "origin", f"{head}:refs/heads/cs-agent/x")
    assert saw._remote_ref_oid(root, "cs-agent/x") == head      # the real one is found exactly


def test_commit_identity_verification_rejects_a_wrong_tree_or_parent(tmp_path: Path) -> None:
    """These guards only fire when something is already wrong, so the happy path never exercises them -
    they are asserted directly. A tree-only check is not enough: commit-tree can produce an identical
    tree under a HOSTILE PARENT (carrying someone else's history and an attacker-chosen message)."""
    root, _remote = _repo_with_remote(tmp_path)
    base = _g(root, "rev-parse", "HEAD").stdout.strip()
    tree = _g(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    good = _g(root, "commit-tree", tree, "-p", base, "-m", "ok").stdout.strip()
    saw._verify_commit_identity(root, good, tree, base)                       # the honest case passes

    # same tree, WRONG parent (an orphan) -> refused
    orphan = _g(root, "commit-tree", tree, "-m", "orphan").stdout.strip()
    with pytest.raises(saw.WriteAdapterError, match="parents"):
        saw._verify_commit_identity(root, orphan, tree, base)
    # right parent, WRONG tree -> refused
    with pytest.raises(saw.WriteAdapterError, match="!= classified tree"):
        saw._verify_commit_identity(root, good, "0" * 40, base)


def test_a_remote_that_does_not_carry_the_assured_commit_blocks_the_pr(tmp_path: Path, monkeypatch) -> None:
    """If the ref the remote actually holds is not our commit, no PR may be opened for it - the push
    receipt is `ls-remote`, never git push's own output (which echoes the source spec, not a remote oid)."""
    root, _remote = _repo_with_remote(tmp_path)
    calls: list = []
    monkeypatch.setattr(saw, "_remote_ref_oid", lambda *a, **k: "f" * 40)   # remote holds something else
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), calls), max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]
    # either refusal is correct - the property is that no PR is opened for an object we did not assure
    # (the pre-push check now catches a foreign branch first; the post-push receipt catches the rest)
    reason = state.termination_reason or ""
    assert "not the assured candidate" in reason or "refusing to overwrite" in reason, reason


def test_the_push_names_the_candidate_commit_explicitly(tmp_path: Path) -> None:
    """The refspec is `<oid>:refs/heads/<branch>`, not `HEAD:...` - an explicit oid cannot publish
    anything but the assured commit even if HEAD moves, and the lease is create-only."""
    root, _remote = _repo_with_remote(tmp_path)
    seen: list = []
    real = saw._git

    def spy(cwd, *args, **kw):
        if args and args[0] == "push":
            seen.append(list(args))
        return real(cwd, *args, **kw)
    import unittest.mock as _m
    with _m.patch.object(saw, "_git", spy):
        state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
        run_loop(state, _build(tmp_path, root, _StubAgent(), []))
    commit = state.review_state["candidate_assurance"]["candidate_commit"]
    assert seen, "no push was attempted"
    argv = seen[0]
    assert any(a.startswith(f"{commit}:refs/heads/") for a in argv), argv
    assert any(a.startswith("--force-with-lease=") for a in argv), argv


# ------------------------------------------------------ crash recovery + safe publish (phase 7.1.4)


def test_the_pr_is_opened_as_a_draft(tmp_path: Path) -> None:
    """Machine-authored code must not be one click from merge. Draft is a STRUCTURAL signal that survives
    someone editing the PR body, unlike the text disclosure alone."""
    root, _remote = _repo_with_remote(tmp_path)
    calls: list = []
    run_loop(LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL),
             _build(tmp_path, root, _StubAgent(), calls))
    create = [c for c in calls if c[:2] == ("pr", "create")]
    assert create and "--draft" in create[0], create


def test_re_running_the_same_goal_republishes_idempotently(tmp_path: Path) -> None:
    """7.1.3 made the candidate commit content-addressed, so an identical re-run produces the SAME oid:
    the push is a no-op and the existing PR is reused rather than duplicated."""
    root, remote = _repo_with_remote(tmp_path)
    calls: list = []
    ctx1 = _build(tmp_path, root, _StubAgent(), calls)
    run_loop(LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL), ctx1)
    first = [c for c in calls if c[:2] == ("pr", "create")]
    assert len(first) == 1

    # a second run of the same goal: the branch is already at our commit, and an open PR exists
    calls2: list = []

    def gh_with_existing(*argv: str) -> tuple[int, str, str]:
        calls2.append(tuple(argv))
        if argv[:2] == ("pr", "list"):
            return (0, json.dumps([{"url": "https://example/pull/1"}]), "")
        if argv[:2] == ("pr", "create"):
            return (0, "https://example/pull/2\n", "")
        return (0, "", "")
    ctx2 = saw.build_context(root, "main", agent_client=_StubAgent(), proposals_dir=tmp_path / "prop",
                             worktrees_dir=tmp_path / "wt", gh_runner=gh_with_existing,
                             run_cs_assure=_cs_assure_green(), ci_attested_safe=True)
    state2 = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state2, ctx2)
    assert ("pr", "create") not in [c[:2] for c in calls2], "a duplicate PR was opened on re-run"
    ca = state2.review_state["candidate_assurance"]
    assert "resumed" in ca["reason"] or "reused" in ca["reason"], ca["reason"]
    # exactly one branch, still carrying the assured commit
    assert _g(remote, "rev-parse", ca["branch"]).stdout.strip() == ca["candidate_commit"]


def test_a_foreign_commit_on_our_branch_is_never_overwritten(tmp_path: Path) -> None:
    root, remote = _repo_with_remote(tmp_path)
    # put a DIFFERENT commit on the branch the goal would use
    branch = "cs-agent/" + saw._sanitize_branch_suffix("g1")
    head = _g(root, "rev-parse", "HEAD").stdout.strip()
    _g(root, "push", "-q", "origin", f"{head}:refs/heads/{branch}")
    calls: list = []
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), calls), max_steps=25)
    assert ("pr", "create") not in [c[:2] for c in calls]
    assert "refusing to overwrite" in (state.termination_reason or "")
    assert _g(remote, "rev-parse", branch).stdout.strip() == head   # untouched


def test_the_publish_journal_records_intent_before_each_effect(tmp_path: Path) -> None:
    """A SIGKILL between push and `gh pr create` bypasses both of step()'s persistence points, so the
    remote could hold an agent-authored branch loop state never heard of. The journal is the durable trace."""
    root, _remote = _repo_with_remote(tmp_path)
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, _build(tmp_path, root, _StubAgent(), []))
    ca = state.review_state["candidate_assurance"]
    jp = saw._journal_path(tmp_path / "prop", "g1", ca["candidate_commit"])
    entry = saw._journal_read(jp)
    assert entry["state"] == "PR_OPENED"
    assert entry["candidate_commit"] == ca["candidate_commit"]
    assert entry["branch"] == ca["branch"] and entry["pr_url"]
    # an unreadable/absent journal is "never started", never a crash
    assert saw._journal_read(tmp_path / "nope.json") == {}


def test_the_journal_records_push_intent_even_when_the_push_fails(tmp_path: Path) -> None:
    """The write-ahead property: intent is durable BEFORE the effect. If the process dies (or the push
    fails) the journal still names the branch and commit that may now be live on the remote - which is the
    whole point, since a SIGKILL bypasses both of step()'s persistence points."""
    root, _remote = _repo_with_remote(tmp_path)
    real = saw._git

    def push_fails(cwd, *args, **kw):
        if args and args[0] == "push":
            raise saw.WriteAdapterError("simulated network failure mid-publish")
        return real(cwd, *args, **kw)
    import unittest.mock as _m
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    with _m.patch.object(saw, "_git", push_fails):
        run_loop(state, _build(tmp_path, root, _StubAgent(), []), max_steps=25)
    assert state.current_phase is Phase.ESCALATED
    journals = sorted((tmp_path / "prop" / "journal").glob("*.json"))
    assert journals, "no write-ahead record survived the failed push"
    entry = saw._journal_read(journals[0])
    assert entry["state"] == "PUSH_INTENDED"          # intent was durable BEFORE the effect
    assert entry["branch"].startswith("cs-agent/g1-") and entry["candidate_commit"]
