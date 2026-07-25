"""Phase 7.0 - the read/propose-only single-agent adapter (scripts/loop_adapters/single_agent.py).

Pins the SAFETY properties: it declares no capability (read-only), the injected agent's output is validated
fail-closed and sealed, the sealed proposal is written OUTSIDE the working tree, the loop ESCALATES (never
finalizes a proposal), and - the whole point - it makes ZERO writes to the repo. A stub AgentClient keeps
the test deterministic (no real ``claude``).
"""

from __future__ import annotations

import hashlib
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
from loop.controller import LoopState, Phase  # noqa: E402
from loop.orchestrate import run_loop  # noqa: E402

_FILES = {"README.md": "new\n"}          # the agent returns WHOLE FILES; git computes the diff
_DIFF = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"


class _StubAgent:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response if response is not None else {"files": _FILES, "rationale": "tweak"}
        self.calls = 0
        self.seen_cwd: str | None = None
        self.cwd_was_dir: bool | None = None

    def propose(self, request: dict) -> dict:
        self.calls += 1
        self.last_request = request
        # observe the confinement AT CALL TIME (the disposable worktree is removed once we return)
        self.seen_cwd = request.get("_cwd")
        self.cwd_was_dir = bool(self.seen_cwd) and Path(self.seen_cwd).is_dir()
        return self.response


def _cs_assure_green():
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


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "a@b.c")
    _git(root, "config", "user.name", "t")
    (root / "README.md").write_text("old\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _run(tmp_path: Path, agent: _StubAgent) -> LoopState:
    root = _repo(tmp_path)
    ctx = sa.build_context(root, "main", agent_client=agent, proposals_dir=tmp_path / "proposals",
                           run_cs_assure=_cs_assure_green())
    state = LoopState(goal="tidy the README", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx)
    return state


# --------------------------------------------------------------------------- the safety properties


def test_the_adapter_declares_no_capability() -> None:
    ctx = sa.build_context(REPO_ROOT, "main", agent_client=_StubAgent())
    assert ctx.capabilities == frozenset()  # read-only: the capability gate runs it with no opt-in


def test_a_run_proposes_a_sealed_diff_escalates_and_writes_nothing(tmp_path: Path) -> None:
    agent = _StubAgent()
    state = _run(tmp_path, agent)
    # the agent was asked to propose, and the loop ESCALATED (a proposal is not a finished goal)
    assert agent.calls == 1 and state.current_phase is Phase.ESCALATED
    # a sealed proposal is referenced on the state + written OUTSIDE the working tree
    refs = state.review_state["agent_proposals"]
    assert len(refs) == 1 and refs[0]["changed_paths"] == ["README.md"]
    proposal_file = Path(refs[0]["path"])
    assert proposal_file.is_file() and (tmp_path / "proposals") in proposal_file.parents
    # ZERO writes to the repo: the working tree is clean and there is still exactly one commit
    root = tmp_path / "repo"
    assert _git(root, "status", "--porcelain").stdout == ""
    assert _git(root, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert (root / "README.md").read_text() == "old\n"  # the proposed diff was NOT applied


def test_the_propose_agent_is_confined_to_a_disposable_worktree(tmp_path: Path) -> None:
    # Even PROPOSE-only, the untrusted agent runs with cwd inside a throwaway worktree (never the dev tree),
    # so a mis-behaving agent cannot edit the working tree while "just proposing".
    agent = _StubAgent()
    root = _repo(tmp_path)
    ctx = sa.build_context(root, "main", agent_client=agent, proposals_dir=tmp_path / "proposals",
                           worktrees_dir=tmp_path / "wt", run_cs_assure=_cs_assure_green())
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx)
    assert agent.cwd_was_dir is True and agent.seen_cwd is not None
    seen = Path(agent.seen_cwd).resolve()
    assert (tmp_path / "wt").resolve() in seen.parents and seen != root.resolve()
    assert not seen.exists()  # disposed after the propose
    assert _git(root, "status", "--porcelain").stdout == ""  # dev tree still pristine


def test_a_corrupt_agent_proposals_field_is_normalized_not_silently_skipped(tmp_path: Path) -> None:
    # If review_state["agent_proposals"] is present but not a list, the ref must still be recorded (the
    # invariant "a written proposal is referenced" holds) - normalize, never silently skip.
    root = _repo(tmp_path)
    ctx = sa.build_context(root, "main", agent_client=_StubAgent(), proposals_dir=tmp_path / "proposals",
                           run_cs_assure=_cs_assure_green())
    state = LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL)
    state.review_state["agent_proposals"] = "corrupt-not-a-list"
    run_loop(state, ctx)
    refs = state.review_state["agent_proposals"]
    assert isinstance(refs, list) and len(refs) == 1 and refs[0]["changed_paths"] == ["README.md"]


def test_the_sealed_proposal_record_verifies(tmp_path: Path) -> None:
    state = _run(tmp_path, _StubAgent())
    record = json.loads(Path(state.review_state["agent_proposals"][0]["path"]).read_text())
    assert record["record_type"] == "agent_proposal"
    envelope = {k: v for k, v in record.items() if k != "record_digest"}
    redigest = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    assert record["record_digest"] == f"sha256:{redigest}"
    assert "README.md" in record["payload"]["unified_diff"]   # git-computed, not model-authored


def test_a_malformed_agent_response_fails_closed(tmp_path: Path) -> None:
    # untrusted output: a non-string unified_diff must RAISE (AgentError) -> the loop escalates, never
    # advances on garbage and never applies anything.
    state = _run(tmp_path, _StubAgent({"files": 123, "rationale": "x"}))
    assert state.current_phase is Phase.ESCALATED
    assert "AgentError" in (state.termination_reason or "")


# --------------------------------------------------------------------------- helpers (pure)


def test_validate_proposal_is_fail_closed() -> None:
    with pytest.raises(sa.AgentError):
        sa._validate_proposal(["not", "a", "dict"])
    with pytest.raises(sa.AgentError):
        sa._validate_proposal({"rationale": "no files"})
    with pytest.raises(sa.AgentError):                       # traversal is refused BEFORE any write
        sa._validate_proposal({"files": {"../escape.py": "x"}})
    with pytest.raises(sa.AgentError):
        sa._validate_proposal({"files": {".git/config": "x"}})
    assert sa._validate_proposal({"files": {"a.py": "c"}, "rationale": "r"}) == ({"a.py": "c"}, "r")


def test_changed_paths_of_ignores_dev_null() -> None:
    diff = "--- a/kept.py\n+++ b/kept.py\n--- /dev/null\n+++ b/added.py\n--- a/removed.py\n+++ /dev/null\n"
    assert sa._changed_paths_of(diff) == ["added.py", "kept.py", "removed.py"]


def test_claude_subprocess_client_fails_closed_on_a_missing_binary(tmp_path: Path) -> None:
    # a non-existent transport binary raises AgentError (fail-closed), never a silent empty proposal.
    client = sa.ClaudeSubprocessClient(argv=("definitely-not-a-real-binary-xyz",), timeout=5)
    with pytest.raises(sa.AgentError):
        client.propose({"goal": "x", "_cwd": str(tmp_path)})


def test_claude_subprocess_client_refuses_to_run_unconfined() -> None:
    # confinement is mandatory at the transport: a missing / non-directory _cwd fails closed rather than
    # running the agent in the process's own cwd (the developer's tree).
    client = sa.ClaudeSubprocessClient()
    with pytest.raises(sa.AgentError, match="unconfined"):
        client.propose({"goal": "x"})                                  # no _cwd at all
    with pytest.raises(sa.AgentError, match="unconfined"):
        client.propose({"goal": "x", "_cwd": "/no/such/worktree/here"})  # _cwd is not a directory


def test_read_only_gh_is_wired_and_refuses_a_merge(tmp_path: Path) -> None:
    # with a pr_ref the adapter wires the read-only gh + dangerous=True; the gh runner refuses `pr merge`.
    ctx = sa.build_context(_repo(tmp_path), "main", agent_client=_StubAgent(), pr_ref="1",
                           proposals_dir=tmp_path / "p")
    assert ctx.dangerous is True and ctx.pr_ref == "1"
    code, _out, err = ctx.gh_runner("pr", "merge", "1")
    assert code != 0 and "refused" in err


class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_claude_subprocess_client_parses_and_validates_a_good_response(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the REAL envelope: the agent's payload is nested in `result` as a string (see _REAL_ENVELOPE)
    good = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                       "result": json.dumps({"files": {"a.py": "c"}, "rationale": "r", "extra": "ignored"})})
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, good))
    got = sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})
    assert got == {"files": {"a.py": "c"}, "rationale": "r"}


def test_claude_subprocess_client_fails_closed_on_a_nonzero_exit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(2, "", "boom"))
    with pytest.raises(sa.AgentError, match="exited 2"):
        sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})


def test_claude_subprocess_client_fails_closed_on_unparseable_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, "not json at all"))
    with pytest.raises(sa.AgentError, match="no usable JSON"):
        sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})


def test_default_proposals_dir_falls_back_when_not_a_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(128, "", "not a git repo"))
    assert sa._default_proposals_dir(tmp_path) == tmp_path / ".corpusstudio-loop-proposals"


# --------------------------------------------------------------------------- confinement (7.1.1)


def test_sanitized_env_strips_credential_shaped_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # secrets by SUBSTRING and by known auth PREFIX are stripped; benign vars (PATH/HOME/locale) survive.
    for k in ("GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "MY_API_KEY", "DB_PASSWORD",
              "SESSION_ID", "npm_config_registry_AUTH", "ANTHROPIC_API_KEY", "SSH_AUTH_SOCK"):
        monkeypatch.setenv(k, "secret")
    for k in ("PATH", "HOME", "LANG", "CORPUS_STUDIO_MODE"):
        monkeypatch.setenv(k, "ok")
    clean = sa._sanitized_env()
    assert not any(bad in clean for bad in (
        "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "MY_API_KEY", "DB_PASSWORD", "SESSION_ID",
        "ANTHROPIC_API_KEY", "SSH_AUTH_SOCK", "npm_config_registry_AUTH"))
    assert clean["PATH"] == "ok" and clean["HOME"] == "ok" and clean["CORPUS_STUDIO_MODE"] == "ok"


def test_the_default_tool_policy_is_read_only() -> None:
    # the version-sensitive propose policy: read/grep/glob allowed; edit/write/bash/nested-agents/net denied.
    argv = sa._READONLY_TOOL_ARGV
    assert argv[0] == "claude" and "--output-format" in argv and "json" in argv
    allowed = argv[argv.index("--allowedTools") + 1]
    denied = argv[argv.index("--disallowedTools") + 1]
    assert set(allowed.split(",")) == {"Read", "Grep", "Glob"}
    for tool in ("Edit", "Write", "Bash", "Task", "WebFetch", "WebSearch", "NotebookEdit"):
        assert tool in denied.split(",")


def test_the_subprocess_client_runs_with_a_sanitized_env_and_the_confined_cwd(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "should-not-leak")
    captured: dict[str, object] = {}

    def fake_run(*a: object, **k: object) -> _FakeProc:
        captured.update(k)
        return _FakeProc(0, json.dumps({"type": "result", "subtype": "success", "is_error": False,
                                        "result": json.dumps({"files": {"a.py": "c"}, "rationale": "r"})}))

    monkeypatch.setattr(sa.subprocess, "run", fake_run)
    sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})
    assert captured["cwd"] == str(tmp_path)                    # confined to the injected worktree
    assert "GITHUB_TOKEN" not in captured["env"]               # secret-free env


def test_the_subprocess_client_fails_closed_on_oversized_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    huge = json.dumps({"unified_diff": "x" * 64, "rationale": "r"})
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, huge))
    client = sa.ClaudeSubprocessClient(max_output_bytes=8)  # cap below the output size
    with pytest.raises(sa.AgentError, match="oversized"):
        client.propose({"goal": "g", "_cwd": str(tmp_path)})


# --------------------------------------------------------- confined HOME (independent-review residual)


def test_the_confined_home_hides_credential_paths_but_keeps_the_transport_working(tmp_path: Path) -> None:
    """The agent's sanitized env preserved HOME, so `~/.config/gh/hosts.yml` (a token with WRITE access to
    this repo), `~/.ssh`, and ~/.claude's sessions/history/other-projects transcripts were all reachable by
    plain `~` expansion. The confined HOME repoints HOME + every XDG base at a throwaway dir containing
    ONLY what the transport needs to authenticate itself."""
    with sa._confined_home(tmp_path) as home:
        env = sa._sanitized_env(home)
        assert env["HOME"] == str(home)
        for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
            assert env[var].startswith(str(home)), var
        # the credential-shaped paths do not exist under the confined home...
        for rel in (".config/gh/hosts.yml", ".ssh", ".aws/credentials", ".netrc",
                    ".claude/history.jsonl", ".claude/sessions", ".claude/projects"):
            assert not (home / rel).exists(), rel
        # ...while the deliberate passthroughs are linked when they exist on the real home
        for rel in sa._AGENT_HOME_PASSTHROUGH:
            if (Path.home() / rel).exists():
                assert (home / rel).exists(), rel
    assert not home.exists(), "the confined home must be disposed"


def test_sanitized_env_without_a_home_is_unchanged(tmp_path: Path) -> None:
    # the confinement is opt-in per call: no home -> the caller's HOME survives (used by non-agent callers)
    assert "XDG_STATE_HOME" not in {k: v for k, v in sa._sanitized_env().items() if k == "XDG_STATE_HOME"} \
        or sa._sanitized_env().get("HOME") == os.environ.get("HOME")


def test_the_propose_executor_hands_the_agent_a_confined_home(tmp_path: Path) -> None:
    seen: dict = {}

    class _Probe:
        def propose(self, request: dict) -> dict:
            seen.update(request)
            seen["home_existed"] = bool(request.get("_home")) and Path(request["_home"]).is_dir()
            return {"unified_diff": _DIFF, "rationale": "r"}

    root = _repo(tmp_path)
    ctx = sa.build_context(root, "main", agent_client=_Probe(), proposals_dir=tmp_path / "p",
                           worktrees_dir=tmp_path / "wt", run_cs_assure=_cs_assure_green())
    run_loop(LoopState(goal="g", goal_id="g1", current_phase=Phase.RECEIVE_GOAL), ctx)
    assert seen.get("home_existed") is True, "the agent was not given a confined HOME"
    assert Path(seen["_home"]).resolve() != Path.home().resolve()
    assert not Path(seen["_home"]).exists(), "the confined home must be disposed after the propose"


# ------------------------------------------------------------------ agent sandboxing (phase 7.1.5)


class _NoIsolationSandbox:
    """Runs the command with NO confinement - what a broken/misconfigured sandbox looks like."""

    def wrap(self, argv, *, cwd, home):  # noqa: ANN001,ANN201
        return argv


class _DenyingSandbox:
    """Confines by refusing to read anything outside the bound surfaces (what a working one does)."""

    def wrap(self, argv, *, cwd, home):  # noqa: ANN001,ANN201
        return ("/bin/sh", "-c", "echo __DENIED__")


def test_verify_sandbox_catches_one_that_runs_but_does_not_isolate(tmp_path: Path) -> None:
    """Presence is not proof. A sandbox that executes but does not confine is WORSE than none: it turns a
    known gap into a false assurance, and an operator would grant unattended use on the strength of it.
    The probe plants a canary secret outside the bound surfaces and asserts it cannot be read."""
    with pytest.raises(sa.SandboxUnavailable, match="does not isolate"):
        sa.verify_sandbox(_NoIsolationSandbox(), tmp_path)


def test_verify_sandbox_accepts_one_that_denies(tmp_path: Path) -> None:
    sa.verify_sandbox(_DenyingSandbox(), tmp_path)          # no raise


def test_verify_sandbox_fails_closed_when_the_sandbox_cannot_run(tmp_path: Path) -> None:
    class _Missing:
        def wrap(self, argv, *, cwd, home):  # noqa: ANN001,ANN201
            return ("definitely-not-a-real-sandbox-xyz", *argv)
    with pytest.raises(sa.SandboxUnavailable):
        sa.verify_sandbox(_Missing(), tmp_path)


def test_the_transport_wraps_the_agent_when_a_sandbox_is_supplied(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}

    class _Recording:
        def wrap(self, argv, *, cwd, home):  # noqa: ANN001,ANN201
            seen["cwd"], seen["home"] = str(cwd), str(home)
            return ("SANDBOXED", *argv)

    def fake_run(a, **k):  # noqa: ANN001,ANN202
        seen["argv"] = list(a)
        return _FakeProc(0, json.dumps({"type": "result", "subtype": "success", "is_error": False,
                                        "result": json.dumps({"files": {"a.py": "c"}, "rationale": "r"})}))
    monkeypatch.setattr(sa.subprocess, "run", fake_run)
    client = sa.ClaudeSubprocessClient(sandbox=_Recording())
    client.propose({"goal": "g", "_cwd": str(tmp_path), "_home": str(tmp_path / "h")})
    assert seen["argv"][0] == "SANDBOXED", seen["argv"]
    assert seen["cwd"] == str(tmp_path) and seen["home"] == str(tmp_path / "h")
    # ...and with no sandbox the argv is unwrapped (the seam is opt-in, never silently assumed)
    sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})
    assert seen["argv"][0] != "SANDBOXED"


def test_bubblewrap_argv_binds_only_the_worktree_and_home(tmp_path: Path) -> None:
    argv = sa.BubblewrapSandbox().wrap(("claude", "-p"), cwd=tmp_path / "wt", home=tmp_path / "home")
    assert argv[0] == "bwrap" and "--unshare-all" in argv       # no network, no shared namespaces
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    assert binds == [str(tmp_path / "wt"), str(tmp_path / "home")]   # the ONLY WRITABLE surfaces
    assert argv[-1] == "-p"
    # the agent binary is bound READ-ONLY and invoked by its resolved path: it is very often not under
    # /usr (measured here: a self-contained ELF under an XDG data dir via a ~/.local/bin symlink), and
    # binding only the system runtime made the sandbox fail with "execvp claude: No such file".
    import shutil as _sh
    if _sh.which("claude"):
        resolved = str(Path(_sh.which("claude")).resolve())
        ro = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
        assert resolved in ro and argv[-2] == resolved
    else:                                    # no agent installed: nothing extra to bind, argv unchanged
        assert argv[-2] == "claude"


def test_verify_sandbox_fails_closed_when_the_sandbox_exits_nonzero(tmp_path: Path) -> None:
    # distinct from "cannot run": the sandbox EXISTS and executes but errors out (exactly what bwrap does
    # on a host with apparmor_restrict_unprivileged_userns=1). It must not be read as confinement.
    class _Erroring:
        def wrap(self, argv, *, cwd, home):  # noqa: ANN001,ANN201
            return ("/bin/sh", "-c", "echo 'bwrap: setting up uid map: Permission denied' >&2; exit 1")
    with pytest.raises(sa.SandboxUnavailable, match="exited 1"):
        sa.verify_sandbox(_Erroring(), tmp_path)


# ---------------------------------------------- the REAL transport contract (recorded, not invented)

# Captured from an actual `claude -p --output-format json` run. The adapter used to do
# json.loads(stdout) and look for `unified_diff` at the TOP level, so against a real agent it failed
# 100% of the time - and no test caught it because every test injected a stub.
_REAL_ENVELOPE = {
    "type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
    "duration_ms": 1459, "num_turns": 1, "result": "pong", "stop_reason": "end_turn",
    "session_id": "6c80b5c5-31c6-47aa-8b92-4c01eff15ece", "total_cost_usd": 0.1923,
}


def _envelope(result_text: str, **over) -> str:
    return json.dumps({**_REAL_ENVELOPE, "result": result_text, **over})


def test_the_transport_unwraps_the_real_claude_envelope(tmp_path: Path, monkeypatch) -> None:
    payload = json.dumps({"files": {"x.py": "contents\n"}, "rationale": "r"})
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, _envelope(payload)))
    got = sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})
    assert got == {"files": {"x.py": "contents\n"}, "rationale": "r"}


def test_the_transport_accepts_a_single_markdown_fence(tmp_path: Path, monkeypatch) -> None:
    # models very often fence JSON; a single fence is unambiguous, anything else is refused
    fenced = '```json\n{"files": {"a.py": "c"}, "rationale": "r"}\n```'
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, _envelope(fenced)))
    assert sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})["rationale"] == "r"


@pytest.mark.parametrize("stdout,why", [
    (_envelope("pong"), "prose instead of JSON"),
    (_envelope('{"rationale": "r"}'), "no files"),
    (json.dumps({**_REAL_ENVELOPE, "is_error": True}), "the CLI reported an error"),
    (json.dumps({**_REAL_ENVELOPE, "subtype": "error_max_turns"}), "a non-success subtype"),
    (json.dumps({**_REAL_ENVELOPE, "result": None}), "result is not a string"),
    ("not json at all", "unparseable envelope"),
    ('["not", "an", "object"]', "envelope is not an object"),
])
def test_the_transport_fails_closed_on_a_bad_envelope(stdout: str, why: str, tmp_path: Path,
                                                      monkeypatch) -> None:
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, stdout))
    with pytest.raises(sa.AgentError):
        sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})


def test_the_agent_is_actually_told_what_to_return_and_the_bounds() -> None:
    """The transport used to pipe the raw request dict with NO instruction, so the model was never told
    what shape to reply in - the other half of why the propose path could not work for real."""
    prompt = sa.render_prompt({"goal": "tidy a docstring",
                               "_limits": {"writable_globs": ["src/**/*.py"], "max_changed_paths": 2,
                                           "max_changed_lines": 60, "max_changed_bytes": 8192}})
    assert "tidy a docstring" in prompt
    assert "files" in prompt and "rationale" in prompt
    assert "COMPLETE new contents" in prompt   # whole files, never a diff
    assert "src/**/*.py" in prompt                      # the surface it may touch
    assert "MODIFY EXISTING FILES ONLY" in prompt       # the rule that refuses the most candidates
    # a request with no limits still renders, and says plainly that nothing is writable
    assert "nothing is writable" in sa.render_prompt({"goal": "g"})


def test_the_confined_home_carries_the_credential_the_transport_needs(tmp_path: Path) -> None:
    """MEASURED: without `.claude/.credentials.json` the CLI returns "Not logged in" and the propose path
    cannot run at all. That file IS the agent's OAuth token and it is here deliberately - the honest
    boundary is that the untrusted agent can read ITS OWN credential, and only that one."""
    assert ".claude/.credentials.json" in sa._AGENT_HOME_PASSTHROUGH
    with sa._confined_home(tmp_path) as home:
        for rel in sa._AGENT_HOME_PASSTHROUGH:
            src = Path.home() / rel
            if not src.is_file():
                continue
            dst = home / rel
            assert dst.is_file() and not dst.is_symlink(), (
                f"{rel} must be COPIED, not symlinked: a symlink points outside the sandbox mount "
                "namespace and dangles (measured as 'Not logged in')")
            assert dst.stat().st_mode & 0o777 == 0o600      # a credential copy stays restrictive
        # and the operator's OTHER credentials are still absent
        for rel in (".config/gh/hosts.yml", ".ssh", ".aws/credentials", ".claude/history.jsonl"):
            assert not (home / rel).exists(), rel


def test_the_sandbox_shares_the_network_but_not_the_filesystem() -> None:
    """HONEST TRADE-OFF, pinned so it cannot be silently changed: the agent is an API client and CANNOT
    work without the network (measured: --unshare-all made it exit 1 immediately). Network egress is
    therefore NOT a control here - the isolation this sandbox provides is of the FILESYSTEM."""
    argv = sa.BubblewrapSandbox().wrap(("claude",), cwd=Path("/tmp/wt"), home=Path("/tmp/home"))
    assert "--unshare-all" in argv and "--share-net" in argv
    # the operator's home is never bound
    assert not any(str(Path.home()) in a for a in argv if a not in ("--ro-bind", "--bind"))
