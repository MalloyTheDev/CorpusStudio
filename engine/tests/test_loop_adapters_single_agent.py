"""Phase 7.0 - the read/propose-only single-agent adapter (scripts/loop_adapters/single_agent.py).

Pins the SAFETY properties: it declares no capability (read-only), the injected agent's output is validated
fail-closed and sealed, the sealed proposal is written OUTSIDE the working tree, the loop ESCALATES (never
finalizes a proposal), and - the whole point - it makes ZERO writes to the repo. A stub AgentClient keeps
the test deterministic (no real ``claude``).
"""

from __future__ import annotations

import hashlib
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
from loop.controller import LoopState, Phase  # noqa: E402
from loop.orchestrate import run_loop  # noqa: E402

_DIFF = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n"


class _StubAgent:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response if response is not None else {"unified_diff": _DIFF, "rationale": "tweak"}
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
    assert record["payload"]["unified_diff"] == _DIFF


def test_a_malformed_agent_response_fails_closed(tmp_path: Path) -> None:
    # untrusted output: a non-string unified_diff must RAISE (AgentError) -> the loop escalates, never
    # advances on garbage and never applies anything.
    state = _run(tmp_path, _StubAgent({"unified_diff": 123, "rationale": "x"}))
    assert state.current_phase is Phase.ESCALATED
    assert "AgentError" in (state.termination_reason or "")


# --------------------------------------------------------------------------- helpers (pure)


def test_validate_proposal_is_fail_closed() -> None:
    with pytest.raises(sa.AgentError):
        sa._validate_proposal(["not", "a", "dict"])
    with pytest.raises(sa.AgentError):
        sa._validate_proposal({"rationale": "no diff"})
    assert sa._validate_proposal({"unified_diff": "d", "rationale": "r"}) == ("d", "r")


def test_changed_paths_of_ignores_dev_null() -> None:
    diff = "--- a/kept.py\n+++ b/kept.py\n--- /dev/null\n+++ b/added.py\n--- a/removed.py\n+++ /dev/null\n"
    assert sa._changed_paths_of(diff) == ["added.py", "kept.py", "removed.py"]


def test_claude_subprocess_client_fails_closed_on_a_missing_binary() -> None:
    # a non-existent transport binary raises AgentError (fail-closed), never a silent empty proposal.
    client = sa.ClaudeSubprocessClient(argv=("definitely-not-a-real-binary-xyz",), timeout=5)
    with pytest.raises(sa.AgentError):
        client.propose({"goal": "x"})


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


def test_claude_subprocess_client_parses_and_validates_a_good_response(monkeypatch: pytest.MonkeyPatch) -> None:
    good = json.dumps({"unified_diff": "d", "rationale": "r", "extra": "ignored"})
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, good))
    assert sa.ClaudeSubprocessClient().propose({"goal": "g"}) == {"unified_diff": "d", "rationale": "r"}


def test_claude_subprocess_client_fails_closed_on_a_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(2, "", "boom"))
    with pytest.raises(sa.AgentError, match="exited 2"):
        sa.ClaudeSubprocessClient().propose({"goal": "g"})


def test_claude_subprocess_client_fails_closed_on_unparseable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, "not json at all"))
    with pytest.raises(sa.AgentError, match="no usable JSON"):
        sa.ClaudeSubprocessClient().propose({"goal": "g"})


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
        return _FakeProc(0, json.dumps({"unified_diff": "d", "rationale": "r"}))

    monkeypatch.setattr(sa.subprocess, "run", fake_run)
    sa.ClaudeSubprocessClient().propose({"goal": "g", "_cwd": str(tmp_path)})
    assert captured["cwd"] == str(tmp_path)                    # confined to the injected worktree
    assert "GITHUB_TOKEN" not in captured["env"]               # secret-free env


def test_the_subprocess_client_fails_closed_on_oversized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    huge = json.dumps({"unified_diff": "x" * 64, "rationale": "r"})
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: _FakeProc(0, huge))
    client = sa.ClaudeSubprocessClient(max_output_bytes=8)  # cap below the output size
    with pytest.raises(sa.AgentError, match="oversized"):
        client.propose({"goal": "g"})
