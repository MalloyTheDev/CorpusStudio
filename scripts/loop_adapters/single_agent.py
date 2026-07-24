"""Phase 7.0 runtime adapter: a REAL single agent, PROPOSE-ONLY (zero writes).

This is the second concrete adapter and the first that wires a real Claude-Code agent into the loop - but
strictly read/propose-only. It drives a goal through the whole state machine against live repo state; at
the EXECUTE phase it asks the agent to PROPOSE a unified diff (it never applies it), seals that proposal as
a tamper-evident ``agent_proposal`` record, and ENDS at ESCALATED so a human decides whether to apply it.
It makes NO writes: no source edits, no commits, no push, no PR, no merge, and it declares
``capabilities=frozenset()`` (read-only), so the capability gate lets it run with no ``--allow-capabilities``.

Design (docs/PRODUCTION_SINGLE_AGENT_RUNTIME.md, phase 7.0):
  * The agent is invoked through an INJECTED :class:`AgentClient` (the loop-level seam); the real transport
    is an out-of-process, fixed-argv ``claude`` subprocess with a framed JSON contract over stdio
    (:class:`ClaudeSubprocessClient`). Tests inject a deterministic stub. No SDK (that would break the
    adapter's stdlib-only rule and run the untrusted agent in-process); no shell.
  * The agent's output is UNTRUSTED: it is validated fail-closed into the sealed record; a bad transport /
    unparseable / wrong-shaped response raises, which escalates the loop (never a silent advance).
  * The agent is CONFINED even while it only proposes (phase 7.1.1): the executor runs it with cwd inside a
    disposable, detached worktree at ``base`` (never the developer's tree) and a sanitized (secret-free)
    environment, and the propose diff is stored as a sealed record OUTSIDE the working tree. Whatever the
    agent writes into the throwaway worktree is discarded; only the unified diff it RETURNS is used.

stdlib-only; every git/gh read and the agent transport fail closed. Read-only ``gh`` + the whole-tree diff
building block are reused from the dry-run adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # noqa: S404 - fixed-argv git / claude only; never a shell string.
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from loop.completeness import Criterion, CriterionKind
from loop.controller import LoopState, Observation, Phase
from loop.driver import Directive
from loop.orchestrate import LoopContext
from loop_adapters.dry_run import read_only_gh  # the same default-deny read-only gh the dry run uses

_AGENT_TIMEOUT_S = 300  # a bounded, killable agent call - never an unbounded hang
_GIT_TIMEOUT_S = 60
_MAX_AGENT_OUTPUT_BYTES = 8 * 1024 * 1024  # 8 MiB cap on agent output - a proposal diff is far smaller

# Environment variables that MUST NOT reach the untrusted agent subprocess: anything credential-shaped, plus
# known VCS / cloud / registry auth. Stripped from the inherited env (an explicit allowlist is stronger
# still; this denylist is the fail-safer floor, and PATH / HOME / locale survive).
_SECRET_ENV_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE_KEY", "APIKEY",
                          "API_KEY", "ACCESS_KEY", "SECRET_KEY", "AUTH", "COOKIE", "SESSION")
_SECRET_ENV_PREFIXES = ("GITHUB_", "GH_", "AWS_", "GOOGLE_", "GCP_", "AZURE_", "OPENAI_", "ANTHROPIC_",
                        "HF_", "HUGGINGFACE_", "NPM_", "PYPI_", "TWINE_", "DOCKER_", "SSH_", "GPG_")


def _sanitized_env() -> dict[str, str]:
    """A copy of the environment with credential-shaped variables STRIPPED, so the untrusted agent
    subprocess never inherits GitHub / cloud / release / registry secrets."""
    clean: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(s in upper for s in _SECRET_ENV_SUBSTRINGS) or any(upper.startswith(p) for p in _SECRET_ENV_PREFIXES):
            continue
        clean[key] = value
    return clean


def _git(cwd: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git -C <cwd> <args>`` (fixed argv, no shell, bounded). Raises :class:`AgentError` on a non-zero
    exit / un-runnable git - shared by the read (propose) and write adapters."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(cwd), *args], input=stdin, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AgentError(f"git {' '.join(args[:2])} could not run: {exc}") from exc
    if proc.returncode != 0:
        raise AgentError(f"git {' '.join(args[:2])} failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    return proc


@contextmanager
def _detached_worktree(repo_root: Path, base_oid: str, worktrees_dir: Path) -> Iterator[Path]:
    """An ISOLATED, disposable, DETACHED ``git worktree`` at ``base_oid`` under ``worktrees_dir`` (outside
    the main working tree) - the confined checkout the agent runs in for a read/propose. Removed on exit
    (best-effort); anything the agent writes here is thrown away."""
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt = worktrees_dir / f"propose-{base_oid[:12]}-{os.getpid()}"
    _git(repo_root, "worktree", "add", "--detach", str(wt), base_oid)
    try:
        yield wt
    finally:
        try:
            _git(repo_root, "worktree", "remove", "--force", str(wt))
        except AgentError:
            pass  # best-effort disposal; a leftover worktree is GC-able, never data loss


def default_worktrees_dir(repo_root: Path) -> Path:
    """``<git-dir>/corpusstudio-loop/worktrees`` (OUTSIDE the working tree), or a fallback when not in a repo."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(repo_root), "rev-parse", "--git-path", "corpusstudio-loop/worktrees"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return repo_root / ".corpusstudio-loop-worktrees"
    if proc.returncode != 0:
        return repo_root / ".corpusstudio-loop-worktrees"
    rel = proc.stdout.strip()
    return Path(rel) if Path(rel).is_absolute() else repo_root / rel


class AgentClient(Protocol):
    """The one seam through which the loop reaches the (untrusted) agent. ``propose`` returns a proposed
    edit for the current goal; it RAISES on any transport / output failure (the caller fails closed)."""

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:
        """Given ``{goal, goal_id, base_oid, directive, repo_root}`` return ``{"unified_diff": str,
        "rationale": str}``. Must raise (not return a partial/garbage dict) on failure."""
        ...


class AgentError(RuntimeError):
    """The agent transport failed or returned output the adapter refuses to trust (fail-closed)."""


def _validate_proposal(raw: Any) -> tuple[str, str]:
    """Fail-closed validation of the agent's response into ``(unified_diff, rationale)``. Untrusted input:
    a non-dict, a missing/non-string ``unified_diff``, or a non-string ``rationale`` all raise."""
    if not isinstance(raw, dict):
        raise AgentError(f"agent response is not an object (got {type(raw).__name__})")
    diff = raw.get("unified_diff")
    if not isinstance(diff, str):
        raise AgentError("agent response has no string 'unified_diff'")
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str):
        raise AgentError("agent response 'rationale' is not a string")
    return diff, rationale


# The read-only tool policy for the PROPOSE phase: the agent may inspect the checkout but not edit it, run
# shell, spawn nested agents, or mutate git/network. Passed to ``claude`` as a DEFAULT; the flag names are
# version-sensitive, so ``argv`` is operator-tunable and MUST be verified against the installed CLI. NOTE:
# prompt/tool restrictions are DEFENCE-IN-DEPTH, not process isolation - the load-bearing confinement is
# that the agent runs with cwd inside a DISPOSABLE worktree with a sanitized (secret-free) environment.
_READONLY_TOOL_ARGV: tuple[str, ...] = (
    "claude", "-p", "--output-format", "json",
    "--allowedTools", "Read,Grep,Glob",
    "--disallowedTools", "Edit,Write,Bash,Task,WebFetch,WebSearch,NotebookEdit",
)


class ClaudeSubprocessClient:
    """The real transport: run ``claude`` OUT OF PROCESS (fixed argv, no shell), CONFINED to a disposable
    worktree and a sanitized environment, feed the request as JSON on stdin, and read one JSON object with
    ``unified_diff`` / ``rationale`` from stdout. Confinement:
      * ``cwd`` = ``request['_cwd']`` (the isolated worktree the executor created) - the agent operates on a
        throwaway checkout, never the developer's tree;
      * ``env`` = :func:`_sanitized_env` - no GitHub / cloud / release / registry secrets are inherited;
      * a read-only tool policy (``argv`` default) denies edit/write/bash/nested-agents/net (defence in depth);
      * the call is bounded by a timeout AND ``max_output_bytes`` (killable) - an oversized/hung agent fails
        closed, never hangs the loop or exhausts memory.
    A non-zero exit / un-runnable binary / unparseable / wrong-shaped / oversized response raises
    :class:`AgentError`. Live behaviour is env-dependent (the CLI must exist + honour the argv)."""

    def __init__(self, *, argv: tuple[str, ...] = _READONLY_TOOL_ARGV, timeout: float = _AGENT_TIMEOUT_S,
                 max_output_bytes: int = _MAX_AGENT_OUTPUT_BYTES) -> None:
        self.argv = argv
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:
        cwd = request.get("_cwd")  # the isolated worktree; the agent runs THERE, not the developer's tree
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, bounded timeout, confined cwd+env.
                list(self.argv), input=json.dumps(request), text=True, capture_output=True,
                timeout=self.timeout, cwd=cwd, env=_sanitized_env())
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentError(f"claude could not run / timed out: {type(exc).__name__}: {exc}") from exc
        if len(proc.stdout.encode("utf-8", "replace")) > self.max_output_bytes:
            raise AgentError(f"claude output exceeded {self.max_output_bytes} bytes; refusing oversized output")
        if proc.returncode != 0:
            raise AgentError(f"claude exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        try:
            data = json.loads(proc.stdout)
        except (ValueError, RecursionError) as exc:
            raise AgentError(f"claude produced no usable JSON: {exc}") from exc
        diff, rationale = _validate_proposal(data)
        return {"unified_diff": diff, "rationale": rationale}


def _changed_paths_of(unified_diff: str) -> list[str]:
    """Repo-relative paths named by a unified diff's ``--- a/<p>`` / ``+++ b/<p>`` headers (``/dev/null``
    excluded). Descriptive only - the human reviews the diff itself."""
    paths: set[str] = set()
    for line in unified_diff.splitlines():
        for marker in ("--- a/", "+++ b/"):
            if line.startswith(marker) and line[len(marker):] != "/dev/null":
                paths.add(line[len(marker):].split("\t", 1)[0])
    return sorted(paths)


def _seal_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    """A tamper-evident ``agent_proposal`` record: the envelope plus a ``record_digest`` over the canonical
    (sorted-key) bytes of everything but the digest itself - so a reviewer can re-verify it."""
    envelope = {"record_type": "agent_proposal", "schema_version": 1, "payload": payload}
    digest = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return {**envelope, "record_digest": f"sha256:{digest}"}


def _write_proposal_record(proposals_dir: Path, record: dict[str, Any]) -> Path:
    """Persist a sealed proposal to a CONTENT-ADDRESSED file (named by its record digest) OUTSIDE the
    working tree, and return the path. Content-addressing makes the name collision-free and idempotent -
    no dependence on directory state or a run counter."""
    proposals_dir.mkdir(parents=True, exist_ok=True)
    out = proposals_dir / f"{record['record_digest'].split(':', 1)[-1]}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _resolve_base_oid(repo_root: Path, base: str) -> str:
    """The 40-hex commit the proposal is against (read-only ``git rev-parse``); '' if it cannot resolve."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _make_executor(agent_client: AgentClient, repo_root: Path, base: str, proposals_dir: Path,
                   worktrees_dir: Path):  # noqa: ANN202
    """Build the propose-only executor. At EXECUTE it runs the agent CONFINED to a disposable worktree and
    seals the returned diff; at DECOMPOSE it installs one self-owned placeholder task (so the graph phases
    proceed without a delegated, write-capable sub-agent); at every other executor phase it advances."""

    def execute(state: LoopState, directive: Directive) -> Observation:
        if state.current_phase is Phase.DECOMPOSE and not state.task_graph:
            state.task_graph = [{
                "id": "propose", "description": "single-agent proposal (no edits applied)",
                "owner": "self", "allowed_paths": [], "depends_on": [], "status": "PENDING",
            }]
            return Observation.SUCCESS
        if state.current_phase is not Phase.EXECUTE:
            return Observation.SUCCESS  # reasoning phases: advance (nothing to propose yet)

        base_oid = _resolve_base_oid(repo_root, base)
        if not base_oid:
            raise AgentError(f"cannot resolve base {base!r} to a commit for the confined worktree")
        # CONFINE the agent: run it with cwd inside a disposable, detached worktree at base (never the
        # developer's tree) + a secret-free env. Anything it writes there is discarded; we use only its
        # returned diff. So even a mis-behaving agent cannot edit the working tree while "proposing".
        with _detached_worktree(repo_root, base_oid, worktrees_dir) as wt:
            request = {"goal": state.goal, "goal_id": state.goal_id, "base_oid": base_oid,
                       "repo_root": str(repo_root), "_cwd": str(wt),
                       "directive": {"phase": directive.phase, "action": directive.action,
                                     "allowed_paths": list(directive.allowed_paths)}}
            result = agent_client.propose(request)  # RAISES on failure -> step escalates (fail-closed)
        diff, rationale = _validate_proposal(result)
        record = _seal_proposal({"goal_id": state.goal_id, "base_oid": base_oid, "unified_diff": diff,
                                 "changed_paths": _changed_paths_of(diff), "rationale": rationale})
        # Persist the sealed proposal OUTSIDE the working tree (content-addressed) and reference it.
        out = _write_proposal_record(proposals_dir, record)
        # Invariant: a written proposal is ALWAYS referenced on the state. Normalize a missing/corrupt
        # value to a list (never silently skip the append and leave disk + state disagreeing).
        refs = state.review_state.get("agent_proposals")
        if not isinstance(refs, list):
            refs = state.review_state["agent_proposals"] = []
        refs.append({"record_digest": record["record_digest"], "path": str(out),
                     "changed_paths": record["payload"]["changed_paths"]})
        return Observation.SUCCESS

    return execute


def _signoff_critic(_state: LoopState) -> list[Criterion]:
    """A proposal is not a completed goal: require a HUMAN_APPROVAL sign-off, so the loop ESCALATES at
    VERIFY (surfacing the proposed diff) instead of autonomously finalizing - the agent cannot self-certify."""
    return [Criterion("agent-proposal-signoff", "a human reviews + decides whether to apply the proposed diff",
                      kind=CriterionKind.HUMAN_APPROVAL)]


def build_context(repo_root: Path | str, base: str = "main", *, agent_client: AgentClient | None = None,
                  proposals_dir: Path | str | None = None, worktrees_dir: Path | str | None = None,
                  pr_ref: str | None = None, run_cs_assure: Any = None, gh_runner: Any = None) -> LoopContext:
    """A READ-ONLY, propose-only :class:`LoopContext` driven by a real single agent. ``capabilities`` is
    EMPTY (the capability gate runs it with no opt-in). The executor runs ``agent_client`` (defaulting to
    the out-of-process :class:`ClaudeSubprocessClient`) CONFINED to a disposable, detached worktree at
    ``base`` (never the developer's tree) and seals the diff it returns; nothing is ever applied, pushed,
    or merged. Pass a stub ``agent_client`` + a ``proposals_dir`` in tests. A ``pr_ref`` additionally
    exercises the real CI read + merge gate, still guaranteed not to merge (``dangerous=True`` escalates
    first and the gh runner refuses mutations)."""
    root = Path(repo_root)
    client = agent_client if agent_client is not None else ClaudeSubprocessClient()
    pdir = Path(proposals_dir) if proposals_dir is not None else _default_proposals_dir(root)
    wtdir = Path(worktrees_dir) if worktrees_dir is not None else default_worktrees_dir(root)
    kwargs: dict[str, Any] = {
        "repo_root": root, "base": base,
        "executor": _make_executor(client, root, base, pdir, wtdir),
        "reviewer": lambda _state: [],
        "critic": _signoff_critic,
        "multi_agent": False,                 # single-agent: no delegated (write-capable) sub-agents
        "capabilities": frozenset(),          # READ-ONLY / propose-only - declares no write capability
    }
    if run_cs_assure is not None:
        kwargs["run_cs_assure"] = run_cs_assure
    if pr_ref is not None:
        kwargs.update(gh_runner=gh_runner or read_only_gh(root), pr_ref=pr_ref, dangerous=True)
    return LoopContext(**kwargs)


def _default_proposals_dir(repo_root: Path) -> Path:
    """``<git-dir>/corpusstudio-loop/proposals`` (OUTSIDE the working tree), or a worktree-local fallback
    when not inside a git repo. A read-only ``git rev-parse`` resolves the git dir."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(repo_root), "rev-parse", "--git-path", "corpusstudio-loop/proposals"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return repo_root / ".corpusstudio-loop-proposals"
    if proc.returncode != 0:
        return repo_root / ".corpusstudio-loop-proposals"
    rel = proc.stdout.strip()
    p = Path(rel)
    return p if p.is_absolute() else repo_root / rel
