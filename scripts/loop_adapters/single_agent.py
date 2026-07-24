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
  * Worktree isolation is a 7.1 concern (there is nothing to isolate when nothing is written); 7.0 proposes
    a diff against ``base`` and stores the sealed record OUTSIDE the working tree.

stdlib-only; every git/gh read and the agent transport fail closed. Read-only ``gh`` + the whole-tree diff
building block are reused from the dry-run adapter.
"""

from __future__ import annotations

import hashlib
import json
import subprocess  # noqa: S404 - fixed-argv git / claude only; never a shell string.
from pathlib import Path
from typing import Any, Protocol

from loop.completeness import Criterion, CriterionKind
from loop.controller import LoopState, Observation, Phase
from loop.driver import Directive
from loop.orchestrate import LoopContext
from loop_adapters.dry_run import read_only_gh  # the same default-deny read-only gh the dry run uses

_AGENT_TIMEOUT_S = 300  # a bounded, killable agent call - never an unbounded hang
_GIT_TIMEOUT_S = 60


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


class ClaudeSubprocessClient:
    """The real transport: run ``claude`` OUT OF PROCESS with a fixed argv (no shell), feed the request as
    JSON on stdin, and read one JSON object with ``unified_diff`` / ``rationale`` from stdout. The call is
    bounded by a timeout (killable); a non-zero exit, an un-runnable binary, or an unparseable / wrong-shaped
    response raises :class:`AgentError`. The exact prompt/flags are operator-tunable via ``argv``; the
    OUTPUT CONTRACT is what the adapter validates. Live behaviour is env-dependent (like ``gh`` auth)."""

    def __init__(self, *, argv: tuple[str, ...] = ("claude", "-p", "--output-format", "json"),
                 timeout: float = _AGENT_TIMEOUT_S) -> None:
        self.argv = argv
        self.timeout = timeout

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, bounded timeout.
                list(self.argv), input=json.dumps(request), text=True, capture_output=True,
                timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentError(f"claude could not run: {type(exc).__name__}: {exc}") from exc
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


def _resolve_base_oid(repo_root: Path, base: str) -> str:
    """The 40-hex commit the proposal is against (read-only ``git rev-parse``); '' if it cannot resolve."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _make_executor(agent_client: AgentClient, repo_root: Path, base: str, proposals_dir: Path):  # noqa: ANN202
    """Build the propose-only executor. At EXECUTE it asks the agent for a diff and seals it; at DECOMPOSE
    it installs one self-owned placeholder task (so the graph phases proceed without a delegated,
    write-capable sub-agent); at every other executor phase it records the directive and advances."""

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
        request = {"goal": state.goal, "goal_id": state.goal_id, "base_oid": base_oid,
                   "repo_root": str(repo_root),
                   "directive": {"phase": directive.phase, "action": directive.action,
                                 "allowed_paths": list(directive.allowed_paths)}}
        result = agent_client.propose(request)  # RAISES on failure -> step escalates (fail-closed)
        diff, rationale = _validate_proposal(result)
        record = _seal_proposal({"goal_id": state.goal_id, "base_oid": base_oid, "unified_diff": diff,
                                 "changed_paths": _changed_paths_of(diff), "rationale": rationale})
        # Persist the sealed proposal OUTSIDE the working tree and reference it on the state.
        proposals_dir.mkdir(parents=True, exist_ok=True)
        index = len([p for p in proposals_dir.glob("*.json")])
        out = proposals_dir / f"{state.goal_id or 'goal'}-{index}.json"
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
                  proposals_dir: Path | str | None = None, pr_ref: str | None = None,
                  run_cs_assure: Any = None, gh_runner: Any = None) -> LoopContext:
    """A READ-ONLY, propose-only :class:`LoopContext` driven by a real single agent. ``capabilities`` is
    EMPTY (the capability gate runs it with no opt-in). The executor asks ``agent_client`` (defaulting to
    the out-of-process :class:`ClaudeSubprocessClient`) for a proposed diff at EXECUTE and seals it; nothing
    is ever applied, pushed, or merged. Pass a stub ``agent_client`` + a ``proposals_dir`` in tests. A
    ``pr_ref`` additionally exercises the real CI read + merge gate, still guaranteed not to merge
    (``dangerous=True`` escalates first and the gh runner refuses mutations)."""
    root = Path(repo_root)
    client = agent_client if agent_client is not None else ClaudeSubprocessClient()
    pdir = Path(proposals_dir) if proposals_dir is not None else _default_proposals_dir(root)
    kwargs: dict[str, Any] = {
        "repo_root": root, "base": base,
        "executor": _make_executor(client, root, base, pdir),
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
