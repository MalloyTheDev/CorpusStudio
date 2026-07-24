"""Phase 7.1 runtime adapter: write-capable single agent (GATED - needs an explicit operator opt-in).

The write step of the Production Single-Agent Runtime (docs/PRODUCTION_SINGLE_AGENT_RUNTIME.md). The agent
still PROPOSES a unified diff (exactly the 7.0 behaviour, sealed as an ``agent_proposal`` record); 7.1 then
APPLIES that exact sealed diff in an ISOLATED, throwaway ``git worktree`` - never the developer's working
tree - commits it on a fresh branch, pushes the branch, and opens a PR. It NEVER merges (the merge gate
escalates; a human merges the PR).

Safety (why this is the least-capable *write* rung):
  * ``capabilities = {"write"}`` - the capability gate REFUSES to run it without ``--allow-capabilities
    write``. The read-only 7.0 adapter is untouched; this write path is a separate, separately-reviewed file.
  * The unit of change is the agent's OWN sealed diff, applied deterministically with ``git apply`` - the
    agent does not edit arbitrarily. After applying, the worktree's real staged diff is verified to match
    the sealed proposal's ``changed_paths`` (integrity), and a mismatch fails closed.
  * All edits/commits happen in an isolated worktree created from ``base``; the main working tree is never
    touched. The worktree is disposed on exit (the pushed branch persists as the PR head).
  * NO autonomous merge: the ``gh`` runner allows ``pr create`` + reads but REFUSES ``pr merge`` (and every
    other mutation), and ``dangerous=True`` escalates the merge gate; a human reviews + merges the PR.

stdlib-only; every git/gh effect is a fixed-argv subprocess (no shell) and fails closed. Reuses the 7.0
building blocks (the agent client, proposal sealing, diff parsing).
"""

from __future__ import annotations

import subprocess  # noqa: S404 - fixed-argv git / gh only; never a shell string.
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from loop.controller import LoopState, Observation, Phase
from loop.driver import Directive
from loop.orchestrate import CAP_WRITE, LoopContext
from loop_adapters.single_agent import (
    AgentClient,
    AgentError,
    ClaudeSubprocessClient,
    _changed_paths_of,
    _default_proposals_dir,
    _resolve_base_oid,
    _seal_proposal,
    _signoff_critic,
    _validate_proposal,
)

_GIT_TIMEOUT_S = 120
_GH_TIMEOUT_S = 60

# gh subcommands the WRITE adapter permits: the read-only allowlist PLUS ``pr create``. Everything else -
# and crucially ``pr merge`` / ``pr close`` / ``pr edit`` / ``pr review`` / any ``api`` write - is REFUSED,
# so 7.1 can open a PR but can NEVER merge or otherwise mutate a PR autonomously.
_GH_WRITE_ALLOWED = frozenset({
    ("pr", "create"),
    ("pr", "view"), ("pr", "checks"), ("pr", "status"), ("pr", "list"), ("pr", "diff"),
    ("repo", "view"), ("run", "list"), ("run", "view"),
})


class WriteAdapterError(RuntimeError):
    """A git/gh write effect failed or produced output the adapter refuses to trust (fail-closed)."""


def write_gh(repo_root: Path | str) -> Callable[..., "tuple[int, str, str]"]:
    """A ``gh`` runner that runs the write-adapter allowlist (reads + ``pr create``) and REFUSES everything
    else - most importantly ``pr merge``. A refusal is a non-zero exit (never an exception), so the loop's
    CI/merge observation fails closed on it."""
    root = str(repo_root)

    def run(*argv: str) -> tuple[int, str, str]:
        if tuple(argv[:2]) not in _GH_WRITE_ALLOWED:
            return (97, "", f"write_gh: refused non-allowlisted gh '{' '.join(argv[:2])}' (7.1 never merges)")
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv (allowlisted subcommand), no shell.
                ["gh", *argv], cwd=root, capture_output=True, text=True, timeout=_GH_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return (127, "", f"write_gh: gh could not run: {exc}")
        return (proc.returncode, proc.stdout, proc.stderr)

    return run


def _git(cwd: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``git -C <cwd> <args>`` (fixed argv, no shell, bounded). Raises :class:`WriteAdapterError` on a
    non-zero exit or an un-runnable git."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", str(cwd), *args], input=stdin, capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WriteAdapterError(f"git {' '.join(args[:2])} could not run: {exc}") from exc
    if proc.returncode != 0:
        raise WriteAdapterError(f"git {' '.join(args[:2])} failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    return proc


@contextmanager
def _worktree(repo_root: Path, base_oid: str, branch: str, worktrees_dir: Path) -> Iterator[Path]:
    """An ISOLATED, disposable ``git worktree`` checked out on a NEW ``branch`` at ``base_oid``, under
    ``worktrees_dir`` (outside the main working tree). Removed on exit (the branch persists as the PR head);
    a best-effort ``worktree remove --force`` never masks the primary error."""
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt = worktrees_dir / branch.replace("/", "-")
    _git(repo_root, "worktree", "add", "-b", branch, str(wt), base_oid)
    try:
        yield wt
    finally:
        try:
            _git(repo_root, "worktree", "remove", "--force", str(wt))
        except WriteAdapterError:
            pass  # best-effort disposal; a leftover worktree is GC-able, never data loss


def _make_write_executor(agent_client: AgentClient, repo_root: Path, base: str, proposals_dir: Path,
                         worktrees_dir: Path, gh_runner: Callable[..., "tuple[int, str, str]"],
                         branch_prefix: str):  # noqa: ANN202
    """The write executor: at EXECUTE, PROPOSE (agent) -> seal -> APPLY the exact diff in an isolated
    worktree -> verify the applied change matches the proposal -> commit -> push the branch -> open a PR.
    At DECOMPOSE it installs one self-owned task. Every step fails closed (raises -> the loop escalates)."""

    def execute(state: LoopState, directive: Directive) -> Observation:
        if state.current_phase is Phase.DECOMPOSE and not state.task_graph:
            state.task_graph = [{
                "id": "apply-and-pr", "description": "apply the agent's diff in an isolated worktree + open a PR",
                "owner": "self", "allowed_paths": [], "depends_on": [], "status": "PENDING",
            }]
            return Observation.SUCCESS
        if state.current_phase is not Phase.EXECUTE:
            return Observation.SUCCESS

        base_oid = _resolve_base_oid(repo_root, base)
        if not base_oid:
            raise WriteAdapterError(f"cannot resolve base {base!r} to a commit to branch from")
        request = {"goal": state.goal, "goal_id": state.goal_id, "base_oid": base_oid,
                   "repo_root": str(repo_root),
                   "directive": {"phase": directive.phase, "action": directive.action,
                                 "allowed_paths": list(directive.allowed_paths)}}
        diff, rationale = _validate_proposal(agent_client.propose(request))  # RAISES -> fail-closed
        record = _seal_proposal({"goal_id": state.goal_id, "base_oid": base_oid, "unified_diff": diff,
                                 "changed_paths": _changed_paths_of(diff), "rationale": rationale})
        proposals_dir.mkdir(parents=True, exist_ok=True)
        (proposals_dir / f"{state.goal_id or 'goal'}-{len(list(proposals_dir.glob('*.json')))}.json").write_text(
            _dumps(record), encoding="utf-8")

        branch = f"{branch_prefix}{state.goal_id or 'goal'}"
        with _worktree(repo_root, base_oid, branch, worktrees_dir) as wt:
            # Apply the agent's OWN sealed diff, staged. A diff that does not apply cleanly fails closed.
            _git(wt, "apply", "--index", "-", stdin=diff)
            # INTEGRITY: the staged change must be exactly what the sealed proposal described.
            applied = sorted(p for p in _git(wt, "diff", "--cached", "--name-only").stdout.splitlines() if p)
            if applied != record["payload"]["changed_paths"]:
                raise WriteAdapterError(
                    f"applied paths {applied} != proposed {record['payload']['changed_paths']} (diff drift)")
            _git(wt, "-c", "user.name=corpusstudio-agent", "-c", "user.email=agent@corpusstudio.local",
                 "commit", "-m", f"{state.goal or 'agent change'}\n\n{rationale}\n\n[single-agent proposal, human-reviewed]")
            _git(wt, "push", "-u", "origin", branch)
            code, out, err = gh_runner("pr", "create", "--head", branch, "--base", base,
                                       "--title", (state.goal or "agent change")[:120],
                                       "--body", rationale or "Opened by the single-agent write runtime (7.1).")
            if code != 0:
                raise WriteAdapterError(f"gh pr create failed (exit {code}): {err.strip()[:200]}")

        refs = state.review_state.get("agent_proposals")
        if not isinstance(refs, list):
            refs = state.review_state["agent_proposals"] = []
        refs.append({"record_digest": record["record_digest"], "branch": branch,
                     "changed_paths": record["payload"]["changed_paths"], "pr": out.strip()})
        return Observation.SUCCESS

    return execute


def _dumps(record: dict[str, Any]) -> str:
    import json  # noqa: PLC0415 - local to keep the module import light
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def build_context(repo_root: Path | str, base: str = "main", *, agent_client: AgentClient | None = None,
                  proposals_dir: Path | str | None = None, worktrees_dir: Path | str | None = None,
                  gh_runner: Any = None, branch_prefix: str = "cs-agent/", run_cs_assure: Any = None,
                  pr_ref: str | None = None) -> LoopContext:
    """A WRITE-CAPABLE, single-agent :class:`LoopContext` (Phase 7.1). ``capabilities={CAP_WRITE}`` - the
    capability gate REFUSES to run it without ``--allow-capabilities write``. It applies the agent's sealed
    diff in an isolated worktree, commits, pushes a branch, and opens a PR; it NEVER merges (``write_gh``
    refuses ``pr merge`` and ``dangerous=True`` escalates the merge gate). Inject a stub ``agent_client`` +
    dirs + ``gh_runner`` in tests."""
    root = Path(repo_root)
    client = agent_client if agent_client is not None else ClaudeSubprocessClient()
    pdir = Path(proposals_dir) if proposals_dir is not None else _default_proposals_dir(root)
    wdir = Path(worktrees_dir) if worktrees_dir is not None else _default_worktrees_dir(root)
    gh = gh_runner or write_gh(root)
    kwargs: dict[str, Any] = {
        "repo_root": root, "base": base,
        "executor": _make_write_executor(client, root, base, pdir, wdir, gh, branch_prefix),
        "reviewer": lambda _state: [],
        "critic": _signoff_critic,
        "multi_agent": False,               # single-agent: no delegated wave (verify_paths is a 7.3 concern)
        "capabilities": frozenset({CAP_WRITE}),
        "gh_runner": gh,
        "dangerous": True,                  # the merge gate ESCALATES: a human merges the PR, never the loop
    }
    if run_cs_assure is not None:
        kwargs["run_cs_assure"] = run_cs_assure
    if pr_ref is not None:
        kwargs["pr_ref"] = pr_ref
    return LoopContext(**kwargs)


def _default_worktrees_dir(repo_root: Path) -> Path:
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
    p = Path(rel)
    return p if p.is_absolute() else repo_root / rel


# Re-export so `AgentError` is catchable via this module too (the executor may surface either).
__all__ = ["AgentError", "WriteAdapterError", "build_context", "write_gh"]
