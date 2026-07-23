"""The DRY-RUN runtime adapter: run the loop end-to-end against a REAL repository, but PROPOSE only.

This is the first, deliberately SAFE concrete adapter. It wires the loop's read plane to real effects -
the real ``cs_assure`` reads (change-set / impact / verify / doclint), and read-only ``git`` / ``gh``
building blocks - while the executor is a stand-in that RECORDS what the loop would do at each phase
instead of doing it. It makes NO writes: it never pushes, never merges, never spawns a write-capable
sub-agent. So ``cs_loop run --adapters scripts/loop_adapters/dry_run.py`` drives a goal through the whole
state machine against live repo state and leaves a proposal log + real observations, ending at ESCALATED
(a dry run proposes; a human signs off) rather than autonomously finalizing.

What is wired:
  * ``run_cs_assure`` - the real, repo-bound cs_assure runner (read-only) by default; injectable for tests.
  * ``executor``      - a proposal recorder: logs the per-phase directive to ``review_state`` and returns
                        SUCCESS so the loop advances; at DECOMPOSE it installs one self-owned placeholder
                        task so the graph phases proceed WITHOUT delegating to a (write-capable) agent.
  * ``critic``        - always requires a HUMAN_APPROVAL sign-off -> the loop ESCALATES at VERIFY (it can
                        never certify a real goal complete from a dry run).
  * ``reviewer``      - a clean, no-findings review.
  * ``gh_runner``     - only wired when a ``pr_ref`` is given: a READ-ONLY gh (refuses every mutating
                        subcommand), and ``dangerous=True`` so the merge gate ESCALATES before any merge
                        is attempted. Absent a pr_ref, INTEGRATE is handled by the proposal recorder (no gh
                        call at all).

Building blocks (``git_changed_paths`` / ``read_only_gh``) are also the read effects a future multi-agent /
real-PR adapter needs; they are unit-tested here even though the single-agent dry run does not route
through them. stdlib-only; a failing git/gh read fails closed.
"""

from __future__ import annotations

import subprocess  # noqa: S404 - fixed-argv git/gh reads only; never a shell string.
from pathlib import Path
from typing import Any, Callable

from loop.completeness import Criterion, CriterionKind
from loop.controller import LoopState, Observation, Phase
from loop.driver import Directive
from loop.orchestrate import LoopContext

_READ_TIMEOUT_S = 60

# gh subcommands (by their first two argv tokens) that only READ. Default-DENY: anything not here is
# refused, so a mutating command (merge / close / comment / edit / review / ready / create / api POST) can
# never run through this adapter even by mistake.
_GH_READS = frozenset({
    ("pr", "view"), ("pr", "checks"), ("pr", "status"), ("pr", "list"), ("pr", "diff"),
    ("repo", "view"), ("run", "list"), ("run", "view"),
})


def git_changed_paths(repo_root: Path | str, base: str = "main") -> list[str]:
    """Repo-relative paths that differ from ``base`` (a read-only ``git diff --name-only``). A worktree-diff
    building block for a future multi-agent ``verify_paths``; used directly it diffs the whole tree. Honors
    the PathVerifier contract: it RAISES on a git failure (it never returns ``[]`` to mean 'could not tell')."""
    root = str(repo_root)
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", root, "diff", "--name-only", base, "--"],
            capture_output=True, text=True, timeout=_READ_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"git diff --name-only {base} failed to run: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"git diff --name-only {base} failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
    return [line for line in proc.stdout.splitlines() if line]


def read_only_gh(repo_root: Path | str) -> Callable[..., "tuple[int, str, str]"]:
    """A ``gh`` runner that executes REAL gh for the read-only allowlist and REFUSES everything else. The
    refusal is a non-zero exit (never an exception), so the loop's CI observation fails closed on it."""
    root = str(repo_root)

    def run(*argv: str) -> tuple[int, str, str]:
        if tuple(argv[:2]) not in _GH_READS:
            return (97, "", f"read_only_gh: refused non-read gh '{' '.join(argv[:2])}' (dry-run makes no writes)")
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv (allowlisted read subcommand), no shell.
                ["gh", *argv], cwd=root, capture_output=True, text=True, timeout=_READ_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return (127, "", f"read_only_gh: gh could not run: {exc}")
        return (proc.returncode, proc.stdout, proc.stderr)

    return run


def _record_proposal(state: LoopState, directive: Directive) -> Observation:
    """The dry-run executor: RECORD the proposed action for this phase, then advance (SUCCESS). It does no
    reasoning and makes no edits. At DECOMPOSE it installs one self-owned placeholder task so the graph
    phases proceed without delegating to a (write-capable) agent."""
    proposals = state.review_state.setdefault("dry_run_proposals", [])
    if isinstance(proposals, list):
        proposals.append({"phase": directive.phase, "action": directive.action,
                          "allowed_paths": list(directive.allowed_paths)})
    if state.current_phase is Phase.DECOMPOSE and not state.task_graph:
        state.task_graph = [{
            "id": "dry-run", "description": "dry-run placeholder (no real work performed)",
            "owner": "self", "allowed_paths": [], "depends_on": [], "status": "PENDING",
        }]
    return Observation.SUCCESS


def _signoff_critic(_state: LoopState) -> list[Criterion]:
    """A dry run PROPOSES; it cannot itself certify a real goal complete. Require a HUMAN_APPROVAL sign-off,
    so the loop ESCALATES at VERIFY (surfacing the proposed plan) instead of autonomously finalizing."""
    return [Criterion("dry-run-signoff", "a human reviews the dry run's proposed plan + observations",
                      kind=CriterionKind.HUMAN_APPROVAL)]


def build_context(repo_root: Path | str, base: str = "main", *, pr_ref: str | None = None,
                  run_cs_assure: Any = None, gh_runner: Any = None) -> LoopContext:
    """A read-only, propose-only :class:`LoopContext`. Wires the real cs_assure read plane (override
    ``run_cs_assure`` for tests) and a proposal-recording executor; makes NO writes. Pass a ``pr_ref`` to
    also exercise the real CI read + merge gate against an existing PR - the merge is still guaranteed not
    to happen (``dangerous=True`` escalates the gate first, and the gh runner refuses mutations)."""
    root = Path(repo_root)
    kwargs: dict[str, Any] = {
        "repo_root": root, "base": base,
        "executor": _record_proposal,
        "reviewer": lambda _state: [],   # dry-run: a clean, no-findings review
        "critic": _signoff_critic,
        "multi_agent": False,            # single-agent: no delegated (write-capable) sub-agents
    }
    if run_cs_assure is not None:
        kwargs["run_cs_assure"] = run_cs_assure
    if pr_ref is not None:
        kwargs.update(gh_runner=gh_runner or read_only_gh(root), pr_ref=pr_ref, dangerous=True)
    return LoopContext(**kwargs)
