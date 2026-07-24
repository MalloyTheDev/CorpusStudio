"""Tests for the concrete runtime adapters (scripts/loop_adapters/).

Pins the DRY-RUN adapter: it drives the loop end-to-end to ESCALATED with a proposal log and makes NO
writes; its read-only building blocks (git_changed_paths, read_only_gh) behave and fail closed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loop.controller import LoopState, Phase  # noqa: E402
from loop.orchestrate import LoopContext, run_loop  # noqa: E402
from loop_adapters.dry_run import (  # noqa: E402
    build_context,
    git_changed_paths,
    read_only_gh,
)


def _green_cs_assure():
    """A fast, green cs_assure stand-in (so the dry-run test does not run the real ~1min gate)."""
    steps = [{"name": n, "passed": True, "exit_code": 0, "timed_out": False} for n in ("ruff", "mypy", "pytest")]
    rec = {"verify": {"record_type": "workspace_verification", "schema_version": 2, "record_digest": "sha256:v",
           "payload": {"gate_passed": True, "gate_steps": steps, "workspace_stable": True,
           "fired_obligations": [], "change_set_fingerprint": "cs:x"}},
           "changeset": {"payload": {"changed_paths": []}}, "impact": {"payload": {"fired_obligations": []}},
           "doclint": {"finding_count": 0}}
    return lambda _r, *a: (0, json.dumps(rec.get(a[0] if a else "", {})), "")


# --------------------------------------------------------------------------- the dry-run loop


def test_dry_run_drives_end_to_end_to_escalated_with_a_proposal_log() -> None:
    ctx = build_context(REPO_ROOT, "main", run_cs_assure=_green_cs_assure())
    state = LoopState(goal="add a schema-conformance scorer", current_phase=Phase.RECEIVE_GOAL)
    run_loop(state, ctx, max_steps=100)
    # A dry run PROPOSES; it escalates for a human sign-off rather than autonomously finalizing.
    assert state.current_phase is Phase.ESCALATED
    proposals = state.review_state["dry_run_proposals"]
    assert proposals and {p["phase"] for p in proposals} >= {"RECEIVE_GOAL", "EXECUTE", "INTEGRATE"}


def test_dry_run_makes_no_writes_without_a_pr_ref() -> None:
    # No pr_ref -> no gh runner is wired at all, so the loop can make no gh (push/merge) call.
    ctx = build_context(REPO_ROOT, run_cs_assure=_green_cs_assure())
    assert ctx.gh_runner is None and ctx.pr_ref is None and ctx.multi_agent is False


def test_dry_run_with_a_pr_ref_reads_ci_but_forces_no_merge() -> None:
    # With a pr_ref the real CI read + merge gate are exercised, but dangerous=True guarantees the gate
    # escalates before any merge, and the gh runner is read-only.
    ctx = build_context(REPO_ROOT, pr_ref="42", run_cs_assure=_green_cs_assure())
    assert ctx.gh_runner is not None and ctx.pr_ref == "42" and ctx.dangerous is True
    assert isinstance(ctx, LoopContext)


# --------------------------------------------------------------------------- read-only building blocks


def test_git_changed_paths_reads_and_fails_closed() -> None:
    # A real read against the repo -> a list[str] of repo-relative paths (whatever the current diff is;
    # do NOT assume a clean working tree). Every entry is a string.
    changed = git_changed_paths(REPO_ROOT, "HEAD")
    assert isinstance(changed, list) and all(isinstance(p, str) for p in changed)
    # a bogus base is a git failure -> RAISE (never [] on error, per the PathVerifier contract).
    try:
        git_changed_paths(REPO_ROOT, "not-a-real-ref-xyz")
    except RuntimeError as exc:
        assert "git diff" in str(exc)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("git_changed_paths must raise on a bad base, not return []")


def test_read_only_gh_refuses_every_mutating_subcommand() -> None:
    gh = read_only_gh(REPO_ROOT)
    for mutating in (("pr", "merge", "42"), ("pr", "close", "42"), ("pr", "comment", "42"),
                     ("pr", "edit", "42"), ("pr", "review", "42"), ("pr", "ready", "42"),
                     ("api", "-X", "POST"), ("pr",)):
        code, _out, err = gh(*mutating)
        assert code == 97 and "refused" in err, mutating  # refused WITHOUT invoking gh


def test_read_only_gh_allows_the_read_subcommands() -> None:
    # The allowlist is the set of read pairs the loop's observe_ci uses (pr view / pr checks).
    from loop_adapters.dry_run import _GH_READS
    assert ("pr", "view") in _GH_READS and ("pr", "checks") in _GH_READS
    assert ("pr", "merge") not in _GH_READS  # a mutation is never on the read allowlist
