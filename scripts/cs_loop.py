#!/usr/bin/env python3
"""cs_loop: the interactive CLI for the CorpusStudio autonomous engineering loop.

For the case where the LLM/human is the executor: ``cs_loop next`` prints what to do this phase (+ its
constraints); you do it; ``cs_loop observe`` runs the assurance gate and records the classified result;
repeat until the loop is terminal. Deterministic state lives in a JSON file that defaults UNDER THE GIT DIR
(``<git-dir>/corpusstudio-loop/state.json``), OUTSIDE the worktree, so it never contaminates the assurance
change-set fingerprint; pass ``--state`` to override.

  cs_loop init --goal "add a scorer"     # create a loop for a goal
  cs_loop status | inspect               # where the loop is (summary | full state dump)
  cs_loop next                           # the next directive (phase, action, allowed paths, budget)
  cs_loop observe [--base main]          # run cs_assure, classify, route, persist (one interactive step)
  cs_loop pause | resume | abort         # lifecycle control
  cs_loop authorize [--request <id>]     # show the pending escalation, or grant that SPECIFIC request

For the AUTONOMOUS surface, the loop's effects (executor / reviewer / agents / gh / critic) are supplied
by an INJECTED adapter module - the seam a concrete Claude-Code runtime fills:

  cs_loop run --adapters path/to/adapters.py [--base main] [--ledger L.json]
  cs_loop campaign --adapters path/to/adapters.py --goals goals.json [--store-dir D]

An adapter module exposes ``build_context(repo_root: Path, base: str) -> loop.orchestrate.LoopContext``.
For a campaign it MAY also expose ``build_context_for_goal(goal, repo_root, base, campaign_dir) ->
LoopContext`` - a PER-GOAL isolation factory (each goal its own branch/worktree/PR/state); when present,
``cs_loop campaign`` uses it instead of one shared context.

stdlib-only; fail-closed (a refusal exits 2, never a bare traceback), mirroring cs_assure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # noqa: S404 - fixed-argv `git rev-parse` only; never a shell string.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contextlib import contextmanager  # noqa: E402
from typing import Iterator  # noqa: E402

from loop.controller import LoopState, Phase  # noqa: E402
from loop.driver import next_directive  # noqa: E402
from loop.locking import FileLock  # noqa: E402 - single-writer enforcement on the state file
from loop.observe import LoopObserveError, observe_and_apply  # noqa: E402
from loop.store import LoopStateError, load, save  # noqa: E402


@contextmanager
def _state_write_lock(path: Path, *, timeout: float = 10.0) -> "Iterator[None]":
    """Hold the single-writer lock while a command READ-MODIFY-WRITES the state file, so a concurrent
    writer (e.g. a ``cs_loop run`` in progress, which holds this same lock) FAILS CLOSED rather than
    silently clobbering the other's update. A LockError propagates to main() -> exit 2 (fail-closed)."""
    with FileLock(path, timeout=timeout):
        yield

EXIT_OK = 0
EXIT_FAIL_CLOSED = 2
# The loop's operational state (state file, per-goal campaign states, the learning ledger, and - since
# they live inside the state - the proposal log + authorization records) defaults UNDER THE GIT DIR, not
# in the worktree. A state file in the worktree would be a non-ignored untracked file that the change-set
# kernel folds into the assurance fingerprint, so every save (including after an assurance observation)
# would immediately stale the record it just produced. Placing it at `git rev-parse --git-path` keeps it
# out of the change set and isolates it per (linked) worktree.
_OP_DIR_NAME = "corpusstudio-loop"
_WORKTREE_FALLBACK = ".loop"  # only used when the target is not inside a git repo (then there is no gate)


def _emit(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _operational_dir(repo_root: str = ".") -> Path | None:
    """The directory for the loop's operational state, OUTSIDE the worktree (under the git dir). Returns
    None if ``repo_root`` is not inside a git repo (then there is no assurance change set to contaminate)."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell.
            ["git", "-C", repo_root, "rev-parse", "--git-path", _OP_DIR_NAME],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    rel = proc.stdout.strip()
    if proc.returncode != 0 or not rel:
        return None
    return (Path(repo_root) / rel).resolve()  # join handles both a relative and an absolute git-path result


def _state_path(args: argparse.Namespace) -> Path:
    """Resolve the loop state file: an explicit ``--state`` wins; otherwise default under the git dir
    (``<git-dir>/corpusstudio-loop/state.json``), falling back to ``.loop/state.json`` only outside a repo."""
    if args.state:
        return Path(args.state)
    op = _operational_dir(getattr(args, "repo_root", ".") or ".")
    return (op / "state.json") if op is not None else Path(_WORKTREE_FALLBACK) / "state.json"


def _cmd_init(args: argparse.Namespace) -> int:
    path = _state_path(args)
    with _state_write_lock(path):
        if path.exists() and not args.force:
            raise LoopStateError(f"{path} already exists (use --force to overwrite)")
        state = LoopState(goal=args.goal, goal_id=args.goal_id or "", current_phase=Phase.RECEIVE_GOAL)
        save(state, path)
    _emit({"initialized": str(path), "goal": state.goal, "phase": state.current_phase.value})
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    state = load(_state_path(args))
    tasks = [{"id": t.get("id"), "status": t.get("status")} for t in state.task_graph if isinstance(t, dict)]
    _emit({
        "goal": state.goal, "phase": state.current_phase.value, "terminal": state.is_terminal,
        "termination_reason": state.termination_reason, "budgets": state.budgets,
        "tasks": tasks, "observations": len(state.observations),
        "assurance_records": len(state.assurance_records),
    })
    return EXIT_OK


def _cmd_next(args: argparse.Namespace) -> int:
    state = load(_state_path(args))
    _emit(next_directive(state).to_dict())
    return EXIT_OK


def _cmd_observe(args: argparse.Namespace) -> int:
    path = _state_path(args)
    with _state_write_lock(path):
        state = load(path)
        if state.is_terminal:
            raise LoopStateError(f"loop is already terminal ({state.current_phase.value}); nothing to observe")
        transition = observe_and_apply(state, Path(args.repo_root), args.base)
        save(state, path)
    _emit({
        "decision": transition.decision.value, "phase": state.current_phase.value,
        "note": transition.note, "terminal": state.is_terminal,
        "termination_reason": state.termination_reason,
    })
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    state = load(_state_path(args))
    _emit({
        "goal": state.goal, "goal_id": state.goal_id, "phase": state.current_phase.value,
        "terminal": state.is_terminal, "termination_reason": state.termination_reason,
        "success_criteria": state.success_criteria, "budgets": state.budgets,
        "task_graph": state.task_graph, "observations": state.observations,
        "hypotheses": state.hypotheses, "failed_approaches": state.failed_approaches,
        "assurance_records": state.assurance_records, "blockers": state.blockers,
        "review_state": state.review_state,
    })
    return EXIT_OK


def _cmd_pause(args: argparse.Namespace) -> int:
    path = _state_path(args)
    with _state_write_lock(path):
        state = load(path)
        state.review_state["paused"] = True
        save(state, path)
    _emit({"paused": True, "phase": state.current_phase.value})
    return EXIT_OK


def _cmd_resume(args: argparse.Namespace) -> int:
    path = _state_path(args)
    with _state_write_lock(path):
        state = load(path)
        state.review_state.pop("paused", None)
        save(state, path)
    _emit({"paused": False, "phase": state.current_phase.value,
           "next": None if state.is_terminal else next_directive(state).to_dict()})
    return EXIT_OK


def _cmd_abort(args: argparse.Namespace) -> int:
    path = _state_path(args)
    with _state_write_lock(path):
        state = load(path)
        state.current_phase = Phase.STOPPED
        state.termination_reason = args.reason or "aborted by a human"
        state.review_state.pop("paused", None)  # a STOPPED loop is unambiguous - never both terminal + paused
        save(state, path)
    _emit({"aborted": True, "phase": "STOPPED", "reason": state.termination_reason})
    return EXIT_OK


def _pending_authorization(state: LoopState) -> dict[str, str] | None:
    """The pending human-authorization REQUEST for an ESCALATED loop (else None). Its ``request_id`` is
    the deterministic identity of the CURRENT blocker (goal + termination reason), so a grant must name
    THIS specific request - it never universally un-escalates an unrelated blocker, and it goes STALE if
    the loop later escalates for a different reason (the id changes)."""
    if state.current_phase is not Phase.ESCALATED:
        return None
    # The id is DETERMINISTIC on (goal, blocker) BY DESIGN: re-escalating for the same blocker yields the
    # same id, so a valid grant persists and is idempotent; a different blocker (reason) yields a new id
    # (staleness). No timestamp/step index - that would break both idempotency and the loop's no-wall-clock
    # determinism. The finer per-subject discriminator (change_set_fingerprint / head) is #5 slice 2.
    reason = state.termination_reason or "(no reason recorded)"
    rid = "auth-" + hashlib.sha256(f"{state.goal_id}\x00{reason}".encode("utf-8")).hexdigest()[:16]
    return {"request_id": rid, "goal_id": state.goal_id, "capability": reason}


def _cmd_authorize(args: argparse.Namespace) -> int:
    # A human authorizes a SPECIFIC pending request by its id (not a free-form blanket un-escalate). With
    # no --request, SHOW the pending request so the human learns its id; with a matching --request, record
    # the decision (bound to the request + any completeness --grant) and un-escalate ONLY that blocker.
    path = _state_path(args)
    with _state_write_lock(path):
        state = load(path)
        pending = _pending_authorization(state)
        if not args.request:
            if pending is None:
                _emit({"pending_authorization": None,
                       "note": "the loop is not ESCALATED; there is nothing to authorize"})
            else:
                _emit({"pending_authorization": pending,
                       "note": "re-run: cs_loop authorize --request <request_id> [--grant <criterion-id>]"})
            return EXIT_OK
        if pending is None:
            raise LoopStateError("the loop is not ESCALATED; there is no pending authorization request")
        if args.request != pending["request_id"]:
            raise LoopStateError(
                f"request {args.request!r} does not match the pending request {pending['request_id']!r} - a "
                "grant must name the CURRENT blocker (it never universally un-escalates an unrelated one)")
        # Keep a `grant` field (the completeness criterion id, if any) so a HUMAN_APPROVAL criterion is
        # still satisfied by grant == criterion.id; add the request binding + the capability authorized.
        state.review_state.setdefault("authorizations", []).append({
            "request_id": pending["request_id"], "capability": pending["capability"],
            "grant": args.grant or "", "note": args.note or "", "granted": True})
        state.current_phase = Phase.DIAGNOSE
        state.termination_reason = None
        save(state, path)
    _emit({"authorized": pending["request_id"], "grant": args.grant or "",
           "phase": state.current_phase.value})
    return EXIT_OK


def _load_adapters(path: str):  # noqa: ANN202 - a loaded module
    """Import an adapter module (by FILE PATH) that supplies the loop's injected effects. It must expose
    ``build_context(repo_root: Path, base: str)`` returning a ``loop.orchestrate.LoopContext`` (executor /
    reviewer / agent runner / gh / critic). This is the seam a concrete Claude-Code runtime fills.

    The module is registered under a name DERIVED FROM ITS ABSOLUTE PATH (not a fixed ``cs_loop_adapters``),
    so two different adapters loaded in one process (tests, an embedding runtime) do not collide, and it is
    placed in ``sys.modules`` before execution per the importlib recipe so an adapter that defines
    dataclasses or references itself resolves."""
    import hashlib
    import importlib.util
    abs_path = str(Path(path).resolve())
    mod_name = "cs_loop_adapters_" + hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise LoopStateError(f"cannot load adapter module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # so self-reference / dataclasses in the adapter resolve
    try:
        spec.loader.exec_module(module)
    except (OSError, SyntaxError, ImportError) as exc:
        sys.modules.pop(mod_name, None)
        raise LoopStateError(f"cannot load adapter module {path!r}: {type(exc).__name__}: {exc}") from exc
    if not hasattr(module, "build_context"):
        sys.modules.pop(mod_name, None)
        raise LoopStateError(f"adapter module {path!r} must define build_context(repo_root, base)")
    return module


def _default_ledger(args: argparse.Namespace) -> Path | None:
    """The cross-goal learning ledger path: an explicit ``--ledger`` wins; otherwise default under the git
    dir (OUTSIDE the worktree). None only when not inside a git repo (then no ledger is wired)."""
    if args.ledger:
        return Path(args.ledger)
    op = _operational_dir(getattr(args, "repo_root", ".") or ".")
    return (op / "ledger.json") if op is not None else None


def _campaign_store_dir(args: argparse.Namespace) -> Path | None:
    """The per-goal campaign state directory: an explicit ``--store-dir`` wins; otherwise default under the
    git dir (OUTSIDE the worktree), so per-goal state files never enter the change set."""
    if args.store_dir:
        return Path(args.store_dir)
    op = _operational_dir(getattr(args, "repo_root", ".") or ".")
    return (op / "campaigns") if op is not None else None


def _wire_ledger(ctx, ledger):  # noqa: ANN001,ANN202 - a LoopContext
    """Give a LoopContext the shared cross-goal ledger, unless the adapter already set one. A non-
    LoopContext is returned untouched so the campaign layer's fail-closed validation reports it clearly."""
    from dataclasses import replace

    from loop.orchestrate import LoopContext
    if ledger is not None and isinstance(ctx, LoopContext) and ctx.ledger_path is None:
        return replace(ctx, ledger_path=ledger)
    return ctx


def _context(args: argparse.Namespace):  # noqa: ANN202 - a LoopContext
    ctx = _load_adapters(args.adapters).build_context(Path(args.repo_root), args.base)
    return _wire_ledger(ctx, _default_ledger(args))


def _cmd_run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from loop.orchestrate import run_loop
    path = _state_path(args)
    state = load(path)
    if state.review_state.get("paused"):
        raise LoopStateError("loop is paused; `cs_loop resume` before running")
    if state.is_terminal:
        raise LoopStateError(f"loop is already terminal ({state.current_phase.value})")
    ctx = replace(_context(args), store_path=path)  # persist to the state file each cycle
    run_loop(state, ctx, max_steps=args.max_steps)
    _emit({"phase": state.current_phase.value, "terminal": state.is_terminal,
           "termination_reason": state.termination_reason, "observations": len(state.observations),
           "tasks": [{"id": t.get("id"), "status": t.get("status")} for t in state.task_graph
                     if isinstance(t, dict)]})
    return EXIT_OK


def _goals_from_json(goals_raw: object):  # noqa: ANN202 - a list[loop.campaign.Goal]
    """Parse the goals JSON into Goal objects, fail-closed. Every entry must be an object; ``depends_on``
    (if present) must be a list of strings. We do NOT coerce - ``list("g1")`` would silently become
    ``['g','1']`` and read as a valid-looking dependency, so a mistyped config is rejected here with a
    clear message instead of failing later with a confusing "unknown goal 'g'". No goal is silently
    dropped (honesty: no silent truncation)."""
    from loop.campaign import Goal
    if not isinstance(goals_raw, list):
        raise LoopStateError("goals file must be a JSON list of {goal, goal_id, depends_on?}")
    goals = []
    for index, g in enumerate(goals_raw):
        if not isinstance(g, dict):
            raise LoopStateError(f"goals[{index}] must be an object, got {type(g).__name__}")
        deps_raw = g.get("depends_on", [])
        if deps_raw is None:
            deps_raw = []
        if not isinstance(deps_raw, list) or not all(isinstance(d, str) for d in deps_raw):
            raise LoopStateError(
                f"goals[{index}] 'depends_on' must be a list of strings (or null), "
                f"got {type(deps_raw).__name__}")
        goals.append(Goal(goal=str(g.get("goal", "")), goal_id=str(g.get("goal_id", "")),
                          depends_on=list(deps_raw)))
    return goals


def _cmd_campaign(args: argparse.Namespace) -> int:
    from loop.campaign import run_campaign
    try:
        goals_raw = json.loads(Path(args.goals).read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise LoopStateError(f"cannot read goals file {args.goals!r}: {exc}") from exc
    module = _load_adapters(args.adapters)
    goals = _goals_from_json(goals_raw)
    store_dir = _campaign_store_dir(args)
    ledger = _default_ledger(args)
    factory = getattr(module, "build_context_for_goal", None)
    if callable(factory):  # callable, not merely present - a non-callable attribute is not a factory
        # PER-GOAL ISOLATION: the adapter exposes a factory, so each goal gets its OWN LoopContext (the
        # seam a runtime fills with a per-goal branch / worktree / PR / state). Required for a
        # write-capable multi-goal campaign, where a shared working tree would let goals clobber each other.
        def context_for(goal):  # noqa: ANN001,ANN202 - a LoopContext
            gctx = factory(goal, Path(args.repo_root), args.base, store_dir)
            return _wire_ledger(gctx, ledger)
        outcomes = run_campaign(goals, context_for=context_for, store_dir=store_dir, max_steps=args.max_steps)
    else:
        ctx = _wire_ledger(module.build_context(Path(args.repo_root), args.base), ledger)
        outcomes = run_campaign(goals, ctx, store_dir=store_dir, max_steps=args.max_steps)
    _emit({"outcomes": [{"goal_id": o.goal_id, "final_phase": o.final_phase, "finalized": o.finalized,
                         "status": o.status} for o in outcomes]})
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cs_loop", description="Autonomous-loop CLI.")
    parser.add_argument("--state", default="",
                        help="loop state file (default: <git-dir>/corpusstudio-loop/state.json, outside the worktree)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a loop for a goal")
    p_init.add_argument("--goal", required=True)
    p_init.add_argument("--goal-id", default="")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing state file")
    p_init.set_defaults(func=_cmd_init)

    sub.add_parser("status", help="show where the loop is").set_defaults(func=_cmd_status)
    sub.add_parser("next", help="print the next directive").set_defaults(func=_cmd_next)

    p_obs = sub.add_parser("observe", help="run the assurance gate, classify + route, persist")
    p_obs.add_argument("--base", default="main")
    p_obs.add_argument("--repo-root", default=".")
    p_obs.set_defaults(func=_cmd_observe)

    sub.add_parser("inspect", help="full loop-state dump").set_defaults(func=_cmd_inspect)
    sub.add_parser("pause", help="pause the loop (run/step refuse until resume)").set_defaults(func=_cmd_pause)
    sub.add_parser("resume", help="clear the pause + show the next directive").set_defaults(func=_cmd_resume)

    p_abort = sub.add_parser("abort", help="stop the loop (terminal)")
    p_abort.add_argument("--reason", default="")
    p_abort.set_defaults(func=_cmd_abort)

    p_auth = sub.add_parser("authorize",
                            help="show the pending authorization request, or grant it by --request <id>")
    p_auth.add_argument("--request", default="",
                        help="the request_id of the pending escalation to authorize (run `authorize` with no args to see it)")
    p_auth.add_argument("--grant", default="", help="a completeness criterion id this authorization satisfies (optional)")
    p_auth.add_argument("--note", default="")
    p_auth.set_defaults(func=_cmd_authorize)

    for name, help_text, cmd in (
        ("run", "drive the integrated loop to terminal/HOLD/step-cap via an adapter module", _cmd_run),
        ("campaign", "run a queue of goals (goals JSON) via an adapter module", _cmd_campaign),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--adapters", required=True, help="path to a module exposing build_context(repo_root, base)")
        p.add_argument("--base", default="main")
        p.add_argument("--repo-root", default=".")
        p.add_argument("--ledger", default="",
                       help="cross-goal learning ledger JSON (default: <git-dir>/corpusstudio-loop/ledger.json)")
        p.add_argument("--max-steps", type=int, default=200)
        if name == "campaign":
            p.add_argument("--goals", required=True, help="JSON list of {goal, goal_id, depends_on?}")
            p.add_argument("--store-dir", default="",
                           help="per-goal state dir (default: <git-dir>/corpusstudio-loop/campaigns)")
        p.set_defaults(func=cmd)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (LoopStateError, LoopObserveError) as exc:
        sys.stderr.write(f"cs_loop: {type(exc).__name__}: {exc}\n")
        return EXIT_FAIL_CLOSED
    except Exception as exc:  # noqa: BLE001 - fail-closed backstop; never leak a bare traceback as exit 1
        sys.stderr.write(f"cs_loop: UnexpectedRefusal: {type(exc).__name__}: {exc}\n")
        return EXIT_FAIL_CLOSED


if __name__ == "__main__":
    raise SystemExit(main())
