#!/usr/bin/env python3
"""cs_loop: the interactive CLI for the CorpusStudio autonomous engineering loop.

For the case where the LLM/human is the executor: ``cs_loop next`` prints what to do this phase (+ its
constraints); you do it; ``cs_loop observe`` runs the assurance gate and records the classified result;
repeat until the loop is terminal. Deterministic state lives in a JSON file (default ``.loop/state.json``).

  cs_loop init --goal "add a scorer"     # create a loop for a goal
  cs_loop status | inspect               # where the loop is (summary | full state dump)
  cs_loop next                           # the next directive (phase, action, allowed paths, budget)
  cs_loop observe [--base main]          # run cs_assure, classify, route, persist (one interactive step)
  cs_loop pause | resume | abort         # lifecycle control
  cs_loop authorize --grant "..."        # record a human authorization; un-escalate a blocked loop

For the AUTONOMOUS surface, the loop's effects (executor / reviewer / agents / gh / critic) are supplied
by an INJECTED adapter module - the seam a concrete Claude-Code runtime fills:

  cs_loop run --adapters path/to/adapters.py [--base main] [--ledger L.json]
  cs_loop campaign --adapters path/to/adapters.py --goals goals.json [--store-dir D]

An adapter module exposes ``build_context(repo_root: Path, base: str) -> loop.orchestrate.LoopContext``.

stdlib-only; fail-closed (a refusal exits 2, never a bare traceback), mirroring cs_assure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loop.controller import LoopState, Phase  # noqa: E402
from loop.driver import next_directive  # noqa: E402
from loop.observe import LoopObserveError, observe_and_apply  # noqa: E402
from loop.store import LoopStateError, load, save  # noqa: E402

EXIT_OK = 0
EXIT_FAIL_CLOSED = 2
DEFAULT_STATE = ".loop/state.json"


def _emit(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.state)
    if path.exists() and not args.force:
        raise LoopStateError(f"{path} already exists (use --force to overwrite)")
    state = LoopState(goal=args.goal, goal_id=args.goal_id or "", current_phase=Phase.RECEIVE_GOAL)
    save(state, path)
    _emit({"initialized": str(path), "goal": state.goal, "phase": state.current_phase.value})
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    state = load(Path(args.state))
    tasks = [{"id": t.get("id"), "status": t.get("status")} for t in state.task_graph if isinstance(t, dict)]
    _emit({
        "goal": state.goal, "phase": state.current_phase.value, "terminal": state.is_terminal,
        "termination_reason": state.termination_reason, "budgets": state.budgets,
        "tasks": tasks, "observations": len(state.observations),
        "assurance_records": len(state.assurance_records),
    })
    return EXIT_OK


def _cmd_next(args: argparse.Namespace) -> int:
    state = load(Path(args.state))
    _emit(next_directive(state).to_dict())
    return EXIT_OK


def _cmd_observe(args: argparse.Namespace) -> int:
    path = Path(args.state)
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
    state = load(Path(args.state))
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
    path = Path(args.state)
    state = load(path)
    state.review_state["paused"] = True
    save(state, path)
    _emit({"paused": True, "phase": state.current_phase.value})
    return EXIT_OK


def _cmd_resume(args: argparse.Namespace) -> int:
    path = Path(args.state)
    state = load(path)
    state.review_state.pop("paused", None)
    save(state, path)
    _emit({"paused": False, "phase": state.current_phase.value,
           "next": None if state.is_terminal else next_directive(state).to_dict()})
    return EXIT_OK


def _cmd_abort(args: argparse.Namespace) -> int:
    path = Path(args.state)
    state = load(path)
    state.current_phase = Phase.STOPPED
    state.termination_reason = args.reason or "aborted by a human"
    save(state, path)
    _emit({"aborted": True, "phase": "STOPPED", "reason": state.termination_reason})
    return EXIT_OK


def _cmd_authorize(args: argparse.Namespace) -> int:
    # Record a human authorization (audit trail). If the loop ESCALATED waiting for one, move it back to
    # DIAGNOSE so it can be re-driven now that a human has granted the pending decision.
    path = Path(args.state)
    state = load(path)
    state.review_state.setdefault("authorizations", []).append({"grant": args.grant, "note": args.note or ""})
    unescalated = state.current_phase is Phase.ESCALATED
    if unescalated:
        state.current_phase = Phase.DIAGNOSE
        state.termination_reason = None
    save(state, path)
    _emit({"authorized": args.grant, "unescalated": unescalated, "phase": state.current_phase.value})
    return EXIT_OK


def _load_adapters(path: str):  # noqa: ANN202 - a loaded module
    """Import an adapter module (by FILE PATH) that supplies the loop's injected effects. It must expose
    ``build_context(repo_root: Path, base: str)`` returning a ``loop.orchestrate.LoopContext`` (executor /
    reviewer / agent runner / gh / critic). This is the seam a concrete Claude-Code runtime fills."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs_loop_adapters", path)
    if spec is None or spec.loader is None:
        raise LoopStateError(f"cannot load adapter module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_context"):
        raise LoopStateError(f"adapter module {path!r} must define build_context(repo_root, base)")
    return module


def _context(args: argparse.Namespace):  # noqa: ANN202 - a LoopContext
    from dataclasses import replace
    ctx = _load_adapters(args.adapters).build_context(Path(args.repo_root), args.base)
    if args.ledger:
        ctx = replace(ctx, ledger_path=Path(args.ledger))
    return ctx


def _cmd_run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from loop.orchestrate import run_loop
    path = Path(args.state)
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


def _cmd_campaign(args: argparse.Namespace) -> int:
    from loop.campaign import Goal, run_campaign
    try:
        goals_raw = json.loads(Path(args.goals).read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise LoopStateError(f"cannot read goals file {args.goals!r}: {exc}") from exc
    if not isinstance(goals_raw, list):
        raise LoopStateError("goals file must be a JSON list of {goal, goal_id, depends_on?}")
    goals = [Goal(goal=str(g.get("goal", "")), goal_id=str(g.get("goal_id", "")),
                  depends_on=list(g.get("depends_on", []))) for g in goals_raw if isinstance(g, dict)]
    store_dir = Path(args.store_dir) if args.store_dir else None
    outcomes = run_campaign(goals, _context(args), store_dir=store_dir, max_steps=args.max_steps)
    _emit({"outcomes": [{"goal_id": o.goal_id, "final_phase": o.final_phase, "finalized": o.finalized}
                        for o in outcomes]})
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cs_loop", description="Autonomous-loop CLI.")
    parser.add_argument("--state", default=DEFAULT_STATE, help="loop state file (default: .loop/state.json)")
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

    p_auth = sub.add_parser("authorize", help="record a human authorization; un-escalate a blocked loop")
    p_auth.add_argument("--grant", required=True, help="what is being authorized (audit label)")
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
        p.add_argument("--ledger", default="", help="cross-goal learning ledger JSON (optional)")
        p.add_argument("--max-steps", type=int, default=200)
        if name == "campaign":
            p.add_argument("--goals", required=True, help="JSON list of {goal, goal_id, depends_on?}")
            p.add_argument("--store-dir", default="", help="dir for per-goal state files (optional)")
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
