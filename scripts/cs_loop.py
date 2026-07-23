#!/usr/bin/env python3
"""cs_loop: the interactive CLI for the CorpusStudio autonomous engineering loop.

For the case where the LLM/human is the executor: ``cs_loop next`` prints what to do this phase (+ its
constraints); you do it; ``cs_loop observe`` runs the assurance gate and records the classified result;
repeat until the loop is terminal. Deterministic state lives in a JSON file (default ``.loop/state.json``).

  cs_loop init --goal "add a scorer"     # create a loop for a goal
  cs_loop status                         # where the loop is + budget + tasks
  cs_loop next                           # the next directive (phase, action, allowed paths, budget)
  cs_loop observe [--base main]          # run cs_assure, classify, route, persist

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cs_loop", description="Interactive autonomous-loop CLI.")
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
