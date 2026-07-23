"""Long-horizon self-correction (controller slice 9, Level 8).

Two capabilities that make the loop a self-correcting, long-horizon system rather than a one-shot pass:

  * COMPLETENESS CRITIC - at VERIFY, do not FINALIZE just because the gate is green; check the GOAL's
    own success criteria are actually MET. An unmet criterion is a CHANGES_REQUESTED that folds into a
    correction task, so the loop keeps working until the goal is genuinely achieved, not merely compiling.
  * CROSS-GOAL LEARNING - a durable ledger of prior goals' dead ends (failed-approach fingerprints).
    Seeding a new loop from it means the same (failure, approach) another goal already exhausted is
    recognised as a dead end immediately, so effort accumulates across goals/sessions instead of resetting.

Same discipline as the rest of the loop: the critic itself is an INJECTED callback (the real judge is the
LLM / an agent); this module is the deterministic mechanism. stdlib-only, fail-closed.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loop.controller import LoopState, Observation


_LEDGER_MAX_ENTRIES = 500  # bound the cross-goal ledger so it cannot grow without limit


class CompletenessError(Exception):
    """The critic returned a non-Criterion result / raised, or the ledger is malformed (fail-closed)."""


@dataclass(frozen=True)
class Criterion:
    """One goal success criterion and whether the critic judged it MET (with the evidence it cited)."""

    id: str
    description: str
    met: bool = False
    evidence: str = ""


@dataclass(frozen=True)
class CompletenessVerdict:
    complete: bool
    unmet: list[Criterion]
    note: str


# The critic judges each success criterion met/unmet for the current state (the LLM/agent, injected).
Critic = Callable[[LoopState], "list[Criterion]"]


def check_completeness(state: LoopState, critic: Critic) -> CompletenessVerdict:
    """Run the critic and reduce it to a verdict. Fail-closed on a non-Criterion result. A goal with NO
    declared success criteria is INCOMPLETE by default (a goal must define what 'done' means) unless the
    critic explicitly returns criteria - so the loop never declares an unmeasured goal complete."""
    try:
        criteria = critic(state)
    except CompletenessError:
        raise
    except Exception as exc:  # noqa: BLE001 - the injected critic (LLM/agent) is untrusted; fail closed
        raise CompletenessError(f"critic raised {type(exc).__name__}: {exc}") from exc
    if not isinstance(criteria, list) or not all(isinstance(c, Criterion) for c in criteria):
        raise CompletenessError("critic must return a list[Criterion]")
    if not criteria:
        return CompletenessVerdict(False, [], "no success criteria were evaluated; goal completion is unproven")
    # A criterion is met ONLY if `met` is exactly True (a truthy non-bool must not score as met).
    unmet = [c for c in criteria if c.met is not True]
    if unmet:
        return CompletenessVerdict(False, unmet, f"{len(unmet)}/{len(criteria)} success criteria unmet")
    return CompletenessVerdict(True, [], f"all {len(criteria)} success criteria met")


def completeness_observation(verdict: CompletenessVerdict) -> Observation:
    """Complete -> SUCCESS (the loop may FINALIZE); otherwise CHANGES_REQUESTED (keep working the gaps)."""
    return Observation.SUCCESS if verdict.complete else Observation.CHANGES_REQUESTED


def completeness_correction_tasks(verdict: CompletenessVerdict) -> list[dict[str, Any]]:
    """Turn each unmet criterion into a correction task (a goal-completion gap to close)."""
    return [
        {
            "id": f"meet-{c.id}",
            "description": f"Meet unmet success criterion: {c.description}",
            "owner": "self",
            "allowed_paths": [],
            "depends_on": [],
            "success_criteria": [c.description],
            "status": "PENDING",
        }
        for c in verdict.unmet
    ]


# --------------------------------------------------------------------------- cross-goal learning ledger


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise CompletenessError(f"cross-goal ledger is unreadable/malformed ({path}): {exc}") from exc
    if not isinstance(data, list):
        raise CompletenessError(f"cross-goal ledger root must be a list ({path})")
    return [e for e in data if isinstance(e, dict)]


def seed_known_dead_ends(state: LoopState, ledger_path: Path) -> int:
    """Seed ``state.failed_approaches`` with prior goals' failed-approach fingerprints, so a (failure,
    approach) another goal already exhausted is recognised as a dead end immediately. Only EXACT repeats
    are blocked (the fingerprint includes the approach), so a genuinely new approach is still allowed.
    Returns the number of fingerprints seeded."""
    seeded = 0
    for entry in _load_ledger(ledger_path):
        approaches = entry.get("failed_approaches", [])
        if not isinstance(approaches, list):
            continue  # a non-list failed_approaches (e.g. a scalar) must not be iterated / seeded
        for fingerprint in approaches:
            if isinstance(fingerprint, str) and fingerprint not in state.failed_approaches:
                state.failed_approaches.append(fingerprint)
                seeded += 1
    return seeded


def record_outcome(state: LoopState, ledger_path: Path, *, lessons: list[str] | None = None) -> None:
    """Append this goal's outcome + dead ends to the cross-goal ledger (atomic write), so the next goal
    starts from accumulated experience rather than a blank slate."""
    entries = _load_ledger(ledger_path)
    entries.append({
        "goal": state.goal,
        "goal_id": state.goal_id,
        "outcome": state.current_phase.value,
        "termination_reason": state.termination_reason,
        "failed_approaches": copy.deepcopy(list(state.failed_approaches)),
        "lessons": list(lessons or []),
    })
    if len(entries) > _LEDGER_MAX_ENTRIES:
        entries = entries[-_LEDGER_MAX_ENTRIES:]  # bound the ledger; keep the most recent goals
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_name(f"{ledger_path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, ledger_path)
