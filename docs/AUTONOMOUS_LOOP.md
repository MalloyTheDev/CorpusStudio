# The Autonomous Engineering Loop

CorpusStudio has a **bounded autonomous engineering loop** — a controller that receives a goal, plans,
executes, observes real results, classifies failures, revises/retries/replans/delegates, reviews,
integrates, verifies, and continues without a human prompting every step. It lives in
[`scripts/loop/`](../scripts/loop/) and is stdlib-only.

This doc is the current-state map of that loop. It is coupled to the code: a change under
`scripts/loop/` is expected to update this doc (see the docs-freshness check in
[`scripts/loop/docs.py`](../scripts/loop/docs.py)).

## Two planes

The loop is deliberately split into two connected systems:

1. **The operational loop (the doer)** — `scripts/loop/`. Decides the next action, executes (via the
   LLM/agents), observes, routes, and continues. This doc.
2. **The assurance / evidence plane (the judge)** — `scripts/assurance/` + `cs_assure`. *Answers*
   questions: what changed, which obligations fired, which checks ran, are the docs stale. The loop
   **queries** it; it does not replace it.

The controller asks the assurance plane for facts and records its sealed evidence; the assurance plane
never runs the engineering itself. See [`PRODUCT_VS_RESEARCH.md`](PRODUCT_VS_RESEARCH.md) for the wider
boundary.

## The state machine

```
RECEIVE_GOAL → RECON → DEFINE_SUCCESS → PLAN → DECOMPOSE → ASSIGN → EXECUTE → OBSERVE → DIAGNOSE
   → { ADVANCE→REVIEW | REVISE→EXECUTE | REPLAN→PLAN | RESCHEDULE→ASSIGN | HOLD→(stay) | ESCALATE | STOP }
   → REVIEW → INTEGRATE → VERIFY → FINALIZE
```

`HOLD` keeps the loop in the current phase, waiting on an external condition (e.g. CI that has not yet
reported), without charging the retry budget — so `INTEGRATE` never advances *past* the merge gate on an
unsettled CI, and the caller re-invokes when the condition may have changed.

`FINALIZE` / `ESCALATED` / `STOPPED` are terminal. The loop is **bounded** (a per-goal attempt budget
*and* a hard driver step cap) and **fails closed** (an unclassifiable observation escalates; a
degenerate budget stops rather than spinning). Human approval is **retained**: an
`AUTHORIZATION_REQUIRED` (credential / dangerous / irreversible / release / sealed-research) or a
worker-lineage impact escalates rather than proceeding.

## The modules

| module | role |
| --- | --- |
| `controller.py` | The routing brain: phases, the failure taxonomy (`Observation`), decision routing, the bounded-retry guard, and the pure `apply()` transition. |
| `store.py` | Durable `LoopState`: deterministic atomic JSON persistence, fail-closed load, crash-resumable. |
| `observe.py` | Reads the repo via `cs_assure verify` + `doclint` and maps the sealed result to one **mechanical** `Observation` (gate step → failure class; obligations → human-gated / worker / drift). |
| `driver.py` | The runnable cycle: emit a per-phase directive, run the executor (or the observer at the verify phases), route, persist. `run()` drives to a terminal phase or a hard step cap. |
| `tasks.py` | The task graph: owned tasks with dependencies and an `allowed_paths` **ownership boundary**; ready/blocked derivation; parallel-safe assignment. |
| `router.py` | The agent router: a parallel-safe wave (≤10 agents), dispatch, and **boundary enforcement** — an agent that edits outside its lane is rejected regardless of what it claims. |
| `review.py` | Review-feedback: accepted findings become **correction tasks** appended to the graph; a clean review advances to `INTEGRATE`. |
| `integrate.py` | CI / PR continuation: observe CI, diagnose a failure into an `Observation`, and the **merge-authorization gate** (product auto-merges; self-modify / worker-lineage / dangerous escalate). |
| `docs.py` | Docs-freshness: a code↔doc coupling check - a change that touches coupled code but not its docs is `CONTRACT_DRIFT` at OBSERVE (never let docs go stale). |
| `orchestrate.py` | **The capstone** - one integrated loop that dispatches each phase to its module (decompose→validate, assign, execute/wave, observe+docs+task-close, review, integrate, verify) with every effect injected. |

The interactive surface is **`scripts/cs_loop.py`** (`cs_loop init --goal … / next / status / observe`) for
the case where the LLM is the executor: `next` says what to do this phase; you do it; `observe` runs the
assurance gate and records the classified result; repeat until terminal.

## The fact / judgment seam

`observe.py` emits only the observations that are **mechanically determinable** from assurance evidence
(gate red → the failing step; obligations → authorization / worker / drift). The **judgment**
observations (`WRONG_PLAN`, `WRONG_HYPOTHESIS`, `OWNERSHIP_COLLISION`, `POLICY_BLOCK`,
`NONDETERMINISTIC`) are the executor's (the LLM's) to assign on top of that baseline — never guessed
from a heuristic.

## The injected-effect pattern

The executor is the LLM/agent; a pure-Python loop cannot run the reasoning, spawn a subagent, run the
gate, or call `gh`. So every **effect** is an injected callback — `executor`, `observer`, the agent
`runner`, the `reviewer`, the `gh` runner. The modules are the deterministic coordination *around* those
effects, which keeps the whole loop testable without any of them.

## Maturity

L4 (executable single-agent loop) → L5 (adaptive recovery) → L6 (coordinated multi-agent task graph)
→ L7 (CI/review/integration autonomy) are implemented. L8 (long-horizon self-correction across goals)
is future work. The controller is a control plane *on top of* the assurance plane, not a replacement
for it.
