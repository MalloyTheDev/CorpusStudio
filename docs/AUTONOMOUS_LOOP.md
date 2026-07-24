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
| `observe.py` | Reads the repo via `cs_assure verify` + `doclint` and maps the sealed result to one **mechanical** `Observation` (gate step → failure class; obligations → human-gated / worker / drift). The verify result is a **validated record**, not raw JSON: a refusal exit (≥2), a wrong `record_type`, an unsupported schema (< v2), a missing change-set fingerprint, or `workspace_stable != True` (the tree mutated during the gate) each fails closed; an unavailable `doclint` is surfaced explicitly, never silently read as clean. |
| `driver.py` | The runnable cycle: emit a per-phase directive, run the executor (or the observer at the verify phases), route, persist. `run()` drives to a terminal phase or a hard step cap. |
| `tasks.py` | The task graph: owned tasks with dependencies and an `allowed_paths` **ownership boundary** (a declared lane must be a real subpath — absolute / `..` / and the whole-repo `.` are rejected; a repo-wide task uses an empty lane and runs self-owned); ready/blocked derivation; parallel-safe assignment. |
| `router.py` | The agent router: a parallel-safe wave (≤10 agents), dispatch, and **boundary enforcement** — an agent that edits outside its lane is rejected regardless of what it claims. With an injected **`verify_paths`** seam the boundary is checked against the **independent worktree diff**, not the agent's self-report, so an agent cannot under-report its edits to slip out of lane; a diff that can't be produced is itself a breach (fail-closed). |
| `review.py` | Review-feedback: accepted findings become **correction tasks** appended to the graph; a clean review advances to `INTEGRATE`. |
| `integrate.py` | CI / PR continuation: observe CI + the exact head it ran against in **one snapshot**, diagnose a failure into an `Observation`, and the **merge-authorization gate** whose risk is **derived from policy** - a fired obligation gates the merge if it is human-gated (self-modify / loop-controller / sealed / worker) *or* `blocking` and not candidate-satisfiable, so a **new** blocking obligation escalates by default (fail-closed). The merge is **head-bound** (`--match-head-commit`): a commit pushed since CI is never merged blind - it HOLDs to re-observe. |
| `docs.py` | Docs-freshness: a code↔doc coupling check - a change that touches coupled code but not its docs is `CONTRACT_DRIFT` at OBSERVE (never let docs go stale). |
| `completeness.py` | **L8 self-correction**: at VERIFY, a green gate is not "done" - the GOAL's success criteria must be MET, and each is **typed** (`CriterionKind`) so the *right evidence* is required: DETERMINISTIC / DOMAIN_AUTHORITY criteria count only if they cite a digest **bound to a sealed assurance record** (a bare `met=True` with no bound evidence becomes a correction task); MODEL_JUDGMENT is the model's opinion and can never *alone* close an autonomous finalize; HUMAN_APPROVAL is met only by a **recorded** `cs_loop authorize` grant. Routing: unmet gaps → CHANGES_REQUESTED (work them, via an executor-run correction task - unbounded correction work is never delegated to a boundary-enforced agent); a residual model-judged / human-approval gap → AUTHORIZATION_REQUIRED (escalate the human decision); a critic that errors escalates, never crashes the loop. Plus cross-goal **dead-end memory** (a ledger of exact failed-approach fingerprints, not generalized learning) wired into `run_loop` via `ctx.ledger_path` - it seeds prior goals' dead ends at the start and records this goal's at the end. |
| `orchestrate.py` | **The capstone** - one integrated loop that dispatches each phase to its module (decompose→validate, assign, execute/wave, observe+docs, review, integrate w/ HOLD-on-CI + merge gate, verify+completeness) with every effect injected. |
| `campaign.py` | **Multi-goal orchestration**: a queue of goals, each its own `run_loop`, sharing the learning ledger; dependency-ordered (a goal whose prerequisite did not finalize is skipped) and stop-on-blocker. Each goal's context comes from an injected **`context_for(goal)` factory** (the seam a runtime fills with a per-goal branch / worktree / PR / state) or, absent one, the shared context with a per-goal state file; a goal with an existing state file is **resumed**, not restarted. |
| `locking.py` | A portable stdlib **advisory file lock** (`O_CREAT\|O_EXCL` lockfile, bounded `timeout`, stale-lock breaking) guarding the cross-process read-modify-write of the shared learning ledger, so two concurrent campaigns cannot lose each other's appended entries. |

The interactive surface is **`scripts/cs_loop.py`** (`cs_loop init --goal … / next / status / observe`) for
the case where the LLM is the executor: `next` says what to do this phase; you do it; `observe` runs the
assurance gate and records the classified result; repeat until terminal.

## Runtime adapters

The controller is effect-free; a **runtime adapter** (`scripts/loop_adapters/`) is a
`build_context(repo_root, base) -> LoopContext` that binds the injected seams to real effects, so
`cs_loop run --adapters <file>` drives the loop against a live repo. Adapters are ordered by how much they
may *do*:

- **`dry_run.py` (read-only, propose-only).** Wires the real `cs_assure` read plane and read-only
  `git`/`gh` building blocks (`git_changed_paths`, `read_only_gh`), with a proposal-recording executor. It
  makes **no writes** — never pushes, merges, or spawns a write-capable agent — and ends at `ESCALATED`
  (a dry run proposes; a human signs off). A `pr_ref` additionally exercises the real CI read + merge gate
  but `dangerous=True` escalates before any merge. This is the safe way to see what the loop *would* do.

  The `pr_ref` path has been validated against a **live** open PR (read-only): `build_context(repo, pr_ref=…)`
  → `observe_ci` read the PR's real `statusCheckRollup` + `headRefOid` via `read_only_gh` (a real
  `gh pr view`, parsed to a head-bound `CiSnapshot`: CI green 10/10), the merge gate escalated (no
  autonomous merge), and a `gh pr merge` attempt was refused outright (exit 97). Real GitHub data, zero
  writes. (Env-dependent on `gh` auth, so it is a manual validation, not a committed test.)
- A **write-capable** adapter (real agent spawning + the autonomous merge path) is a later, review-gated
  step needing explicit human authorization; its seams (`verify_paths`, `expected_head`, `required_checks`,
  `context_for`) already exist, and adapter code is under the `loop-controller-self-modify` obligation.

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

CorpusStudio implements the **L4-L7 controller architecture** (executable single-agent loop → adaptive
recovery → coordinated multi-agent task graph → CI/review/integration autonomy) and **experimental L8
long-horizon mechanisms** (a completeness critic, cross-goal dead-end memory, and topologically-scheduled
multi-goal campaigns). It is **not** yet a production-complete L8 autonomous system: that still requires
concrete Claude-Code executor/reviewer/critic/agent adapters, isolated per-agent execution (git
worktrees), a concrete per-goal campaign-isolation runtime behind the `context_for` factory seam
(branch/worktree/PR), and race-safe head-bound GitHub integration. The controller is a control plane
*on top of* the assurance plane, not a replacement for it.

**Known scope limits (do not overstate):** completion is now **typed and evidence-bound** (see
`completeness.py`) - autonomous finalize needs DETERMINISTIC / DOMAIN_AUTHORITY criteria bound to sealed
records, and a bare model judgment or a pending human approval **escalates** rather than finalizing - but
the *evaluators are still injected*, so the strength of a DETERMINISTIC check is only as good as the
evidence the runtime seals into the assurance records. The cross-goal ledger is *exact dead-end memory*,
not generalized learning (its cross-process writes are now lock-guarded, see `locking.py`); multi-agent
boundary enforcement is worktree-derived when a runtime supplies `verify_paths` (else it falls back to the
agent's self-report, the weaker trust-based mode) and `dispatch_wave` runs runners sequentially; and
campaign isolation is a **seam** (`context_for`) with
per-goal state + resume - absent a factory, goals still share one repo/working-tree/PR, and no concrete
branch/worktree/PR factory is wired yet.

The **merge gate** derives risk from policy and is fail-closed, but two limits are honest to state: (1) the
merge-gate obligations are computed from `cs_assure impact` on the local tree - a production runtime must
pass `expected_head` (the commit it validated + pushed) so INTEGRATE refuses to merge a remote head it did
not analyze; without it single-writer is assumed. (2) `ci_observation` has no built-in notion of which
checks are *required* unless the runtime passes `required_checks`; GitHub **branch protection** (which
`gh pr merge --match-head-commit` respects server-side) remains the authoritative required-check backstop -
the loop does not replace it. A related **policy-coverage** gap is tracked separately: the assurance policy
gates `.github/workflows/assurance.yml` (assurance-self-modify) but not the other CI workflows, so a change
to `engine-tests.yml` fires no obligation - fixing that is an `obligations.json` (assurance-self-modify) PR,
not a loop change.

The loop controller (`scripts/loop/**`, `cs_loop.py`, loop tests, this doc) is under the
**`loop-controller-self-modify`** obligation, so a change to it cannot be admitted by the loop's own run -
it needs independent review.
