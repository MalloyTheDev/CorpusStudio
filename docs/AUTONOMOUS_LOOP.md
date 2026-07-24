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
| `store.py` | Durable `LoopState`: deterministic **durable** atomic JSON persistence (fsync'd temp → `os.replace` → dir fsync, so a power-loss crash cannot torn-write *or* silently roll back the last transition), fail-closed load, crash-resumable. `state_revision` distinguishes an **absent** file (a normal first write → 0) from a present-but-**corrupt** one (`CorruptStateFile`), so a CAS writer never silently overwrites corruption as revision 0. For a **concurrent** (multi-process) writer, `save_cas` adds optimistic concurrency — under a file lock it verifies the on-disk `state_revision` matches what the writer loaded (else `ConcurrentStateWrite`), writing `revision+1` with a `state_digest` / `previous_state_digest` / `writer_id`; the single-writer `save` is unchanged. The lock (`locking.py`) breaks a crashed holder by **PID-liveness** (a live holder — even a long-running one past `stale_after` — is never broken; only a dead PID is, promptly) and breaks **atomically** (rename-to-claim), so two waiters can never both break a lock and the second delete a lock a third just re-created. **Single-writer is now ENFORCED, not assumed**: `run_loop` holds this lock for the whole run, and every mutating `cs_loop` command (`observe` / `pause` / `resume` / `abort` / `authorize` / `init`) holds it for its read-modify-write — so a second concurrent writer on the same state file **fails closed** (`run_loop` escalates; a command exits 2) instead of silently clobbering the other's update. PID-liveness is what makes holding the lock across a long run safe. |
| `observe.py` | Reads the repo via `cs_assure verify` + `doclint` and maps the sealed result to one **mechanical** `Observation` (gate step → failure class; obligations → human-gated / worker / drift). The human-gated obligation set is the **shared** `controller.HUMAN_GATED_OBLIGATIONS` (same object the merge gate uses), so a change firing `loop-controller-self-modify` escalates at OBSERVE too - the OBSERVE plane and the merge gate can never drift about what needs a human (`worker-closure` keeps its distinct `WORKER_LINEAGE_IMPACT` label). The verify result is a **validated record**, not raw JSON: a refusal exit (≥2), a wrong `record_type`, an unsupported schema (< v2), a missing change-set fingerprint, or `workspace_stable != True` (the tree mutated during the gate) each fails closed; a non-`True` `gate_passed` never records green evidence; an unavailable `doclint` is surfaced explicitly, never silently read as clean. |
| `driver.py` | The runnable cycle: emit a per-phase directive, run the executor (or the observer at the verify phases), route, persist. `run()` drives to a terminal phase or a hard step cap. |
| `tasks.py` | The task graph: owned tasks with dependencies and an `allowed_paths` **ownership boundary** (a declared lane must be a real subpath — absolute / `..` / and the whole-repo `.` are rejected; a repo-wide task uses an empty lane and runs self-owned); ready/blocked derivation; parallel-safe assignment. |
| `router.py` | The agent router: a parallel-safe wave (≤10 agents), dispatch, and **boundary enforcement** — an agent that edits outside its lane is rejected regardless of what it claims. With an injected **`verify_paths`** seam the boundary is checked against the **independent worktree diff**, not the agent's self-report, so an agent cannot under-report its edits to slip out of lane; a diff that can't be produced is itself a breach (fail-closed). |
| `review.py` | Review-feedback: accepted findings become **correction tasks** appended to the graph; a clean review advances to `INTEGRATE`. |
| `integrate.py` | CI / PR continuation: observe CI + the exact head it ran against in **one snapshot**, diagnose a failure into an `Observation`, and the **merge-authorization gate** whose risk is **derived from policy AND evidence** - never obligation identity. A fired obligation gates the merge if it is human-gated (self-modify / loop-controller / sealed / worker); a low-severity (info/advisory) one does not gate; **any other blocking obligation is satisfied ONLY by a trusted, current `ObligationResolutionRecord`** - an injected dict proving that obligation was `RESOLVED` for **this** change set (`subject_fingerprint` == the impact change-set fingerprint) by a **trusted authority** (never the candidate). Absent such a record it escalates, so `contracts` / `evaluation-honesty` are no longer auto-mergeable on identity ("usually CI-satisfiable" is never equated with "satisfied for this commit"), and a **new** blocking obligation still escalates by default (fail-closed). The gate emits a **`GateEvaluation`** (per-obligation verdict + overall) persisted to `review_state["gate_evaluations"]`. The merge is **head-bound** (`--match-head-commit`): a commit pushed since CI is never merged blind - it HOLDs to re-observe. A **candidate-only** policy assessment (`base_policy_available == false` — the trusted merge-base policy could not be loaded) escalates rather than authorizing, since the candidate could have weakened the policy unseen. |
| `docs.py` | Docs-freshness: a code↔doc coupling check - a change that touches coupled code but not its docs is `CONTRACT_DRIFT` at OBSERVE (never let docs go stale). The change set it reads **fails closed**: a `cs_assure changeset` that cannot be produced (non-zero exit / malformed record) is never read as "nothing changed" - OBSERVE escalates rather than advancing past a possibly-stale coupled doc (the check runs only on an otherwise-green gate). |
| `completeness.py` | **L8 self-correction**: at VERIFY, a green gate is not "done" - the GOAL's success criteria must be MET (and completion is **mandatory**: with NO completeness evaluator the loop escalates rather than finalizing on the gate alone - a green gate is never the implicit definition of done). Each criterion is **typed** (`CriterionKind`) so the *right evidence* is required: DETERMINISTIC / DOMAIN_AUTHORITY criteria count only if they are **backed by real evidence** (a bare `met=True` with no bound evidence becomes a correction task). A criterion may state *what* evidence must back it (`required_record_type` / `required_predicate`, optionally pinned to a `subject_fingerprint`); it is then satisfied only by a matching entry in the structured evidence index (`state.review_state["evidence"]`, which the loop populates on a green gate: `record_type=workspace_verification`, `predicate=WORKSPACE_GATE_GREEN`, `subject_fingerprint=<change-set>`) - so a workspace-verification digest **cannot** stand in for an unrelated claim (e.g. a "docs complete" criterion is not closed by a green-gate record of the wrong type/predicate). A criterion that names **no** semantic requirement falls back to the digest-membership check against the sealed assurance records (backward-compatible). MODEL_JUDGMENT is the model's opinion and can never *alone* close an autonomous finalize; HUMAN_APPROVAL is met only by a **recorded** `cs_loop authorize` grant. The verdict is computed against a **snapshot of the control-plane fields (authorizations / evidence index / assurance records) taken BEFORE the untrusted critic runs**, so a critic cannot self-grant or self-fabricate evidence during its own call; `step()` additionally restores `review_state["authorizations"]` after every dispatch (the loop never writes grants in-process), so no injected callback can persist a fabricated grant across a step. Routing: unmet gaps → CHANGES_REQUESTED (work them, via an executor-run correction task - unbounded correction work is never delegated to a boundary-enforced agent); a residual model-judged / human-approval gap → AUTHORIZATION_REQUIRED (escalate the human decision); a critic that errors escalates, never crashes the loop. Plus cross-goal **dead-end memory** (a ledger of exact failed-approach fingerprints, not generalized learning) wired into `run_loop` via `ctx.ledger_path` - it seeds prior goals' dead ends at the start and records this goal's at the end. |
| `orchestrate.py` | **The capstone** - one integrated loop that dispatches each phase to its module (decompose→validate, assign, execute/wave, observe+docs, review, integrate w/ HOLD-on-CI + merge gate, verify+completeness) with every effect injected. `step()` is **fail-closed on ANY raise**: an injected effect that throws - a reviewer raising `ReviewError`, an executor/agent raising anything, a `LoopTaskError`, a subprocess `OSError` - is caught and the loop lands **durably at `ESCALATED`** (persisted), so a raising effect never crashes the loop mid-run and one goal's raise never kills a whole campaign. The multi-agent EXECUTE branch is symmetric with single-agent: a self-owned task completing re-enters (`CHANGES_REQUESTED`) while any task is still `PENDING`, so it never advances past EXECUTE with work left undone. |
| `campaign.py` | **Multi-goal orchestration**: a queue of goals, each its own `run_loop`, sharing the learning ledger; dependency-ordered (a goal whose prerequisite did not finalize is skipped) and stop-on-blocker. Each outcome carries a `status` (FINALIZED / ESCALATED / STOPPED / **HELD** / SKIPPED); a **HELD** goal (waiting on an external condition like CI) is **not** a failure — it pauses the campaign so a re-run resumes it, rather than skipping its dependents as failed. Each goal's context comes from an injected **`context_for(goal)` factory** (the seam a runtime fills with a per-goal branch / worktree / PR / state) or, absent one, the shared context with a per-goal state file; a goal with an existing state file is **resumed**, not restarted. |
| `locking.py` | A portable stdlib **advisory file lock** (`O_CREAT\|O_EXCL` lockfile, bounded `timeout`, stale-lock breaking, **per-acquisition owner token**) guarding the cross-process read-modify-write of the shared learning ledger + the CAS state write, so two concurrent campaigns cannot lose each other's appended entries — and a process whose stale lock was broken and re-created never deletes the new owner's lock on its late release. |

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
- **`single_agent.py` (Phase 7.0: real agent, read/propose-only).** The first adapter that wires a **real
  Claude-Code agent** into the loop - but strictly propose-only. At EXECUTE it asks the agent (through an
  injected `AgentClient`, whose real transport is an out-of-process fixed-argv `claude` subprocess with a
  framed JSON contract) to PROPOSE a unified diff; the untrusted response is validated fail-closed and
  sealed as a tamper-evident `agent_proposal` record written **outside** the working tree; nothing is ever
  applied. The agent is **confined (7.1.1)**: it runs with cwd inside a **disposable, detached worktree**
  at `base` (never the developer's tree), a **sanitized (secret-free) environment**, a version-pinned
  **read-only tool policy** (`Read,Grep,Glob`; edit/write/bash/nested-agents/net denied), and **bounded
  output** - so a mis-behaving agent cannot edit the working tree even while "just proposing". It declares
  `capabilities=frozenset()` (read-only, so the capability gate runs it with no opt-in), makes **no
  writes**, and ends `ESCALATED` (a human decides whether to apply the proposal). See
  [`docs/PRODUCTION_SINGLE_AGENT_RUNTIME.md`](PRODUCTION_SINGLE_AGENT_RUNTIME.md).
- **`single_agent_write.py` (Phase 7.1: write-capable single agent, GATED).** The agent still PROPOSES a
  sealed diff (7.0 behaviour, run under the same 7.1.1 confinement); 7.1 then APPLIES that exact diff in a
  **separate, pristine, disposable `git worktree`** (never the developer's tree, never the confined propose
  checkout), verifies the applied change matches the sealed proposal, commits it on a fresh branch, pushes,
  and opens a PR. It declares **`capabilities={"write"}`** (so `cs_loop run` REFUSES it without
  `--allow-capabilities write`) and **never merges**: `write_gh` allows `pr create` + reads but refuses
  `pr merge` (and every other mutation), and `dangerous=True` escalates the merge gate - a human reviews +
  merges the PR. Any failure (a diff that won't apply, a drifted apply, a failed PR-create) fails closed
  and both worktrees are disposed; the main tree is left pristine. **Not yet production-safe:** candidate
  assurance, exact candidate identity, crash recovery, draft PRs, and sensitive-path denial are deferred
  hardening (7.1.2 - 7.1.5) that must land before 7.1 runs against a real repository.
- The **autonomous merge** path (Phase 7.2) - evidence-bound `merge_gate` + the obligation-resolution
  producer - remains future, review-gated, and needs its own explicit authorization; its seams
  (`expected_head`, `required_checks`, the head-scope impact) already exist. All adapter code is under the
  `loop-controller-self-modify` obligation. See [`docs/PRODUCTION_SINGLE_AGENT_RUNTIME.md`](PRODUCTION_SINGLE_AGENT_RUNTIME.md).
- **Capability gate (machine-checkable).** An adapter's `LoopContext` DECLARES its effect
  `capabilities` (empty = read-only / propose-only, the default). `cs_loop run` / `campaign` **refuse**
  (exit 2, fail-closed) to run a context that declares a capability the operator did not permit via
  `--allow-capabilities` — so a write-capable adapter cannot be loaded and empowered silently; it is the
  boundary the #7 write-runtime is gated behind (it complements, never replaces, the merge gate).
  Additionally, a **write-capable + multi-agent** context with no `verify_paths` is refused at
  construction (a delegated wave that can write must not fall back to agent self-report).
- **Exit-code taxonomy** (so automation reads the outcome without parsing stdout): `cs_loop run` →
  `0` FINALIZE, `3` HELD (paused on CI), `4` ESCALATED, `5` STOPPED; `campaign` → `6` when not every goal
  finalized; `2` is a fail-closed refusal throughout.

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
