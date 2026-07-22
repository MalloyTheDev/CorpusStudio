---
name: corpus-slice
description: Manual bounded-implementation loop for a single CorpusStudio change - branch first, a typed slice contract, then act -> observe -> verify against the real gate + cs_assure + the assurance-reviewer, with objective stop/escalation rules and a completion record. Invoke it deliberately for a feature or fix worth a PR; it never weakens a gate to make a change pass.
disable-model-invocation: true
---

# corpus-slice - bounded implementation loop

Run ONE coherent CorpusStudio change as an *engineered* loop: a typed slice contract, act -> observe ->
verify against the real authorities, objective stop/escalation rules, and a completion record. This is
the default unit of work. The bar is not "it works" - it is "the evidence, contracts, and honesty
invariants still hold."

Authority order on conflict: `AGENTS.md` (the contract) > `docs/CURRENT_STATE.md` (feature state) >
this skill. Use `cs_assure changeset` for what-changed facts and the `assurance-reviewer` agent for
advisory review - do not narrate those from memory.

## Phases

`INTAKE -> CONTEXT_PLAN -> SLICE_CONTRACT -> [ ACT -> OBSERVE -> UPDATE_HYPOTHESIS ]* -> (REPLAN | ESCALATE | FINALIZE | ABORT)`

1. **INTAKE.** State the goal, the non-goals, and the intent (edit / study / verify). Classify the
   mutation class: `SOURCE` / `GENERATED_OUTPUT` / `LOCAL_STATE` are ordinary; `ENVIRONMENT` /
   `HARDWARE` / `RESEARCH` / `RELEASE` / `CREDENTIAL` / `DESTRUCTIVE` need the gated workflows and
   explicit authorization - STOP and surface if the task has not authorized them.
2. **CONTEXT_PLAN.** Load only the authorities the task activates (a doc typo needs none of the host /
   GPU / research context). The path-scoped `.claude/rules/*` fire when you read the matching files -
   follow them (contracts regen, worker-closure classification, sealed-research stops, evaluation
   honesty, assurance self-modification).
3. **SLICE_CONTRACT.** `git checkout -b feat/<slice>` (or `fix/` / `research/`) - branch first. Write
   the typed session state (below) with `planned_paths` and `forbidden_operations` declared *before*
   editing.
4. **ACT -> OBSERVE -> UPDATE_HYPOTHESIS.** Make the smallest coherent edit; observe the result (gate
   output, `cs_assure changeset`); update `current_hypothesis`. Repeat until FINALIZE or a stop rule.
5. **FINALIZE / REPLAN / ESCALATE / ABORT** per the stop rules.

## Typed session state (operational loop - NOT evidence)

Keep a small, bounded state object. Store it at the worktree-safe git path - never committed, no
secrets, bounded size, cleaned up when done:

`git rev-parse --git-path corpusstudio-assurance/sessions/<session-id>/slice.json`

```json
{
  "schema_version": 1, "session_id": "...", "phase": "EDITING",
  "goal": "...", "non_goals": [], "planned_paths": [], "allowed_mutation_classes": [],
  "forbidden_operations": [], "current_hypothesis": "...", "last_observation": "...",
  "attempts_for_current_failure": 0, "unexpected_impacts": [], "stop_reason": null
}
```

This is the mutable working loop, not an assurance record. Do not commit a generic `STATE.md` as
evidence; the durable record is the change set + the completion record below.

## Verification (before you claim the slice is done)

- **The engine gate, green**, from `engine/`: `.venv/bin/python -m ruff check corpus_studio tests`;
  `.venv/bin/python -m mypy corpus_studio`; `.venv/bin/python -m pytest -q --no-header
  --basetemp=.pytest_tmp` (CI adds `--cov=corpus_studio`, floor 88%). Never report done on red - if
  tests fail, say so with the output.
- If you touched `platform/contracts.py`: regenerate the schemas + TS and update the two counts (the
  contracts rule).
- `cs_assure changeset --scope workspace --base main` - confirm the change set matches `planned_paths`
  (unexpected paths = ownership creep, a stop condition). `cs_assure doclint` if you touched docs.
- Optionally run the `assurance-reviewer` for an advisory pass - a `MODEL_REVIEW`, which is NOT human
  review or CI.

## Stop and escalate when

- the same failure occurs three times without a materially different hypothesis;
- the diff expands outside `planned_paths`;
- an unexpected `WORKER` / `RESEARCH` / `ASSURANCE_SELF` / `GPU` / `CREDENTIAL` / `DESTRUCTIVE` /
  `RELEASE` / `LEGAL` impact appears;
- a canonical authority is unavailable, or verification mutates tracked state unexpectedly;
- a required human judgement cannot be replaced by deterministic evidence.

Do not conceal a blocker by weakening the gate - surface it.

## Completion record (output)

Report: `branch/base`; changed paths (from `cs_assure`); affected product areas; derived outputs
regenerated (or `NOT_APPLICABLE`); verification (actual commands + results); review (MODEL_REVIEW done?
human review still required); supported claims; unsupported claims; remaining uncertainty; hard blockers.

## Forbidden claims

Do not claim "verified" / "fit" / "release-ready" / "sealed" from a green gate alone. A completed step
is not proven fit; a green workspace gate is not commit / PR / release / sealed-research readiness; a
`MODEL_REVIEW` is not human review or CI. Never report a guessed test count or a future result.
