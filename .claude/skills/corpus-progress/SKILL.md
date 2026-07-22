---
name: corpus-progress
description: Manual "where are we / what's next" intake for CorpusStudio. Runs `cs_assure status` for a sealed, deterministic-where-possible facts snapshot, reads the authority docs it points at, forms a per-area where-are-we JUDGMENT, and recommends exactly ONE next slice -> hands to corpus-slice. It gathers no facts from memory (it cites the sealed record) and it neither executes nor authorizes work.
disable-model-invocation: true
---

# corpus-progress - situational intake for the assurance loop

`cs_assure gathers FACTS; you make the JUDGMENT.` This skill turns the sealed snapshot into one
grounded recommendation. Authority order on conflict: `AGENTS.md` > `docs/CURRENT_STATE.md` > this skill.

## The loop

1. **GATHER (facts, sealed).** Run `python scripts/cs_assure.py status --format json`. It exits 0 even
   if issues are unavailable or obligations fired (observation, not a gate); exit 2 is a fail-closed
   git/kernel refusal - STOP and report it, do not improvise a snapshot. Read the record; note its
   `record_digest` and cite it. If `issues.available == false`, carry `issues.reason` forward and make
   NO claim about issue state - you did not read any issues.
2. **READ (the authorities the snapshot pointed at).** Open the `authority_pointers` docs -
   `CURRENT_STATE` (feature state, wins on conflict), `ROADMAP` + `IMPLEMENTATION_PLAN` (forward plan),
   `PRODUCT_AREAS` (the seven-area identity), plus `HANDOFF` / `HOST_STATE` / `PRODUCT_VS_RESEARCH` only
   when the task is volatile or host/research-adjacent. This reading is exactly what the tool is
   forbidden to do; the tool handed you pointers, not summaries.
3. **ASSESS per area (JUDGMENT, not fact).** For each of the seven `product_areas`, form a
   where-are-we reading from: open issues + recent commits in the snapshot, doc staleness
   (`doclint.by_rule`), and `CURRENT_STATE`. Map the raw `issues.by_area` prefix tags to product areas
   yourself (loose, advisory: data->Data Studio; train/plan/objective/trace->Training Studio;
   eval/suite/judge->Evaluation Studio; behavior/lab/steer->Behavior Lab;
   model/artifact/release/export/card/quant->Model & Release Studio;
   env/host/hardware/storage->Environment & Hardware;
   research/paper/amendment/evidence/repro->Evidence & Experiments; and
   contracts/platform/cli/plugin/skill/docs/worker/ci/assurance -> cross-cutting, no single area).
   Label the whole reading MODEL_JUDGMENT.
4. **RECOMMEND one slice.** Output `{goal, target_area, rationale, authorization_class}` for exactly ONE
   next slice, with the rationale grounded in cited snapshot fields + doc lines. Then hand to
   `corpus-slice`, which owns branch -> one CI-green PR -> review. You recommend; you do not execute.

## Forbidden claims (honesty invariants)

- Do NOT fabricate progress. Every "done" / "works today" must cite `CURRENT_STATE` or a snapshot fact
  (a landed commit, a state you actually read). No evidence -> say "unknown / not verified".
- Label the per-area reading MODEL_JUDGMENT - never present it as a fact or as sealed.
- Do NOT claim any issue state (open / closed / count / area) you did not read from the snapshot. If
  `issues.available == false`, say issues were unavailable (with the reason) and assess without them.
- Do NOT claim "verified / fit / sealed / release-ready" from these FACTS alone - those need the gated
  workflows and their own evidence.
- A recommendation is NOT authorization. ENVIRONMENT / HARDWARE / RESEARCH / RELEASE / CREDENTIAL /
  DESTRUCTIVE slices still require the gated workflows + explicit go: STOP and surface, do not proceed.
- Do NOT re-derive the deterministic facts from memory - consume the sealed record and cite its
  `record_digest`. Do NOT edit any doc from this skill (that is corpus-slice's job, under its gate).
