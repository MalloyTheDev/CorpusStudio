---
paths:
  - "scripts/loop/**"
  - "scripts/loop_adapters/**"
  - "scripts/cs_loop.py"
  - "engine/tests/test_loop_*.py"
  - "engine/tests/test_cs_loop.py"
  - "docs/AUTONOMOUS_LOOP.md"
---

# Changing the autonomous loop controller needs independent review

A change under `scripts/loop/`, to a **runtime adapter** in `scripts/loop_adapters/`, to `scripts/cs_loop.py`,
to a loop test, or to `docs/AUTONOMOUS_LOOP.md` edits the **autonomous loop controller or the runtime that
fills its effects** - the thing that decides retry / stop / escalate / hold / merge / finalize and the
adapter that binds those decisions to real `git` / `gh` / agent effects. An adapter is exactly where a
write-capable or autonomous-merge effect would be wired, so it carries the same review requirement as the
controller. The loop's own run is therefore **not** independent verification of a change to it, exactly as
the assurance system cannot vouch for changes to itself (see [`assurance-self-modify`](assurance-self-modify.md)).

- **The loop cannot admit changes to itself.** A loop-controller change must be admitted by trusted-base
  controller tests, CI, and INDEPENDENT human review - never by the loop's own completeness/verify pass
  or its merge gate.
- **No autonomous merge, no self-approval.** A loop-controller change never merges via the loop's own
  `integrate` merge gate, and the candidate does not self-merge it even with admin rights; an independent
  reviewer merges it.
- **Preserve the controller's contracts.** `scripts/loop/` stays stdlib-only (imports nothing from
  `corpus_studio`, no torch), FAIL-CLOSED (an unrecognised observation escalates; a degenerate budget
  stops; a refusal never crashes the loop), and INJECTED-EFFECT (executor / reviewer / agent runner / gh
  / cs_assure / critic are callbacks - no real effect is hardcoded, so the loop stays testable and its
  honesty boundary reaches the merge button). Add deterministic tests under `engine/tests/`.
- **Adapters are the effect-wiring layer, gated by capability.** `scripts/loop_adapters/` is where real
  effects ARE bound (that is its job, so the effect-free rule does not apply there), but it stays
  stdlib-only and is introduced least-capable-first: a read-only / propose-only adapter makes no writes; a
  write-capable or autonomous-merge adapter is a later step that needs explicit human authorization, and
  either way an adapter change is admitted only by independent review, never the loop's own merge gate.
