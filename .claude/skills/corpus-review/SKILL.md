---
name: corpus-review
description: Manual advisory review of the current CorpusStudio change - runs the read-only assurance-reviewer against the honesty invariants and execution boundaries, reports structured findings, and states plainly that a model review is not human review, domain-expert review, or CI. Invoke it to sanity-check a slice before claiming it done.
disable-model-invocation: true
---

# corpus-review - advisory review

Run a READ-ONLY advisory review of the current change before you call a slice done. This is defence in
depth, not a gate: it never edits, merges, installs, or runs CI, and it never weakens a gate to clear a
finding.

## How

1. Get the change facts deterministically: `cs_assure changeset --scope workspace --base main` (the
   changed paths + the sealed record); `cs_assure doclint` if docs were touched.
2. Hand the change-set record + the changed paths to the **`assurance-reviewer`** agent (restricted to
   Read / Grep / Glob). It reviews against the honesty invariants, the dependency-light / no-shell /
   contracts boundaries, the worker-execution closure (classify by tracing the import path), the
   product / sealed-research boundary, and assurance self-modification. Use a SINGLE reviewer - never a
   fan-out.
3. Read the findings; decide REPLAN (fix) or proceed. A finding is not cleared by weakening a gate.

## Review kinds (keep distinct - a gate decides which an obligation needs)

- **MODEL_REVIEW** - this skill / the `assurance-reviewer`. Advisory only.
- **HUMAN_REPOSITORY_REVIEW** - a human reviews the PR.
- **DOMAIN_EXPERT_REVIEW** - a domain owner (e.g. the research authority for sealed evidence).
- **PROTECTED_CI_POLICY_REVIEW** - the repo's CI gate policy (an authenticated GitHub setting, not
  visible in the committed tree; do not assert it from files).

A MODEL_REVIEW never substitutes for the other three.

## Output

Structured findings, most-severe first (`severity | file:line | invariant | what | why`), then the
honest scope line, verbatim:

> This is a MODEL_REVIEW. It is advisory and does NOT count as HUMAN_REPOSITORY_REVIEW,
> DOMAIN_EXPERT_REVIEW, or PROTECTED_CI_POLICY_REVIEW, and no gate or CI was run.
