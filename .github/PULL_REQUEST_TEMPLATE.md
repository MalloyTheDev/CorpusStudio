<!--
Thanks for contributing to Corpus Studio! Keep the change a small, safe slice.
New to the codebase? See docs/DEVELOPER_GUIDE.md.
-->

## Summary

<!-- One or two sentences: what this changes and why. Link the issue: "Closes #123". -->

## What changed

<!-- The key files / behavior. Note new engine CLI commands, IEngineService seam
     methods, VM commands, or models added. -->

-

## Verification

<!-- What you actually ran (not "should pass"). Tick what applies. -->

- [ ] Engine gate green: `ruff` + `mypy corpus_studio` + `pytest -q` (from `engine/`, in the venv)
- [ ] Desktop builds (both heads): `dotnet build apps/desktop/CorpusStudio.Desktop.sln`
- [ ] Desktop tests: `dotnet test apps/desktop/CorpusStudio.Desktop.sln`
- [ ] New/changed behavior has tests
- [ ] Docs updated (if behavior or a command changed)
- [ ] Manual check noted below (for anything CI can't exercise — e.g. a live training/eval run)

<!-- Paste the relevant test counts / gate output. -->

## Honesty & scope

<!-- Corpus Studio is deliberately honest: a gate/suite PASS is a structure/threshold
     verdict, not proof of quality; keyword-overlap is a lexical proxy; provider policy
     keeps cloud models evaluator-only; PII redaction masks known patterns, not full
     de-identification. Confirm you preserved these where touched, and that the change
     is the smallest safe slice (no unrelated refactors / dependency churn). -->

- [ ] Preserves the honesty invariants where relevant
- [ ] Smallest safe change (no drive-by refactors or dependency churn)
