# Handoff Prompt for Claude/Codex

You are working inside the Corpus Studio repository.

Corpus Studio is a local-first dataset creation studio for AI builders. It is MIT-licensed.

Your job is to continue hardening the existing working application while
preserving the architecture. Do not restart the project from scratch.

## Product goal

Build a one-stop shop for creating AI training datasets across multiple schemas.

The app should eventually support:

- raw pretraining corpora
- instruction tuning
- chat/message datasets
- preference/DPO datasets
- code datasets
- image-caption datasets
- classification datasets
- retrieval/embedding datasets
- evaluation datasets

## Current Scope

The working local dataset loop covers:

- raw text
- instruction
- chat/messages
- preference

Core v0.1 flow:

```text
Create project
-> choose schema
-> author/import examples
-> validate
-> inspect quality
-> split train/validation/test
-> export JSONL
```

The repository also has first-pass Evaluation Lab, AI Assist Lab, and Training
Lab config export surfaces. They should be hardened in place:

- Evaluation Lab: Ollama/OpenAI-compatible local runs, backend health checks,
  model discovery, saved reports, two-report comparison, saved regression
  reruns from report `run_settings`, failed-example review, manual score/notes,
  report summaries by tag/failure reason/score band, failed-row edit handoff to
  Writing Studio, and AI Assist triage preparation.
- AI Assist Lab: review-first local model suggestions, persistent queue,
  filters, search, sorting, saved views, bulk triage/undo, schema-aware
  actions, synthetic issue handoff, persistent rewrite batches, and preference
  judge preparation.
- Training Lab: inspectable config export only. Do not implement a trainer
  launcher yet.

## Architecture constraints

- Keep the app local-first.
- Keep schemas first-class.
- Use the Python engine for dataset logic.
- Use the C# desktop app for UX.
- Do not hardcode every dataset type directly into UI behavior.
- Do not silently mutate user data.
- Exports must be deterministic.
- Validation errors must be specific and actionable.

## Next Implementation Targets

The previous top-five board is complete: versioned reviewed-fix tracking,
interactive Evaluation drilldowns with saved failure filters, an optional
SQLite-backed project index, opt-in Ollama integration tests, and a public-repo
release checklist. Desktop-side unit tests now cover the project-local JSON
persistence (reviewed fixes, rewrite batches, saved failure filters) and run in
CI on Windows via `.github/workflows/desktop-tests.yml`. Next candidates:

1. Wire the desktop project list to the optional SQLite index for faster
   load/filter on large project sets (`storage/index.py`, `project-list` CLI).
2. Extend desktop test coverage to AI Assist queue views and the ViewModel
   filter/reconcile logic.
3. Add production-grade synthetic pattern clustering in the AI Assist Lab.
4. Add target-specific DPO/reward-model export formats.
5. Keep full training launch, logs, checkpoints, and resume support out of the
   core app until the Evaluation workflow is stable.

Read these files first:

- README.md
- docs/PRODUCT_SPEC.md
- docs/ARCHITECTURE.md
- docs/SCHEMA_SYSTEM.md
- docs/ROADMAP.md
- docs/QUALITY_GATES.md
