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
  actions, synthetic issue handoff, and preference judge preparation.
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

1. Persist prepared AI Assist rewrite batches so they survive app restart.
2. Add target-specific Training config compatibility warnings.
3. Add dataset card export from project metadata, schema, splits, quality history, and evaluation summary.
4. Add versioned reviewed-fix tracking or explicit in-place replacement for edited failed rows.
5. Add interactive Evaluation drilldowns and saved failure filters.
6. Keep full training launch, logs, checkpoints, and resume support out of the core app until the Evaluation workflow is stable.

Read these files first:

- README.md
- docs/PRODUCT_SPEC.md
- docs/ARCHITECTURE.md
- docs/SCHEMA_SYSTEM.md
- docs/ROADMAP.md
- docs/QUALITY_GATES.md
