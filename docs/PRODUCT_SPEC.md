# Corpus Studio Product Specification

## Product identity

Corpus Studio is a local-first dataset creation studio for AI builders.

It combines:

- a writing application
- a schema-driven dataset editor
- a validation engine
- a cleaning lab
- a quality dashboard
- a split/export manager
- an Evaluation Lab MVP
- an AI Assist Lab MVP
- a Training Lab config-export MVP

The app exists to make training-data creation less fragile, less manual, and less scattered.

## Target users

Primary users:

- independent AI toolmakers
- model fine-tuners
- game developers building AI-assisted tools
- researchers preparing local datasets
- small teams building domain-specific assistants
- people creating code, image-caption, chat, or preference datasets

## Core problem

Dataset creation is usually fragmented:

```text
notes -> scripts -> spreadsheets -> JSONL -> training configs -> fixes by hand
```

This creates broken rows, inconsistent schemas, duplicate examples, data leakage, poor provenance, and painful exports.

## Product promise

Corpus Studio lets a user move from idea to model-ready dataset inside one local application.

Long term, Corpus Studio should support the full dataset-to-model path:

```text
create dataset
-> validate and split
-> test with models
-> improve weak examples
-> export clean dataset
-> generate training config
-> launch local adapter training
-> compare checkpoints with evaluation runs
```

## Current Non-Goals

- training models directly
- cloud collaboration
- automatic scraping
- bulk synthetic generation
- production-grade PII detection
- Hugging Face publishing
- PDF OCR
- advanced multimodal annotation
- full Evaluation Lab model comparison and regression workflow
- automatic AI Assist acceptance and production-grade synthetic-pattern analysis
- full target-specific training compatibility automation
- training launchers

## v0.1 supported dataset types

1. raw text
2. instruction
3. chat/messages
4. preference pairs

These are available in the desktop project creation flow and in the Python engine CLI.

## Future dataset types

5. code
6. image-caption
7. classification
8. retrieval/embedding
9. evaluation

## Core workflow

```text
Create project
-> choose schema
-> author examples
-> validate
-> export
```

## Current working loop

The current app proves the smallest local dataset-authoring loop:

1. Launch the WPF desktop app.
2. Create a local project under `data/projects`.
3. Choose a built-in schema from the engine.
4. Edit the generated JSON example in Writing Studio.
5. Validate the draft through the Python engine and select issues to focus the draft editor.
6. Save the example to the active project's `examples.jsonl`.
7. Preview a JSONL import against the active schema with failed-row reporting.
8. Import valid JSONL rows into the active project's `examples.jsonl` and quarantine rejected rows when requested.
9. Review quarantined import rows and retry a selected raw row in Writing Studio.
10. Run quality checks against the active project's saved examples and record project-level quality history.
11. Generate train/validation/test split files under `exports/<project_id>/splits` with saved settings and warnings for tiny validation/test outputs.
12. Inspect saved example details from the Examples tab.
13. Reopen an existing project from the project list.
14. Run a first-pass Evaluation Lab sample against a configured local model backend.
15. Check configured model backend health and reload saved Evaluation reports.
16. Compare two saved Evaluation reports for score, failure, weak-tag, and row-level deltas.
17. Rerun a saved Evaluation configuration as a regression check.
18. Add manual per-example Evaluation scores and notes to saved reports.
19. Run a first-pass AI Assist review against the current draft.
20. Review queued AI Assist results, mark them accepted or rejected, and move accepted suggestions into Writing Studio for human validation.
21. Export validated JSONL to `exports/<project_id>/export.jsonl`, with an optional cleaning pass (Export Center checkboxes, or the `--dedupe` / `--drop-low-information` CLI flags) that drops exact/normalized-duplicate and low-information rows and writes a `*.cleaning_manifest.json` of exactly what was removed; a verbatim export still warns when duplicate rows remain, and the desktop reports how many rows cleaning removed.
22. Inspect local repository, engine, Python, project, and export paths from Settings.

The Python engine also exposes schema listing, validation, project creation,
import preview, quality reporting, splitting, backend health checks, model
discovery, Evaluation Lab runs, review-first AI Assist runs, Training Lab
config export, and JSONL export commands for developer workflows. There is
still no trainer launcher.

## Current Constraints

- Project data is file-backed JSON and JSONL.
- The desktop app writes one JSON object per saved or imported example.
- Validation currently enforces JSON object rows, required non-empty fields, declared field types, chat message structure, list element types (`item_type`, e.g. `tags` must be a list of strings), fixed value sets (`enum`) for scalar fields, inclusive numeric bounds (`minimum`/`maximum`) for integer/float fields, and nested `object` field shapes (recursive sub-field validation with dotted error paths).
- JSONL import previews all rows; mixed files can import valid rows only after explicit confirmation, with rejected rows written to project-local quarantine and reviewable from the desktop app.
- Quality checks currently report example count, empty rows, exact duplicates, normalized duplicates, low-information rows, and dataset-wide synthetic-pattern warnings with severity and repair suggestions; near-duplicate synthetic patterns of the same kind are additionally grouped into token-similarity clusters so related boilerplate families surface as one finding. A first-pass PII / secret scan flags likely emails, SSNs, private-key blocks, AWS/API keys, JWTs, and Luhn-valid card numbers (with masked samples) so secrets are caught before they ship into training data. Token-length outliers (IQR-based, using a Unicode-aware token estimate) and category-imbalance warnings (a low-cardinality field dominated by one value) round out the dataset-health signals. All text quality signals use Unicode-aware (NFKC) tokenization so near-duplicate, low-information, and synthetic-pattern detection work for non-Latin scripts (CJK, Cyrillic, accented Latin), and the engine forces UTF-8 stdio so non-ASCII payloads round-trip to the desktop instead of corrupting on the Windows console/pipe code page. Snapshots are appended to `quality_history.jsonl`; the desktop can hand a selected synthetic issue or the affected issue rows to AI Assist as prepared rewrite workflows, and prepared batch rewrites persist for restart resume.
- Split generation supports project-persisted train percentage, validation percentage, deterministic seed, and warnings for empty or one-row validation/test outputs. The test split uses the remaining percentage. After splitting it runs non-destructive train/test leakage detection: exact and near-duplicate (NFKC/Unicode-normalized) rows shared across splits are reported as `rows_shared_across_splits` plus a `leakage` breakdown and a warning, so contamination that would inflate evaluation scores is surfaced before training.
- Evaluation Lab model execution exists as an engine CLI and desktop MVP for explicit local endpoint runs. Backend health checks now exist as an engine CLI and desktop button, model discovery can populate local model pickers, project-local `lab_settings` persist backend choices, evaluation reports store repeatable `run_settings` plus tag/failure-reason/score-band summaries, evaluation runs and regression reruns perform a pre-run health gate, saved reports reload from the desktop history list, two saved reports can be compared for score/failure/tag/example deltas, failed examples can be filtered for review, failed rows can be loaded back into Writing Studio for explicit edit/validate/save/rerun loops, failed rows can be prepared for AI Assist triage, manual per-example scores/notes persist back to report JSON, reviewed fixes for edited failed examples are versioned per example and auto-reconciled to resolved/still-failing on re-test, and named failure filters by status/tag/failure-reason/score-band persist per project in `evaluation_failure_filters.json`; multi-model benchmark comparison remains planned.
- AI Assist exists as an engine CLI and desktop MVP for review-first draft suggestions; project-local review queues with accept/reject states, review-state filters, search, sorting, saved queue views, persistent rewrite batches, bulk triage with multi-step undo, schema-aware action presets, local model discovery, project-local `lab_settings`, side-by-side source/suggestion comparison, synthetic issue triage-to-rewrite and batch rewrite handoff, preference-pair ranking/review with AI Assist judge handoff, preference ranking export, visible queue batch judge preparation, and dataset-wide synthetic/preference warnings with near-duplicate synthetic-pattern clustering exist.
- Training config export exists as an engine CLI and desktop MVP that writes
  inspectable config files only, and now reports a token budget for the dataset
  (total/mean/max tokens, tokens-per-epoch after `sequence_len` truncation, and
  how many examples exceed `sequence_len`) using the Unicode-aware token
  estimator; local trainer launch, logs, checkpoints, and resume support remain
  planned.
- An optional SQLite-backed project index is available (`project-list` and `project-index-rebuild` CLI, opt-in via `CORPUS_STUDIO_USE_INDEX`); JSON/JSONL remain the authoritative, inspectable project state.
- Desktop project-file writes are atomic (temp file + replace) so a crash mid-write cannot truncate or corrupt live project state, and a running engine command can be cancelled from the busy overlay, which kills the engine process tree. Long local evaluation/AI Assist runs are intentionally left without a hard timeout so they are not cut off mid-run.

## Product principles

1. Local-first by default.
2. User owns their data.
3. Dataset examples are first-class objects.
4. Schemas drive the editor.
5. Validation must be explicit.
6. Cleaning should be reversible or auditable.
7. Export formats must be deterministic.
8. Evaluation datasets are as important as training datasets.

## Hardening Status

The v0.1 hardening loop now includes import quarantine recovery, quality
history, split setting persistence, tiny-split warnings, first Evaluation Lab
and AI Assist desktop surfaces, and first-pass Training Lab config export
without trainer dependencies.

## Staged Lab Roadmap

- v0.2: Evaluation Lab desktop workflow with Ollama and OpenAI-compatible local endpoints; first-pass run UI, backend checks, pre-run health gates, report history, two-report comparison, saved regression reruns, report summaries by tag/failure reason/score band, failed-example filtering, failed-row edit handoff to Writing Studio, failed-example AI Assist triage preparation, and manual score/notes persistence exist.
- v0.3: AI Assist Lab for reviewed tagging, rewriting, weak-example detection,
  and draft generation; first-pass review UI and persistent accept/reject queue
  plus review-state filters, search, sorting, bulk triage with multi-step undo,
  schema-aware action presets, side-by-side source/suggestion comparison, and
  first-pass synthetic/preference warnings plus synthetic issue
  triage-to-rewrite and batch rewrite handoff with persistent resume batches,
  preference-pair ranking/judge handoff, preference ranking export, and visible batch judge preparation
  exist, along with near-duplicate synthetic-pattern clustering and
  target-specific DPO/KTO/reward preference exports with a preference-integrity
  gate (counts of empty, identical, and low-contrast pairs, plus an opt-in
  `--drop-degenerate` that excludes unusable pairs before export).
- v0.4: Training config generation for Axolotl, TRL, Unsloth, Hugging Face
  Trainer, and LLaMA-Factory; first-pass engine and desktop export exists.
- v0.5: Local LoRA/adapters launcher with logs, checkpoints, resume support,
  and evaluation comparison.
- v1.0: Full dataset-to-model workflow.

Training must remain downstream of dataset validation and evaluation.
