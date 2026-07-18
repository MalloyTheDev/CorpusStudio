# Architecture

CorpusStudio is a **local-first, end-to-end AI development ecosystem and IDE** covering the full model
lifecycle across seven co-equal product areas (see [`PRODUCT_AREAS.md`](PRODUCT_AREAS.md)). The **target
architecture** is a **Rust authoritative core + isolated Python ML workers** ("Rust owns truth; Python
computes ML and returns evidence"); today the control plane is still the dependency-light Python engine
described below, and the Rust-core migration is gated and incremental. Structurally the current control
plane has three layers:

1. **Dataset engine** — the dependency-light Python core (torch-free at import).
2. **Platform run lifecycle** — a language-neutral contracts substrate
   (`engine/corpus_studio/platform/`: RunPlan / RunEvent / BackendManifest / …) + a headless run
   supervisor + supervised in-process **and subprocess** Python training workers + a multi-backend
   trainer registry ("pick your framework": `corpus_studio`, `unsloth`).
3. **UI** — the **Tauri 2 + React** frontend (`apps/web`). The C# WPF/Avalonia desktop prototype was
   **removed** (#545) after the engine CLI took over dataset authoring (#546); `apps/web` is an early
   client whose full Studio-screen port is in progress. It is a *client* of the engine/platform; it
   contains no platform logic. Target architecture: a **Rust authoritative core** + isolated Python ML workers (#522).

The UI (the target Tauri/React head; the desktop is a retiring prototype) owns user interaction.
The Python engine owns dataset logic.

## System diagram

```text
UI client (Tauri/React)
  |
  | shells out to the engine CLI
  v
Dataset Engine
  |
  +-- schemas
  +-- validators
  +-- importers
  +-- cleaners
  +-- splitters
  +-- exporters
  +-- quality analyzers
  +-- quality triage handoffs
  +-- AI Assist prompts and review results
  +-- AI Assist review queue
  +-- AI Assist queue views
  +-- evaluation reports
  +-- model backend configs
  +-- training config templates
  +-- storage adapters
```

## UI layer

The UI layer (the Tauri 2/React frontend; the removed desktop prototype presented these) is responsible for:

- project dashboard
- schema selection
- editing examples
- showing validation results
- showing quality reports
- triggering imports/exports
- showing split/export status
- selecting model backends for Evaluation Studio MVP runs
- reviewing Evaluation Studio summary and report JSON
- reviewing AI Assist suggestions and queue states before they return to Writing Studio
- generating and previewing Training Studio config exports

### UI implementation status

The WPF/Avalonia desktop that first implemented these screens has been **removed** (#545). The
target UI is the **Tauri 2 + React** frontend (`apps/web`) — an early contract-first client whose
full Studio-screen port is in progress. The framework-agnostic design source (tokens, icon set,
screen inventory) lives in [`docs/design/`](design/) and carries forward to it.

## Engine layer

The engine is responsible for:

- loading dataset projects
- validating examples
- converting between schemas
- cleaning rows
- estimating tokens
- detecting duplicates
- splitting datasets
- exporting files
- producing quality reports
- producing AI Assist review-only suggestions
- producing evaluation report objects
- defining model backend contracts
- generating training config templates and rendered config files

## Storage model

Current storage is intentionally simple and inspectable:

```text
data/
└── projects/
    └── my_dataset/
        ├── project.json
        ├── examples.jsonl
        ├── drafts.jsonl
        └── exports/
```

`project.json` stores durable project metadata, including schema identity and
the latest successful split settings (`train_ratio`, `validation_ratio`, and
`seed`) plus project-local lab backend settings for Evaluation and AI Assist.
Dataset rows, quality history, AI Assist queues, and queue views remain in
JSON or JSONL files.

An **optional** SQLite project index (`storage/index.py`, written to
`<projects_root>/index.sqlite3`) accelerates listing and filtering projects
without changing the storage model. It is a derived cache: it is created only on
first use of the `project-list`/`project-index-rebuild` CLI commands (or when
`CORPUS_STUDIO_USE_INDEX` is set during `new-project`), can be rebuilt from the
`project.json` files at any time, and can be deleted with no data loss. JSON/JSONL
remain the inspectable source of truth.

## Testing

The Python engine has a large pytest suite (1,800+ tests) covering schemas, validation,
importers, quality, evaluation reporting, training config, dataset cards, and
the optional project index. Opt-in local Ollama integration tests
(`engine/tests/test_ollama_integration.py`) require
`CORPUS_STUDIO_OLLAMA_INTEGRATION=1` and a running backend, and self-skip when it
is unavailable.

Engine tests run in CI via `.github/workflows/engine-tests.yml`, and the web client via
`.github/workflows/web.yml`. (The desktop's xUnit suite was removed with the desktop, #545.)

## IPC options

Possible app-to-engine communication:

1. run Python engine as a CLI
2. run Python engine as a local HTTP service
3. embed Python later
4. use file-based project exchange for v0.1

Current path:

```text
UI client -> engine CLI writes/validates/exports project files -> client reads results
```

This is simple, debuggable, and avoids premature complexity.

The Evaluation Studio MVP uses the same boundary:

```text
Desktop Evaluation tab -> Python eval-run CLI -> model_backends -> JSON report -> Desktop report view
Desktop Evaluation tab -> Python backend-health CLI -> model_backends -> health summary
Desktop Evaluation tab -> Python model-list CLI -> model_backends -> model picker
Desktop Evaluation tab -> Python backend-health CLI -> pre-run health gate
Desktop Evaluation tab -> exports/<project_id>/evaluation -> report history reload
Python eval-run CLI -> report JSON tag/failure-reason/score-band summaries -> Desktop summary lines
Desktop Evaluation tab -> selected report + comparison report -> report delta summary
Desktop Evaluation tab -> selected report run_settings -> backend health gate -> regression rerun
Desktop Evaluation tab -> report JSON -> failed-example filter + manual score/notes writeback
Desktop Evaluation tab -> failed row -> Writing Studio draft -> explicit validate/save/rerun
Desktop Evaluation tab -> failed example -> expected-answer draft + AI Assist rewrite instruction
Desktop Evaluation tab -> reviewed_fixes.json -> versioned reviewed-fix tracking + resolved/still-failing reconcile
Desktop Evaluation tab -> evaluation_failure_filters.json -> saved status/tag/failure-reason/score-band drilldown filters
Desktop Settings tab -> project.json lab_settings -> Evaluation/AI Assist backend defaults
```

The selected backend must already be running locally. The evaluation / AI-Assist
path embeds no model runtimes, CUDA, PyTorch, or Transformers — it talks to an
external backend. (First-party *training* is a separate, opt-in `[train]` extra;
the dependency-light core still pulls none of those. See `TRAINING.md`.)

AI Assist uses the same boundary and remains review-first:

```text
Desktop AI Assist tab -> schema-aware action preset -> Python ai-assist CLI -> model_backends -> review result -> Writing Studio draft
Desktop AI Assist tab -> Python backend-health CLI -> model_backends -> health summary
Desktop AI Assist tab -> Python model-list CLI -> model_backends -> model picker
Desktop AI Assist tab -> ai_assist_reviews.jsonl -> filtered/searched/sorted accept/reject queue + bulk triage/undo + source/suggestion comparison
Desktop AI Assist tab -> ai_assist_queue_views.json -> saved filter/search/sort views
Desktop AI Assist tab -> ai_assist_rewrite_batches.json -> resume prepared synthetic batch rewrite drafts after restart
Desktop Quality panel -> synthetic_pattern_issues -> draft row or batch draft + AI Assist rewrite instruction
Desktop Preference Review tab -> saved preference rows -> contrast ranking/filter -> draft row or batch draft + AI Assist preference judge instruction
Desktop Preference Review tab -> visible preference ranking -> exports/<project_id>/preference_review JSON artifact
```

AI Assist review results can include validator warnings, repetitive synthetic
pattern warnings, and preference-pair strength warnings. The final save still
goes through the existing validator and explicit user action.

## Lab boundaries

Evaluation Studio, AI Assist, and Training all build on the engine. Training is **shipped and
opt-in**: the first-party QLoRA backend (the `[train]` extra) is admitted and executed only through the
hash-sealed Platform lifecycle. Heavy imports remain lazy inside its worker, so the control plane stays
dependency-light. The engine can also generate a config to launch an external trainer.

```text
Desktop Future Labs
  |
  +-- Evaluation Studio -> model_backends + evaluation reports
  +-- AI Assist  -> model_backends + validators + review-only suggestions
  +-- Training Studio   -> split/export outputs + training config templates + rendered config files
```

Model calls stay behind backend adapters. Real first-party execution is owned by `platform-plan` →
`platform-run`; backend identity, environment, capabilities, immutable inputs, and runner lane form
one chain. The Tauri/React Platform client drives that lifecycle. A UI client may launch reviewed
external-trainer argv, but no UI head owns the first-party path (that is `platform-plan` ->
`platform-run`) or contains training logic.

Training-config generation uses the same UI-to-engine boundary:

```text
UI client -> Python training-config CLI -> training templates -> rendered config file
```

The `training-config` command writes inspectable config files only. External targets may expose an
exact no-shell launch argv. A first-party config exposes no executable argv: it must become a sealed
RunPlan before dispatch. See [`TRAINING.md`](TRAINING.md).

## Design rule

The UI should not hardcode dataset behavior. The schema system should define:

- fields
- required fields
- editor hints
- validation rules
- export mappings

The same rule applies to future labs: schemas should identify how examples are
prompted, evaluated, assisted, and exported for training configs without
hardcoding dataset behavior into the UI.
