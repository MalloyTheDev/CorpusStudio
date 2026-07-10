# Architecture

Corpus Studio is split into two major layers:

1. Desktop application
2. Dataset engine

The desktop app owns user interaction.
The Python engine owns dataset logic.

## System diagram

```text
Desktop UI
  |
  | calls local engine service / CLI / IPC
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

## Desktop layer

The desktop layer is responsible for:

- project dashboard
- schema selection
- editing examples
- showing validation results
- showing quality reports
- triggering imports/exports
- showing split/export status
- selecting model backends for Evaluation Lab MVP runs
- reviewing Evaluation Lab summary and report JSON
- reviewing AI Assist suggestions and queue states before they return to Writing Studio
- generating and previewing Training Lab config exports

### Desktop internal structure

The desktop is a .NET solution with the UI logic being decomposed for cross-platform reuse:

- **`CorpusStudio.Core`** (`net8.0`) — WPF-free shared library: all Models, Services, and the
  per-tab view-models (each `IXxxViewModel` + `XxxViewModel : ViewModelBase`), plus the head-agnostic
  seams — `IEngineService` (engine run-orchestration, faked in tests), `IDialogService`, and
  `IFilePickerService` — resolved via a DI container (each with a Core `Null*` default).
- **`CorpusStudio.Desktop`** (`net8.0-windows`) — the shipping WPF head: Views + `App` + the WPF
  seam adapters (`MessageBoxDialogService`, `Win32FilePickerService`, `PythonEngineService`), referencing Core.
- **`CorpusStudio.Avalonia`** (`net8.0`) — a proof cross-platform head binding the full tab set over
  the *unchanged* Core view-models (not shipped).

All per-tab view-models are now extracted out of the former `MainWindowViewModel` god-object; what
remains on the shell is legitimate orchestration (shell mode, project list, active project, output log)
plus the engine-run commands being consolidated off the per-head code-behind (issue #184). See
[`AVALONIA_MIGRATION_PLAN.md`](AVALONIA_MIGRATION_PLAN.md).

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

The Python engine has a large pytest suite (800+ tests) covering schemas, validation,
importers, quality, evaluation reporting, training config, dataset cards, and
the optional project index. Opt-in local Ollama integration tests
(`engine/tests/test_ollama_integration.py`) require
`CORPUS_STUDIO_OLLAMA_INTEGRATION=1` and a running backend, and self-skip when it
is unavailable.

The desktop app has xUnit tests (`apps/desktop/CorpusStudio.Desktop.Tests`)
covering the `PythonEngineService` project-local JSON persistence for reviewed
fixes, evaluation failure filters, and AI Assist rewrite batches. Engine tests
run in CI via `.github/workflows/engine-tests.yml`; desktop tests run on Windows
via `.github/workflows/desktop-tests.yml`.

## IPC options

Possible app-to-engine communication:

1. run Python engine as a CLI
2. run Python engine as a local HTTP service
3. embed Python later
4. use file-based project exchange for v0.1

Current path:

```text
Desktop writes project files -> Python CLI validates/exports -> Desktop reads results
```

This is simple, debuggable, and avoids premature complexity.

The Evaluation Lab MVP uses the same boundary:

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

Evaluation Lab, AI Assist Lab, and Training Lab should build on the engine
without turning the core app into a trainer.

```text
Desktop Future Labs
  |
  +-- Evaluation Lab -> model_backends + evaluation reports
  +-- AI Assist Lab  -> model_backends + validators + review-only suggestions
  +-- Training Lab   -> split/export outputs + training config templates + rendered config files
```

Model calls stay behind backend adapters. Real training launch should stay out
until the Evaluation workflow and config export path are stable.

The Training Lab MVP uses the same desktop-to-engine boundary:

```text
Desktop Training tab -> Python training-config CLI -> training templates -> rendered config file
```

The `training-config` command writes inspectable config files only. In-app trainer
launch (exact argv, no shell), live log streaming, checkpoint tracking, and
resume-from-latest are separate shipped features — the config generator never
runs a trainer itself. See [`TRAINING.md`](TRAINING.md).

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
