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
Desktop Settings tab -> project.json lab_settings -> Evaluation/AI Assist backend defaults
```

The selected backend must already be running locally. Corpus Studio does not
embed model runtimes, CUDA, PyTorch, Transformers, or trainer orchestration in
this pass.

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

The command writes inspectable config files only. Launchers, trainer process
management, logs, checkpoints, and resume controls remain future v0.5 work.

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
