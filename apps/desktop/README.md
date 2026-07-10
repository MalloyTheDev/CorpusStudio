# Corpus Studio Desktop

C# WPF desktop app for Corpus Studio. (A proof `CorpusStudio.Avalonia` head binds the same
`CorpusStudio.Core` view-models cross-platform; WPF stays the shipping head — see
[`../../docs/AVALONIA_MIGRATION_PLAN.md`](../../docs/AVALONIA_MIGRATION_PLAN.md).)

The desktop workflow supports:

- a Dashboard landing tab with quick-action buttons (author, run quality,
  generate splits, run evaluation, training config, new project) and
  at-a-glance cards for dataset size, quality, splits, and evaluation
- local dataset project creation
- built-in schema selection, with each template showing a description and a
  valid example row that is pre-filled into the editor on project creation
- JSON example authoring
- Python-engine validation with selectable issue navigation
- saving examples to the active project
- JSONL / CSV / TSV / Parquet import preview with failed-row reporting (CSV/TSV/Parquet convert to a staging JSONL through the same preview/quarantine path; Parquet needs the engine's optional `[parquet]` extra)
- quarantine review and retry for rejected import rows
- quality checks for empty rows, duplicates, low-information examples, and first-pass synthetic-pattern warnings with repair suggestions plus single-row and batch triage-to-rewrite handoffs
- project-level quality history
- saved example inspection
- project reopening from the project list
- "Rebuild Index" action that rebuilds the optional engine SQLite project index
  and re-lists projects from it (JSON/JSONL stay authoritative)
- train/validation/test split generation with saved ratios, seed, and tiny-split warnings
- Evaluation Lab runs, backend checks, pre-run health gates, report history, two-report comparison, saved regression reruns, report summaries by tag/failure reason/score band, failed-example review filtering, failed-row edit handoff to Writing Studio, failed-example AI Assist triage preparation, manual per-example notes/scores, and **multi-model benchmark comparison**, for configured Ollama or OpenAI-compatible local endpoints (optional LLM-judge scorer)
- Evaluation **Suites** tab: register, scaffold, and run named evaluation suites with per-metric roll-ups, per-case results, and run history/trend
- AI Assist backend checks plus schema-aware action presets, persistent review queue, filters, search, sorting, saved queue views, persistent rewrite batches, bulk triage with multi-step undo, accept/reject states, source/suggestion comparison, batch synthetic rewrite preparation, preference-pair judge handoff, preference ranking export, and visible batch judge preparation
- Training Lab: config export **and launching the trainer — the opt-in first-party QLoRA trainer (`train-check`/`train-run`/`train-merge`/`model-fetch`/`model-card`) or your installed external trainer — with live streamed logs, checkpoint listing, run history, a training-run regression gate, and resume-from-checkpoint**
- model **Artifacts** registry (register a run's output, promote-gate to keep, reject) and dataset **Versions** (capture, card, diff, restore-in-place with an undo capture)
- prompt **Arena** for side-by-side model comparison, and a **Debt** tab (graded A–F dataset-debt ledger with ranked remediation)
- Hugging Face Hub dataset import (read-only, public) through the normal import-preview/quarantine flow
- gate runs (schema/quality/leakage/PII/eval + chat-structure) with an editable per-project gate-threshold editor and a provider generation-policy approve/revoke surface
- export as JSONL (default, model-ready, all schemas), CSV/TSV for flat schemas (chat/nested-object schemas are refused, not lossy-flattened), or Parquet (columnar, all schemas incl. chat/nested; needs the engine's optional `[parquet]` extra), with optional dedupe / drop-low-information cleaning and PII/secret redaction
- local settings inspection and per-project lab backend settings persistence
- polished desktop shell styling, a workflow stage strip, and a wired sidebar
  Export Center affordance
- a blocking busy overlay with progress feedback during long engine/model runs
  (evaluation, AI Assist, quality, splits, import, export, training config,
  dataset card, backend/model checks, index rebuild) that prevents duplicate runs
- empty-state placeholders on the Examples, Evaluation result, Reviewed Fixes,
  AI Assist review, import quarantine, validation-issue, and synthetic-issue
  lists, plus labeled Preference Review panes
- a shared, dismissible error banner that surfaces operation failures (backend
  unreachable, validation/import/quality/split/training errors) at the top of
  the workspace instead of burying them in a summary box

Build and launch from the repository root:

```powershell
dotnet build apps\desktop\CorpusStudio.Desktop.sln
.\apps\desktop\CorpusStudio.Desktop\bin\Debug\net8.0-windows\CorpusStudio.Desktop.exe
```

### Package a standalone build

A self-contained, single-file Windows build (no installed .NET runtime required):

```powershell
dotnet publish apps\desktop\CorpusStudio.Desktop\CorpusStudio.Desktop.csproj -p:PublishProfile=win-x64
# -> apps\desktop\CorpusStudio.Desktop\bin\publish\win-x64\CorpusStudio.Desktop.exe
```

The version comes from `apps\desktop\Directory.Build.props` (single source). The standalone
`.exe` still needs the local **Python engine** at run time — it is not bundled. If the app
can't find the engine it shows a "Python engine not found" setup screen (locate the folder or
set `CORPUS_STUDIO_ENGINE_DIR`) instead of crashing. Bundling a Python runtime is future work.

JSONL imports are previewed against the active schema and only fully valid files
are appended directly to the active project's `examples.jsonl`. Mixed-validity
imports can append valid rows after confirmation and save rejected rows under
the project's `import_quarantine` folder for repair. CSV, TSV, and Parquet files
are converted to a staging JSONL first and then run through the same
preview/quarantine/commit path. CSV/TSV cells are text (so a cell whose schema
field expects a number or list quarantines like any invalid row), while Parquet
keeps its column types; Parquet import/export needs the engine's optional
`[parquet]` extra and fails fast with an install hint when it is absent.

Split generation lets the user set train and validation percentages plus a
deterministic seed. The test split uses the remaining percentage, and output is
written under the configured export directory. The split report warns when
validation or test output is empty or only one row. Successful split settings
are saved in the project's `project.json` and reload with the project.

The Evaluation tab is intentionally small: it shells through the Python
engine's `eval-run` command, writes a JSON report under the configured export
directory, and shows the report JSON in the app. Its Check Backend button uses
the engine's `backend-health` command for a reachable/model-listed summary, and
Refresh Models uses the engine's `model-list` command to populate the model
picker from running Ollama or OpenAI-compatible endpoints. It
also reloads saved JSON reports from the project's evaluation export folder. It
can compare two saved reports for score, failure, weak-tag, and row-level
deltas, rerun saved configurations from report `run_settings`, summarize
results by tag/failure reason/score band, drill down and save named failure
filters by status/tag/failure-reason/score-band, filter the example review queue
to failed rows, and persist manual per-example scores and notes back into the
report JSON. A failed row can be loaded back into Writing Studio as the current
saved JSON row for explicit edit, validation, save, and rerun, and the edit is
recorded as a versioned reviewed fix that auto-reconciles to resolved or
still-failing on the next evaluation run. A failed example
can also be prepared for AI Assist triage, which loads a draft from the
expected answer and copies the prompt, expected output, model output, and score
into the AI Assist instruction.
Evaluation runs and regression reruns perform a pre-run backend health check
and require the selected local backend to already be running. It also runs a
multi-model benchmark comparison across several models. It does not implement
hosted-provider setup (cloud providers stay evaluator-only by policy and their
transports are not embedded — see [`../../docs/PROVIDER_POLICY.md`](../../docs/PROVIDER_POLICY.md)).

The Settings tab can save the current Evaluation and AI Assist backend, model,
base URL, and timeout into the active project's `project.json` under
`lab_settings`. Saved lab settings reload with the project so local Ollama model
choices do not need to be retyped each launch.

The AI Assist tab is also review-first. It shells through the Python engine's
`ai-assist` command, shows the model response and any suggested JSONL, and can
move a suggestion into Writing Studio for human editing. Its Check Backend
button uses the same `backend-health` command as Evaluation, and Refresh Models
uses `model-list` for local model discovery. Reviews are saved to
`ai_assist_reviews.jsonl` in the project folder with pending, accepted, and
rejected states. Selecting a queued review shows the original source draft next
to the suggested JSONL with a compact comparison summary. The action control
offers schema-aware presets, and engine warnings include first-pass repetitive
synthetic-pattern and preference-pair strength checks. The queue can be filtered
by review state, searched, sorted, bulk-mark visible reviews accepted or
rejected, save named queue views for repeated triage passes, and undo recent
bulk triage actions in sequence. It never saves AI-generated rows directly to
the accepted dataset.

The Quality panel can surface structured synthetic-pattern issues as a triage
list. Preparing a rewrite loads the first affected row into the draft editor,
sets AI Assist to `rewrite-output`, and copies the repair guidance into the AI
Assist instruction. Preparing a batch rewrite loads affected rows from the
current issue set as a JSON array and asks AI Assist for corrected JSONL rows.
Prepared batch rewrites are saved to `ai_assist_rewrite_batches.json` in the
project folder and can be resumed from the AI Assist tab after restart.
The user still runs AI Assist, reviews the suggestion, validates the result,
and saves explicitly.

The Preference Review tab is a lightweight DPO/reward-model review surface for
preference projects. It ranks saved pairs by weak, moderate, or strong
chosen/rejected contrast, can filter the queue by contrast strength, shows
prompt/chosen/rejected/reason fields, and can prepare AI Assist's
`judge-preference-strength` action for the selected pair. The visible queue can
also be prepared as a batch judge pass or exported as an inspectable JSON
ranking artifact under `exports/<project_id>/preference_review`. An "Export for
Training" action reshapes the pairs into a trainer-ready JSONL format (DPO, KTO,
or reward) via the engine `preference-export` command, written under
`exports/<project_id>/preference_export`.

The Training tab shells through the Python engine's `training-config` command.
It prefers generated train/validation split files when they exist, falls back to
the project's saved examples for config preview, and writes rendered config
files under the configured export directory. A "Check Compatibility" button runs
the engine's `training-compat` pre-check and reports schema/format/target
mismatches before generating, so problems surface early. It then **launches the
trainer** with the generated command (after a confirm) — either the **opt-in
first-party QLoRA trainer** (`train-check` preflights the runtime, `train-run`
runs a 4-bit QLoRA in-process and writes the adapter + a model card, `train-merge`
merges it, `model-fetch` downloads a permissive base) or your **installed external
trainer** — streaming its stdout/stderr live, listing checkpoints, recording each
run, resuming from a checkpoint, and running a **regression gate** against the
baseline. The dependency-light core installs no ML packages; those come from the
opt-in `[train]` extra (which delegates to TRL/peft, no bundled framework).

The desktop shell uses shared WPF styles for controls, tabs, side rails, and
the project header so new lab surfaces should reuse the existing visual frame
instead of adding one-off styling.
