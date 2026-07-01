# Corpus Studio Desktop

C# WPF desktop app for Corpus Studio.

The current v0.1 workflow supports:

- dashboard
- local dataset project creation
- built-in schema selection, with each template showing a description and a
  valid example row that is pre-filled into the editor on project creation
- JSON example authoring
- Python-engine validation with selectable issue navigation
- saving examples to the active project
- JSONL import preview with failed-row reporting
- quarantine review and retry for rejected import rows
- quality checks for empty rows, duplicates, low-information examples, and first-pass synthetic-pattern warnings with repair suggestions plus single-row and batch triage-to-rewrite handoffs
- project-level quality history
- saved example inspection
- project reopening from the project list
- "Rebuild Index" action that rebuilds the optional engine SQLite project index
  and re-lists projects from it (JSON/JSONL stay authoritative)
- train/validation/test split generation with saved ratios, seed, and tiny-split warnings
- Evaluation Lab MVP runs, backend checks, pre-run health gates, report history, two-report comparison, saved regression reruns, report summaries by tag/failure reason/score band, failed-example review filtering, failed-row edit handoff to Writing Studio, failed-example AI Assist triage preparation, and manual per-example notes/scores for configured Ollama or OpenAI-compatible local endpoints
- AI Assist MVP backend checks plus schema-aware action presets, persistent review queue, filters, search, sorting, saved queue views, persistent rewrite batches, bulk triage with multi-step undo, accept/reject states, source/suggestion comparison, batch synthetic rewrite preparation, preference-pair judge handoff, preference ranking export, and visible batch judge preparation
- Training Lab MVP config export for inspectable trainer config files
- JSONL export
- local settings inspection and per-project lab backend settings persistence
- polished desktop shell styling, a workflow stage strip, and a wired sidebar
  Export Center affordance

Build and launch from the repository root:

```powershell
dotnet build apps\desktop\CorpusStudio.Desktop.sln
.\apps\desktop\CorpusStudio.Desktop\bin\Debug\net8.0-windows\CorpusStudio.Desktop.exe
```

JSONL imports are previewed against the active schema and only fully valid files
are appended directly to the active project's `examples.jsonl`. Mixed-validity
imports can append valid rows after confirmation and save rejected rows under
the project's `import_quarantine` folder for repair.

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
and require the selected local backend to already be running. It does not
implement multi-model benchmark comparison, hosted-provider setup, or training
launch.

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
ranking artifact under `exports/<project_id>/preference_review`.

The Training tab shells through the Python engine's `training-config` command.
It prefers generated train/validation split files when they exist, falls back to
the project's saved examples for config preview, and writes rendered config
files under the configured export directory. It does not launch trainers,
install ML packages, show logs, manage checkpoints, or resume runs.

The desktop shell uses shared WPF styles for controls, tabs, side rails, and
the project header so new lab surfaces should reuse the existing visual frame
instead of adding one-off styling.
