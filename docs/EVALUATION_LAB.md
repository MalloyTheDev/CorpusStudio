# Evaluation Lab

Evaluation Lab tests datasets by running prompts through selected models and
comparing the responses with expected outputs, rubrics, or human scores — building
on the dataset-creation loop. (Training is now a shipped, opt-in capability via the `[train]` extra;
the eval path itself still needs no CUDA/ML deps — it only calls a serving endpoint.)

The engine `eval-run` command runs instruction or chat JSONL through Ollama or an
OpenAI-compatible local endpoint and exports a JSON report. The desktop Evaluation
tab collects backend settings, runs the same command, checks backend health, and
displays the report; it reloads saved reports from a history list, compares two
reports (score/failure/tag/example deltas), stores repeatable `run_settings`,
reruns saved configurations as regression checks, blocks runs when the pre-run
health check fails, filters review to failed rows, and saves manual per-example
scores/notes back into the report JSON. A failed row can be loaded into Writing
Studio for edit/validate/save/rerun, or prepared as a draft with an AI Assist
`rewrite-output` triage instruction. A separate multi-model `benchmark` command
ranks several models on one dataset. The default automatic score is
keyword-overlap recall — a lexical proxy, **not** a quality judgment; an opt-in
LLM-judge scorer (`--judge-model`) scores each answer 0–100 with a rationale.

## Purpose

The purpose of Evaluation Lab is to answer practical dataset questions before a
user trains or ships a model:

- Does a model already answer these examples well?
- Which examples are too vague, too easy, too hard, or inconsistent?
- Which categories fail most often?
- Did a dataset edit improve model behavior?
- Did a checkpoint regress compared with a previous run?

Evaluation Lab should treat model runs as repeatable dataset tests. A user
should be able to run a sample, inspect failures, edit examples, and rerun the
same test.

## Training Data vs Evaluation Data

Training data teaches the model. Evaluation data tests the model.

Training examples can be used to update model weights, adapters, or prompts.
Evaluation examples must be held out from training so they remain a meaningful
measurement of model behavior. If the same examples are used for both training
and evaluation, scores can look better than the model's real generalization.

Evaluation sets should be:

- separate from train splits
- stable across repeat runs
- representative of target behavior
- tagged by capability, topic, or failure mode
- small enough for fast iteration at first
- expandable into larger regression suites later

## Why Eval Sets Must Stay Separate

Mixing evaluation examples into training creates leakage. Leakage makes the
model look competent on memorized examples while hiding weak generalization. For
Corpus Studio, this means:

- train, validation, and test splits must be visible to the user
- evaluation exports should identify which examples were tested
- Training Lab must warn before using held-out eval files as training input
- before/after model comparisons must use the same held-out eval set

## Initial Workflows

Evaluation Lab should initially support these workflows:

- run a model on instruction examples
- run a model on chat examples
- compare model output against expected output
- score the model answer manually or with a simple rubric
- flag weak examples for editing
- export an evaluation report

The first version should favor human-readable evidence over clever automation.
Scores are useful, but the user needs to see the prompt, expected answer, model
answer, tags, and failure notes.

## Scope

Evaluation runs locally through the model-backend abstraction (Ollama /
OpenAI-compatible); the evaluation path needs no CUDA or heavyweight ML libraries
(those live only in the opt-in `[train]` extra, not the eval flow). Runs are one
dataset at a time, for instruction and chat data, with JSON report export.

Already shipped: multi-model comparison (`benchmark`), regression reruns,
report summaries by tag/failure/score-band, saved failure drilldowns, versioned
reviewed-fix tracking, and an opt-in judge-model scorer. Genuinely future:
rubric-based grading, checkpoint comparison, and prompt-template versioning.
- cost and token accounting
- failed-example review queues

## Example Report Format

```json
{
  "dataset": "coding_tutor_v0.1",
  "model": "qwen2.5-coder:7b",
  "examples_tested": 100,
  "average_score": 86.4,
  "failed_examples": 9,
  "weak_tags": ["recursion", "oop", "error-handling"]
}
```

## Report Design Notes

An implementation-ready evaluation report should include summary fields plus
per-example details:

- dataset id and dataset version
- split or file path used
- model backend and model name
- repeatable run settings, including schema id, backend, base URL, limit, score threshold, and timeout
- prompt template id
- start and end timestamps
- score scale
- per-example prompt, expected output, model output, score, tags, and notes
- failed example count
- weak tags or categories
- tag summary with example count, failed count, and average score
- failure reason summary from result notes or automatic score-threshold failures
- score band summary for `0-49`, `50-69`, `70-84`, and `85-100`
- app and engine version metadata

Reports should be JSON first, with Markdown or HTML summaries later.

## Current CLI MVP

```powershell
python -m corpus_studio.cli eval-run examples\datasets\instruction\train.jsonl instruction --backend ollama --model qwen2.5-coder:7b --limit 10 --output-path exports\eval_report.json
```

The CLI validates the dataset first, extracts instruction/chat prompts, calls
the selected local backend, scores model output, and writes the serializable
evaluation report with a `metric` field, a `run_settings` object, plus derived
`tag_summary`, `failure_reason_summary`, and `score_band_summary` arrays. This
command requires the chosen local backend to already be running.

Add **`--progress`** to stream progress (`[k/N] evaluated`) to **stderr** during a long
run — the report JSON on stdout is unchanged, so `--output-path` and piping still work.
Output is throttled to ~100 updates (always the first and last), so a 10k-row run streams
a readable trickle instead of one line per example. The **desktop Evaluation tab surfaces
this as a live progress bar**: a run (and a regression rerun) passes `--progress`, streams the
`[k/N]` lines back to the UI, and shows an "Evaluating k/N…" bar until the run finishes.

### Scoring metric (read this)

The default automatic score (`metric: "keyword_overlap"`) is **keyword-overlap
recall**: the fraction of the expected output's words that appear in the model
output (case-folded, whitespace-split). It is a **lexical proxy, not a quality
judgment** — a model that echoes the expected keywords plus noise scores 100, and a
correct paraphrase using synonyms scores low. It also drives the benchmark ranking
and the training-regression / eval-score gates, so treat those as keyword-overlap
signals, not quality verdicts.

Trustworthy scoring comes from two places:

- **Manual scoring** — per-example `manual_score` / `manual_notes` (`average_manual_score`
  on the report). Always available.
- **Judge-model scoring** (`metric: "llm_judge"`) — opt-in via
  `eval-run --judge-model <model> [--judge-backend … --judge-base-url … --judge-api-key …]`.
  An evaluator model scores each answer 0–100 with a `rationale` (stored per result). The
  judge provider must be **evaluator-authorized** by provider policy (so OpenAI/Anthropic
  are permitted as judges, local models fine); a run with no `--judge-model` makes no cloud
  call and stays on keyword overlap. Unparseable judge output is flagged
  (`judge_unparseable`, score 0) rather than crashing the run. The desktop Evaluation tab
  exposes the judge **model, backend, and base-url** (all optional): leaving the judge
  backend/base-url blank reuses the eval run's own provider, or set them for a **local eval
  (Ollama) scored by a cloud judge** (openai-compatible) — the classic mixed setup.

## Current Desktop MVP

The desktop Evaluation tab is a thin UI over the CLI MVP:

- backend, model, base URL, limit, score threshold, and timeout fields
- Refresh Models control that lists Ollama/OpenAI-compatible backend models
  through the engine `model-list` command
- input validation before any backend call
- single active project run using the project's `examples.jsonl`
- JSON report written under `exports/<project_id>/evaluation`
- summary and raw report JSON displayed in the app
- backend health summary for the configured provider and model
- pre-run backend health gate before evaluation generation
- saved report history loaded from `exports/<project_id>/evaluation`
- two-report comparison showing score, failure, weak-tag, and row-level deltas
- saved regression reruns that reuse report `run_settings` after a backend health gate
- summary lines for tag count/failure/average score, failure reasons, and score bands
- failed-example review filter
- failed-row edit handoff that loads the current saved row into Writing Studio
  for explicit validation, save, and rerun
- per-example result list with manual score and notes fields saved to report JSON
- failed-example AI Assist triage handoff that loads the expected answer as a
  draft and includes the prompt, expected output, model output, and score in the
  instruction
- reviewed-fix tracking that records each edited failed row, versions repeat
  edits of the same example, and marks a fix resolved or still-failing after the
  next evaluation run, persisted to `reviewed_fixes.json`
- interactive failure drilldowns that cross-filter the result list by status,
  tag, failure reason, and score band, plus named per-project failure filters
  saved to `evaluation_failure_filters.json` for repeated review passes

Multi-model benchmark comparison exists (`benchmark` runs one dataset across
several models and ranks them), and the Model Arena compares models on ad-hoc
prompt suites with optional evaluator-only judging. Streaming progress is shown live in
the desktop (a per-example progress bar, fed by `eval-run --progress`); hosted-provider
credential management and multi-run failure triage are not yet provided.

## UI Screen Basis

Evaluation Lab can become these screens:

- backend selector
- dataset and split selector
- prompt preview
- run progress
- model output review table
- report comparison panel
- score and notes panel
- weak-example queue
- report export panel
