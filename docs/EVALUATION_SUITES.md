# Evaluation Suites

A reusable, file-driven way to run several evaluation **cases** as one unit and get a
single pass/warn/block verdict — a regression suite for "does my model still clear
these bars across these datasets?". Pure orchestration over the existing evaluation
run + evaluation gate; it adds no new scoring.

## Define a suite (a JSON file you write)

```json
{
  "name": "release-gate",
  "cases": [
    {
      "name": "coding-tutor-keyword",
      "schema": "instruction",
      "dataset_path": "data/coding_tutor.jsonl",
      "model": "llama3",
      "backend": "ollama",
      "metric": "keyword_overlap",
      "min_score": 70,
      "min_pass_rate": 0.5
    },
    {
      "name": "coding-tutor-judged",
      "schema": "instruction",
      "dataset_path": "data/coding_tutor.jsonl",
      "model": "llama3",
      "backend": "ollama",
      "metric": "llm_judge",
      "judge_model": "llama3",
      "judge_backend": "ollama",
      "min_score": 75
    }
  ]
}
```

A case sources its dataset from **exactly one** of `dataset_path` (a file — mutable, may
drift between runs) or **`version_id`** (a pinned dataset version). A pinned case
reconstructs and **verifies** that version at run time (needs `--project-dir`) and
evaluates the verified rows — true reproducibility, and the report echoes the
`version_id` it ran. A pinned version that is unknown or unverifiable makes just that
case an isolated `error` (the rest of the suite still runs).

Each **case** is one dataset × model × metric × pass bars. `metric` is `keyword_overlap`
(a lexical proxy, the default) or `llm_judge` (needs a `judge_model`; the judge must be
evaluator-authorized by [provider policy](PROVIDER_POLICY.md), enforced in the engine).
`min_score` / `min_pass_rate` default to 70 / 0.5.

## Run it

```
python -m corpus_studio.cli suite-run release-gate.json --project-dir . [--strict]
```

`suite-run` runs each case's evaluation + the evaluation gate and prints a `SuiteReport`.
Every case is a **live backend evaluation** (it prints the case count first). With
`--project-dir` the report is saved to `suite_reports/<name>.json`.

- **Advisory by default** (exit `0`; the verdict is in the report). `--strict` exits `2`
  when the aggregate verdict is **block** — for CI / release gating.
- **Per-case isolation:** one unreachable/failing backend makes that case `error` (which
  blocks the suite), while the other cases still run.
- **Per-metric roll-up:** the report groups results **per metric** (e.g. keyword_overlap
  3/4 pass; llm_judge 2/2 pass). It **never folds non-comparable metric scales**
  (keyword_overlap vs llm_judge) into one number — the only cross-metric aggregate is the
  worst per-case status.
- **Honest record:** each case stores the dataset's content fingerprint at run time, so a
  saved report is honest about *what it ran on* (a case pins a mutable path; an unreadable
  path records a `null` fingerprint, never a fake one).

## Honesty

A suite **PASS** means every case cleared its threshold — **not** that the model is good
(keyword-overlap is lexical, not a quality judgment). Running a suite makes live model
calls; it is only ever user-invoked, never automatic.

## Registering suites (run by name)

Suites can live in the project as first-class files under `evaluation_suites/<name>.json`
instead of loose paths:

```
python -m corpus_studio.cli suite-init release-gate     # scaffold evaluation_suites/release-gate.json
# …edit its cases…
python -m corpus_studio.cli suite-list                  # release-gate — 2 case(s)
python -m corpus_studio.cli suite-run release-gate --project-dir . --strict
```

- **`suite-init <name> [--project-dir .] [--force]`** — writes an example definition to
  `evaluation_suites/<name>.json` for you to edit. **Refuses to overwrite** an existing
  suite unless `--force`.
- **`suite-list [--project-dir .] [--json]`** — lists registered suites (name + case
  count). A malformed file is shown as `invalid`, never crashing the listing.
- **`suite-run <suite>`** takes a **file path** (as before) **or** a registered **name**:
  if the argument isn't an existing file it's resolved to `evaluation_suites/<name>.json`
  (needs `--project-dir`); an unknown name exits `1`.

The suite `name` is validated (`letters, digits, . _ -`) and the filename stem is the
registry key, so a name can never escape `evaluation_suites/`. There is no edit/delete
command — edit the JSON directly, or `rm` the file.

## Not in M1

No suite registry / list / history, no desktop surface, no trend over time, no
`version_id`-pinned cases, no non-evaluation cases (chat/dataset gates in a suite), no
weighting or cross-suite comparison — all future.
