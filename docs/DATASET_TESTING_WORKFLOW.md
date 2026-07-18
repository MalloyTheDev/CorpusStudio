# Dataset Testing Workflow

This workflow describes how Corpus Studio should move from dataset authoring to
evaluation and later training preparation.

The workflow maps to the shipped UI sequence: authoring and validation, then
splitting and export, then evaluation and training preparation.

Current MVP coverage: the desktop app covers the active-project version of
steps 4 through 8. Users can enter local backend settings, refresh local model
lists, run the engine `eval-run` command, check backend health, inspect the
JSON report, reload saved reports from history, compare two saved reports,
rerun saved configurations as regression checks, save manual per-example scores
and notes, inspect tag/failure-reason/score-band summaries, filter failed
examples, load failed rows back into Writing Studio for explicit edits, and
prepare AI Assist triage. Steps 9 and 10 have an edit/save/rerun loop with
versioned reviewed-fix tracking and re-test resolution status.

## End-to-End Flow

1. Create dataset.
2. Validate schema.
3. Split train/validation/test.
4. Select model backend.
5. Run evaluation sample.
6. Compare model output to expected output.
7. Score examples.
8. Flag weak examples.
9. Edit dataset.
10. Re-run evaluation.
11. Export clean dataset.
12. Later generate training config.

## Screen Basis

### Dataset Setup

The user creates or opens a project, selects a schema, writes examples, imports
JSONL rows, and validates the dataset.

Useful screen elements:

- project picker
- schema picker
- example editor
- validation result panel
- import preview panel
- saved example table

### Split Review

The user generates train, validation, and test splits and confirms that
evaluation examples are held out from training.

Useful screen elements:

- split ratio controls
- seed control
- split counts
- warnings for tiny or empty splits
- export paths

### Backend Selection

The user selects a local model backend for Evaluation Studio.

Useful screen elements:

- provider selector
- base URL field
- model name field
- health check button
- timeout and token controls
- local-first warning when cloud providers are selected

Current desktop MVP fields: backend, model name, base URL, sample limit, score
threshold, and timeout.

### Evaluation Run

The user runs a sample of examples through the selected model.

Useful screen elements:

- split selector
- sample size
- prompt preview
- run progress
- stop button
- output table

Current desktop MVP runs the active project's `examples.jsonl` as one dataset
sample and displays summary plus raw report JSON and per-example review rows.
It summarizes results by tag, failure reason, and score band, compares two
saved reports for score, failure, tag, and common-example deltas, and reruns a
saved report's stored backend/model/threshold/limit settings as a regression
check. Split selection, progress streaming, and cancellation remain planned.

### Scoring and Weak Example Review

The user compares model output against expected output and assigns scores or
failure tags.

Useful screen elements:

- expected answer panel
- model answer panel
- score input
- tag/failure/score-band summaries
- failure tags
- notes field
- flag as weak button
- jump to editor

Current desktop MVP supports failed/passed/manual-score filters, manual score
and notes writeback, an Edit Failed Row action that loads the current saved
`row-N` JSON into Writing Studio, and a Prepare Failure action that loads the
failed example as a draft and writes an AI Assist triage instruction containing
the prompt, expected answer, model answer, and score.

### Edit and Re-Test

The user edits weak examples, validates again, then reruns the same evaluation
sample to compare results.

Useful screen elements:

- weak-example queue
- edit failed row button
- before/after score comparison
- saved report comparison summary
- rerun saved configuration button
- unresolved failures
- export report button

Current desktop MVP supports the basic loop: filter to failed rows, load a
failed row into Writing Studio, edit and validate it, save the reviewed row, and
rerun the saved evaluation configuration for comparison.

For synthetic-pattern quality issues, the desktop can also prepare a selected
row or affected row batch for AI Assist `rewrite-output` review. Prepared batch
rewrite handoffs are saved project-locally so they can be resumed after restart
before validation and explicit save.

### Training Preparation

After dataset quality and evaluation are stable, the user generates a training
config. The current desktop MVP exposes this as a Training tab backed by the
engine `training-config` command. It writes a config file only.

Useful screen elements:

- target tool selector
- base model field
- train/eval dataset path preview
- token and VRAM estimates in later iterations
- LoRA parameter fields
- config preview
- export config button

## Guardrails

- Keep evaluation examples separate from training examples.
- Training is available through two explicit authorities: the app can launch a reviewed argv for an
  installed external trainer, while first-party QLoRA requires a hash-sealed `RunPlan` dispatched by
  the platform supervisor. The desktop does not route a mutable config directly into the first-party
  trainer.
- Do not require cloud APIs for local evaluation.
- Do not accept AI-generated examples without review.
- Do not treat a model score as proof that a dataset is correct.

---

## Dataset Splitting

_Consolidated from the former `docs/SPLITTING.md`._

Corpus Studio should split datasets into:

- train
- validation
- test

### v0.1

Basic random split with configurable desktop ratios and seed:

```text
default train: 90%
default validation: 5%
default test: remaining 5%
default seed: 42
```

The desktop app exposes train percentage, validation percentage, and seed.
The test split is derived from the remaining percentage so the UI cannot create
a fourth hidden split. Successful split settings are saved in the project's
`project.json` and restored when the project is reopened.

The engine CLI accepts the same values:

```powershell
python -m corpus_studio.cli split input.jsonl exports\my_dataset\splits instruction --train-ratio 0.8 --validation-ratio 0.1 --seed 123
```

The split report includes warnings when validation or test output is empty or
only one row. These warnings do not block file generation; they tell the user
that evaluation or regression scores from the split will be weak until more
examples are added or ratios are adjusted.

### Planned Split Hardening

Better split strategies:

- tag-balanced split
- source-aware split
- group-aware split
- dedupe-aware split

### Leakage warning

Near-identical examples should not appear in both train and test sets. The
`split` command now detects this: after splitting it reports
`rows_shared_across_splits` and a `leakage` breakdown (which splits share each
duplicate group, and whether the match is exact or normalized/near-duplicate)
using the same NFKC/Unicode-aware normalization as the quality report, and adds
a train/test-leakage warning. Detection is non-destructive — it reports the
collision rather than silently moving rows, so the user decides how to dedupe.
