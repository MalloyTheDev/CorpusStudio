# Current State

Single source of truth for what Corpus Studio actually does today. When another
doc disagrees with this file, this file wins (and the other doc should be fixed).

Last reconciled: 2026-07-02 (v1.0.0 engine slice landed).

## What works today (implemented and tested)

**Author & validate**
- Local project creation from built-in schema templates with pre-filled examples.
- Schema validation through the Python engine: required non-empty fields,
  declared field types, list element types, enum/label sets, numeric bounds,
  nested object shapes, and chat message structure, with selectable issue
  navigation in the desktop.
- JSONL import preview with failed-row quarantine, review, and retry.
- Full Unicode correctness end to end: NFKC-aware tokenization and UTF-8 stdio
  so CJK/Cyrillic/accented text round-trips between the desktop and engine.

**Clean & measure**
- Quality report: empty rows, exact + normalized duplicates, low-information
  rows, synthetic-pattern warnings with near-duplicate clustering, PII/secret
  detection (emails, SSNs, private keys, AWS/API keys, JWTs, Luhn-valid cards —
  masked samples), token-length outliers, and category-imbalance warnings, with
  project-level quality history.
- Leakage-checked splits: `detect_split_leakage` reports exact and
  near-duplicate rows shared across train/validation/test.
- Export with an optional cleaning pass (dedupe / drop low-information) that
  writes a removal manifest; verbatim exports warn when duplicates remain.
- Preference exports (DPO/KTO/reward) with a pair-integrity gate.
- Dataset card summarizing metadata, schema, splits, quality, and evaluation.

**Evaluate & assist**
- Evaluation Lab against local Ollama or OpenAI-compatible endpoints: health
  checks, model discovery, report history, two-report comparison, regression
  reruns, tag/failure/score-band summaries, failed-row edit loops, manual
  scoring, saved failure filters.
- Multi-model benchmark: one dataset across several models, ranked, with
  per-model deltas and the examples every model failed.
- Review-first AI Assist Lab: persistent accept/reject queue, saved views, bulk
  triage with undo, resumable rewrite batches. AI Assist output is always
  `review_required` and never auto-accepted.

**Train (v0.5 launcher — complete)**
- Training config export for axolotl / TRL / Unsloth / Hugging Face /
  LLaMA-Factory with compatibility warnings, a real token budget
  (tokens-per-epoch after truncation, over-length counts), a rough arithmetic
  VRAM planning estimate (never inspects hardware), and a LoRA rank/alpha
  suggestion.
- In-app launch of the user's installed trainer with explicit confirmation of
  the exact argv (no shell), live log streaming, and a Stop that kills the
  process tree.
- Checkpoint tracking during and after runs, resume-from-latest for targets
  with a CLI resume flag, and before/after evaluation comparison against the
  baseline captured at launch.

## Hard boundaries (by design)

- Corpus Studio orchestrates the user's installed tools; it is **not** a deep
  learning framework. No CUDA, PyTorch internals, backprop, optimizers,
  distributed training, or custom training loops.
- Trainer launches show the exact argv, require explicit confirmation, use no
  shell, and write inspectable run metadata. No hidden trainer behavior.
- No silent cloud behavior, publishing, dataset upload, or auto-acceptance of
  AI-generated dataset rows.

## In progress — v0.6 (Provider Policy + Gate Foundation)

- Role-based provider/model capability policy, enforced in the engine:
  OpenAI/Anthropic evaluator-only by default; Ollama/local generation only when
  explicitly approved; OpenRouter route-aware. Surfaced by a **Provider
  Generation Policy** panel in the desktop Settings tab (shows which providers
  may generate, and approve/revoke a local model). See
  [`PROVIDER_POLICY.md`](PROVIDER_POLICY.md).
- A gate runner producing serializable pass/warn/block reports over existing
  schema, quality, leakage, PII, and evaluation logic, surfaced by a **Run Gates**
  button in the desktop Quality tab (overall status + per-gate pass/warn/block
  with repair hints). See [`GATES.md`](GATES.md).

## In progress — v0.7 (Model Chat Lab / Arena)

- Run a prompt suite across several models and capture responses side by side
  (engine `arena-run` → `ArenaReport`). Responses are comparison artifacts, not
  trainable rows.
- Optional evaluator-only judging (`--judge-model`): a judge scores each
  candidate and picks a winner, aggregated into per-model win counts and average
  judge scores. Judging is an evaluator activity, so OpenAI/Anthropic are
  permitted as the judge (enforced via provider policy).
- Saved comparison reports: `arena-run --project-dir` persists the report under
  project-local `arena_reports/`.
- A desktop **Arena** tab: enter prompts (one per line) + a model list (+ an
  optional judge), Run, and see side-by-side responses per prompt with per-model
  win/score summary and the judge's winner marked.

## In progress — v0.8 (Training Run Registry)

- Durable, inspectable training run records under `training_runs/`: launch
  metadata (argv, config, output dir), status lifecycle, pid, exit code,
  checkpoints, and the before-eval link. The desktop writes records directly;
  a run left `running` whose process is gone reconciles to `interrupted` on
  load. A read-only run history shows past runs in the Training tab.
- A `training_run` **regression gate** (`training-run-gate`): blocks when the
  trained model regressed vs the baseline, and warns with "unverified linkage"
  when the after-eval targeted the base model (provenance via the record's
  `after_eval_model`). Surfaced by a "Gate run" button in the Training tab.

## In progress — v0.9 (Model Artifact / Weight Registry)

- Durable model artifact records under `model_artifacts/` (adapters/checkpoints
  a run produced) with keep/reject status, referenced by path (never moved).
  Base model + eval are resolved live through the source `run_id`, not stored.
  **Path integrity** is re-checked on load — a record flags `missing` or
  `modified` if the weights change on disk. A desktop Artifacts tab registers
  from a run and keeps/rejects.
- A weight card rendered live (`artifact-card`, never stored) with base model +
  eval resolved through the run and the "unverified linkage" caveat, and a
  `model_artifact` **promote gate** (`artifact-gate`) that blocks "keep" when the
  artifact is `modified`/`missing` or the source run regressed. Keep in the
  desktop is promote-gated — a block refuses the keep.

## In progress — v1.0 (Dataset Version History & Lineage)

- Durable dataset version records under `dataset_versions/` (engine): each pins
  the dataset's identity — `row_count` + a streaming SHA-256 `content_fingerprint`
  over the ordered per-row exact signatures (the same signature primitive used by
  cleaning/quality/leakage) — plus links to source training runs, model
  artifacts, an evaluation report, and a dataset gate report. Nothing derivable
  is stored; scores/integrity/gate status resolve live in a version card.
- Live drift detection: listing and the card recompute the current
  `examples.jsonl` fingerprint and report `matches` / `drifted` / `unreadable`,
  so a version can never silently misrepresent a changed dataset. The card leads
  with a warning when drifted or a link is missing.
- CLI `dataset-version-create` / `dataset-version-list` / `dataset-version-show`;
  `--stamp-run` writes the dataset→run back-link (`source_snapshot_id`). The
  engine only reads `examples.jsonl` and writes under `dataset_versions/` — it
  never moves/copies/deletes the dataset. See [`VERSIONING.md`](VERSIONING.md).
- A desktop **Versions** tab surfaces the history: a read-only list with a live
  integrity badge (matches/drifted/unreadable), an opt-in **Capture version**
  button, and **View card**. Capture and listing go through the engine, so the
  desktop never recomputes the fingerprint (integrity is verified, not guessed).
- Stable per-row identity + a content-addressed, deduped row store
  (`dataset_versions/row_store.jsonl`) with a per-version ordered manifest,
  captured in one pass with the fingerprint. `dataset-version-diff` compares two
  versions (multiset added/removed/common + sample rows). Storing rows is the
  default (`--no-store-rows` opts out; cost is surfaced, not silent). Rows are
  stored canonically, so diff/restore normalize key order (not byte-identical).
- `dataset-version-restore` reconstructs a version's rows from the store to an
  `--output` file, verified against the recorded fingerprint (all-or-nothing,
  overwrite-safe, atomic). The engine **refuses to write `examples.jsonl`** — the
  dataset has one writer (the desktop); in-place restore is deferred to the desktop.
- Deferred: desktop in-place restore (atomic replace of `examples.jsonl` +
  quiescence + auto-capture), desktop diff surfacing, auto-capture after import
  commit, reorder detection, store GC, and a normalized row identity.

## Not built yet (future roadmap)

Dataset Version diff & restore (v1.0.2+), Dataset Debt Dashboard (v1.1),
Approved Provider Generation into a review queue (v1.2), Evaluation Suites &
Chat Gates (v1.3).
