# Corpus Studio Roadmap

Corpus Studio should become a one-stop shop for creating, validating, testing,
exporting, and eventually training datasets. The implementation must stay
staged so v0.1 remains focused and usable.

## Current Checkpoint

The repository has a runnable WPF desktop app backed by the Python engine, and
the full local dataset-to-model loop is shipped end to end: authoring/validation,
import quarantine, quality reports + a graded debt ledger, leakage-checked splits,
JSONL/preference export, Evaluation Lab, multi-model benchmark, Model Arena,
review-first AI Assist (with pre-review candidate gating), a governed provider
policy + gate runner, the local training launcher (in-app launch, live logs,
checkpoints, resume, before/after eval), a training run registry + regression
gate, a model artifact registry + weight card + promote gate, and dataset version
history (capture/card/diff/restore) — all local-first and file-backed. See
[`CURRENT_STATE.md`](CURRENT_STATE.md) for the authoritative feature list.

Milestones v0.1–v1.2 are complete. CUDA, PyTorch, Transformers, and cloud
publishing remain deliberately outside the app — Corpus Studio orchestrates the
user's installed trainer, it does not embed a training framework.

## v0.1 - Dataset Creation Studio

Goal: prove the local dataset authoring loop.

Scope:

- local project creation
- built-in schema templates
- dataset editors
- validation
- JSONL import preview
- JSONL export
- train/validation/test splitting
- basic quality checks
- import quarantine review/retry
- project-level quality history

Exit criteria:

- user can create a dataset project
- user can add or import examples
- examples are validated against schema
- user can split dataset into train/validation/test
- user can export valid JSONL
- user can recover rejected import rows into the editor
- user can inspect recent quality history

## v0.2 - Evaluation Lab

Goal: test datasets against local models before training.

Scope:

- Ollama backend; engine CLI and desktop MVP run path exists
- OpenAI-compatible local endpoint; engine CLI and desktop MVP run path exists
- single dataset run
- instruction and chat examples
- compare model output with expected output
- simple scoring interface
- JSON evaluation report export
- desktop Evaluation Lab workflow; first-pass tab exists
- backend health check CLI and desktop buttons
- Ollama/local backend model discovery CLI and desktop pickers
- per-project Evaluation and AI Assist backend settings persistence
- automatic backend health checks before long runs
- report history and reload UI
- comparison for two saved reports
- saved Evaluation configuration reruns for regression checks
- failed-example review queue filter
- failed-row edit handoff to Writing Studio for validate/save/rerun loops
- failed-example AI Assist triage preparation
- report summaries by tag, failure reason, and score band
- manual per-example scoring and notes
- versioned reviewed-fix tracking with re-test resolution status
- interactive failure drilldowns with saved per-project failure filters

Out of scope:

- full training launcher
- multi-model benchmark suites
- cloud-only evaluation requirements

## v0.3 - AI Assist Lab

Goal: help users review and improve datasets while preserving human control.

Scope:

- review-first engine and desktop MVP; first-pass tab exists
- persistent review queue with accept/reject states
- side-by-side source draft and suggested JSONL comparison
- schema-aware action presets instead of free-form action text
- first-pass repetitive synthetic pattern warnings
- first-pass preference-pair strength warnings
- preference-pair review UI and AI Assist judge handoff
- multi-pair preference ranking and contrast filters
- review queue filters and bulk triage controls
- review queue search, sorting, and multi-step bulk triage undo
- saved queue views for repeated review passes
- first-pass dataset-wide synthetic-pattern quality warnings
- synthetic warning severity levels and repair suggestions
- synthetic issue triage-to-rewrite handoff
- batch synthetic rewrite preparation for affected rows
- persistent rewrite batches that survive app restart
- preference ranking export for DPO/reward-model review
- visible preference queue batch judge preparation
- suggest tags
- detect vague examples
- rewrite weak outputs
- generate draft examples
- create chosen/rejected pairs
- judge preference strength
- identify schema violations
- detect repetitive synthetic patterns

Remaining hardening:

- production-grade synthetic pattern clustering
- target-specific DPO/reward-model export formats

Out of scope:

- automatic acceptance of generated data
- bypassing validators or quality gates

## v0.4 - Training Config Generation

Goal: generate inspectable training configs after datasets and evals are stable.

Current MVP status: the engine exposes a `training-config` command and the
desktop app has a Training tab that writes inspectable config files. It does
not launch training jobs.

Scope:

- Axolotl YAML config
- TRL config
- Unsloth notebook/script config
- Hugging Face Trainer config
- LLaMA-Factory config
- token budget estimate (done: Unicode-aware, tokens-per-epoch after truncation)
- VRAM estimate (done: rough arithmetic from the model-name parameter count,
  fp16/8-bit/4-bit totals with listed assumptions; never inspects hardware)
- LoRA parameter helper (done: recommended r/alpha by model size with sanity
  warnings for unusual choices)
- target-specific schema/format compatibility warnings

Out of scope:

- launching trainers
- CUDA/PyTorch/Transformers dependency in the core app

## v0.5 - Local Training Launcher

Goal: launch local LoRA or adapter jobs with enough safety and visibility to be
useful.

Current status (v0.5.0, guided): the engine emits the exact launch command per
target plus the resume variant and required dependencies (`launch` in the
`training-config` output, copyable from the desktop), and `training-checkpoints`
lists checkpoints in an output directory and builds a resume command for the
latest one. Corpus Studio does not run the trainer itself yet; in-app launch and
the log viewer would require live process streaming (see
`docs/TRAINING_LAUNCHER_DESIGN.md`).

Scope:

- local command preview (done, guided)
- training log viewer (done: in-app launch streams live stdout/stderr)
- checkpoint tracking (done: configs carry `output_dir`; the desktop polls the
  checkpoint list slowly during a run and refreshes on end/stop/error)
- resume training (done for targets with a CLI resume flag: "Resume latest"
  relaunches from the newest checkpoint through the same confirmation;
  config-driven targets show an explanatory note)
- stop/cancel support (done: Stop kills the process tree)
- before/after eval comparison (done: the newest evaluation report is captured
  as the baseline at launch; after evaluating the trained model, "Compare vs
  baseline" shows score/failure/tag deltas in the Training tab)

Out of scope:

- cloud training orchestration by default
- hiding trainer commands from the user

## v0.6 - Provider Policy + Gate Foundation

Goal: enforce, in the engine, who may generate trainable data, and gate whether
data/exports/evaluations may move forward.

Scope (done):

- Role-based provider/model capability policy enforced in the engine
  (OpenAI/Anthropic evaluator-only; Ollama/local generation only when approved;
  OpenRouter route-aware). See `PROVIDER_POLICY.md`.
- Project-local, inspectable provider approval overrides.
- Gate runner with serializable pass/warn/block reports over the existing
  schema, quality, leakage, PII, and evaluation logic, plus per-project
  thresholds. See `GATES.md`.
- Desktop surfacing: a **Provider Generation Policy** panel (Settings) and a
  **Run Gates** button (Quality tab).
- `docs/CURRENT_STATE.md` as the single source of truth.

Delivered later, as planned:

- The regression gate landed with the v0.8 run registry; approved-generation
  candidate gating landed in v1.2. Real hosted-provider API clients remain out
  of scope (evaluator-only providers are configured, not embedded).

## v0.7 - Model Chat Lab / Arena

Goal: compare models side by side on prompt suites, with evaluator-only judging.

Scope (done):

- Run a prompt suite across several models and capture responses side by side
  (engine `arena-run`, `ArenaReport`). Responses are comparison artifacts, not
  trainable dataset rows.
- Evaluator-only judging (`--judge-model`): a judge scores each candidate and
  picks a winner, aggregated into per-model win counts / average judge scores;
  the judge must be an evaluator-role provider (OpenAI/Anthropic allowed).
- Saved comparison reports: `arena-run --project-dir` writes an inspectable
  report under `arena_reports/`.
- Desktop Arena tab: prompts + model list (+ optional judge) → side-by-side
  responses with per-model win/score summary and the judge's winner.
- Planned: per-provider backend selection in the UI, saved-report history
  browsing, per-cell rerun.

Out of scope:

- Using arena responses as trainable data without the normal
  generate → validate → gates → review → save flow.

## v0.8 - Training Run Registry

Goal: durable, inspectable records of training runs (the spine v0.9 artifacts and
the regression gate both need).

Scope:

- v0.8.0 (done): a `TrainingRunRecord` (run_id, timestamps, status, target,
  base_model, config_path, argv, output_dir, pid, exit_code, checkpoints,
  before/after eval links, notes) written per-run under `training_runs/`. The
  desktop writes records directly (no subprocess on the crash path); the engine
  owns the schema + `training-run-list`/`training-run-update` + storage. A run
  left `running` whose process is gone reconciles to `interrupted` on load (pid
  liveness), so a force-closed run does not stay `running` forever. A read-only
  run history in the Training tab.
- v0.8.1 (done): the `training_run` regression gate — `training-run-gate` blocks
  when the trained model regressed vs the baseline, and warns with "unverified
  linkage" when the after-eval targeted the base model (provenance enforced via
  the record's `after_eval_model`). Surfaced by a "Gate run" button that links
  the newest trained-model eval and runs the gate.

Out of scope:

- Re-launch from a record, config diffing, delete/archive (that is v0.9).
- Backfill of past ephemeral runs — the registry records from here forward.

## v0.9 - Model Artifact / Weight Registry

Goal: track the model artifacts a training run produced (adapters, promoted
checkpoints) as durable, inspectable first-class objects the user keeps or
rejects — referenced by path, never owned.

Scope:

- v0.9.0 (done): a `ModelArtifactRecord` under `model_artifacts/` with
  keep/reject status, referencing the source `run_id` (base model + eval are
  resolved live through the run, never stored). The headline feature is **path
  integrity**: a cheap fingerprint (size + mtime of a key descriptor file, never
  a hash of weight bytes) is captured at register time and re-checked on load,
  flagging `missing` or `modified` so a record can never quietly point at
  deleted/overwritten weights. Idempotent register (same run+path → one record).
  Engine `artifact-register`/`artifact-list`/`artifact-update`; a desktop
  Artifacts tab (register-from-run, keep/reject, integrity badge).
- v0.9.1 (done): a weight card rendered live (`artifact-card`, never stored, so
  it can't drift; carries the "unverified linkage" caveat) and the
  `model_artifact` promote gate (`artifact-gate`) that blocks "keep" when the
  artifact is `modified`/`missing` or the source run regressed. Keep in the
  desktop Artifacts tab is gated: a block refuses the keep.

Out of scope:

- Any move/copy/delete of weight files; gguf/quantization/merge/serve;
  auto-detection of "the final adapter."

## v1.0 - Full Dataset-to-Model Workflow

Goal: stable end-to-end workflow from dataset creation to evaluated model
artifacts.

Scope:

- stable schema engine
- stable project storage
- stable import/export center
- Evaluation Lab
- AI Assist Lab
- Training Lab
- version history — **complete (v1.0.0–v1.0.3, engine + desktop)**: dataset
  version registry with a content fingerprint, live drift detection, lineage
  links, and a version card (`dataset-version-create/-list/-show`); a
  content-addressed row store with per-version manifests; `dataset-version-diff`
  and `dataset-version-restore`; and a desktop **Versions** tab with capture, view
  card, diff view, and in-place restore. See [VERSIONING.md](VERSIONING.md).
- full documentation
- examples for all built-in schemas

## v1.1 - Dataset Debt Ledger

Goal: turn the raw quality report into a prioritized, graded "what do I fix
first?" answer.

Scope (done):

- Engine `dataset-debt`: quality signals normalized by dataset size, ranked by
  severity, graded A–F, each with a remediation. Secrets/PII graded by presence
  (a single leaked key is critical), everything else by rate. See
  [DEBT.md](DEBT.md).
- A desktop **Debt** tab (color-coded grade + ranked remediation list) whose
  grade invalidates the moment the dataset changes.

Deferred: a Dashboard grade badge (auto-run on open), trend over time, and
folding gate results into the ledger.

## v1.2 - Approved Provider Generation (candidate gating)

Goal: close the `generate → validate → quality → gates → human review` chain for
AI-generated candidates.

Scope (done, engine):

- `run_ai_assist` runs the existing dataset gate runner (schema/quality/PII) over
  the generated candidate rows and attaches the verdict as `candidate_gate` — a
  pre-review signal only. `review_required` stays true; a clean gate is not
  approval; a block does not auto-reject; provider policy is enforced before
  generation. See [AI_ASSIST_LAB.md](AI_ASSIST_LAB.md).

Deferred: **v1.2.1** desktop surfacing of `candidate_gate` + a confirm-on-block
affordance; a candidate-vs-existing novelty/contamination check; bulk
generate-N-from-spec.

## v1.3 - Evaluation Suites & Chat Gates

Not started. Reusable evaluation suites and chat-scope gates.
