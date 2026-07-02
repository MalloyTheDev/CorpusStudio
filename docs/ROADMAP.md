# Corpus Studio Roadmap

Corpus Studio should become a one-stop shop for creating, validating, testing,
exporting, and eventually training datasets. The implementation must stay
staged so v0.1 remains focused and usable.

## Current Checkpoint

The repository now has a runnable WPF desktop app backed by the Python engine.
The local dataset loop, import quarantine, quality reports, split generation,
JSONL export, Evaluation Lab MVP, AI Assist Lab MVP, and Training Lab config
export MVP are present. The app remains local-first and file-backed.

The next roadmap work is hardening, not expanding into full trainer
orchestration. Local training launch, trainer logs, checkpoint tracking, CUDA,
PyTorch, Transformers, and cloud publishing remain outside the current app.

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

Scope (in progress):

- Role-based provider/model capability policy enforced in the engine
  (OpenAI/Anthropic evaluator-only; Ollama/local generation only when approved;
  OpenRouter route-aware). See `PROVIDER_POLICY.md`.
- Project-local, inspectable provider approval overrides.
- Gate runner with serializable pass/warn/block reports over the existing
  schema, quality, leakage, PII, and evaluation logic. See `GATES.md`.
- `docs/CURRENT_STATE.md` as the single source of truth.

Out of scope (future):

- Real hosted-provider API clients; approved-generation review-queue pipeline
  (v1.2); regression gate (needs the v0.8 run registry); desktop surfacing.

## v0.7 - Model Chat Lab / Arena

Goal: compare models side by side on prompt suites, with evaluator-only judging.

Scope (in progress):

- Run a prompt suite across several models and capture responses side by side
  (engine `arena-run`, `ArenaReport`). Responses are comparison artifacts, not
  trainable dataset rows.
- Evaluator-only judging (`--judge-model`): a judge scores each candidate and
  picks a winner, aggregated into per-model win counts / average judge scores;
  the judge must be an evaluator-role provider (OpenAI/Anthropic allowed).
- Saved comparison reports: `arena-run --project-dir` writes an inspectable
  report under `arena_reports/`.
- Planned: a desktop Arena surface.

Out of scope:

- Using arena responses as trainable data without the normal
  generate → validate → gates → review → save flow.

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
- version history
- full documentation
- examples for all built-in schemas
