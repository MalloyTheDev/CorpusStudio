# Corpus Studio

<img src="assets/branding/corpusstudio-256.png" alt="Corpus Studio" width="104" align="right" />

[![Engine coverage](https://codecov.io/gh/MalloyTheDev/CorpusStudio/branch/main/graph/badge.svg)](https://codecov.io/gh/MalloyTheDev/CorpusStudio)

**Corpus Studio** is a local-first, hardware-aware AI engineering platform with a mature dataset
studio at its center.

It is designed to be a one-stop shop for authoring, importing, cleaning, validating, splitting, versioning, and exporting model-ready datasets across multiple schemas:

- raw pretraining corpora
- instruction-tuning datasets
- chat/message datasets
- preference/DPO datasets
- code datasets
- image-caption datasets
- classification datasets
- retrieval/embedding datasets
- evaluation datasets
- reasoning-trace draft datasets

Corpus Studio is not just a JSONL editor. It is a writing-first dataset IDE plus a contract-first
run control plane covering the dataset-to-model workflow: create datasets, validate them,
clean and measure them, grade their outstanding debt, run pass/warn/block gates,
generate or rewrite candidates only with policy-approved providers under human
review, test and compare models, export them, version/diff/restore the dataset,
generate training configs, execute first-party QLoRA only from a hash-sealed RunPlan through the
supervised worker path, or launch your own installed trainer with live logs and checkpoints, plan against measured hardware,
isolate the reference training backend in a lock-pinned managed environment, track every run and the
model artifacts it produces, and measure the before/after improvement.

The single source of truth for what is implemented today is
[`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md).

## Current Status

Corpus Studio covers the full local loop from authoring, through governed
cleaning and gating, evaluation and model comparison, to launching and tracking
a training run — its own opt-in first-party backend through the sealed Platform lifecycle, or your
installed external trainer:

**Author & validate**
- create projects from built-in schema templates with pre-filled examples
- author and validate examples through the Python engine (required fields,
  types, list element types, enums, numeric bounds, nested object shapes,
  chat message structure) with selectable issue navigation
- preview/import JSONL, CSV, and TSV with failed-row quarantine, review, and retry
- full Unicode support end to end (CJK/Cyrillic/accented text round-trips
  correctly between the desktop and engine)

**Clean & measure**
- quality report: empty rows, exact + normalized duplicates, low-information
  rows, synthetic-pattern warnings with near-duplicate clustering,
  PII/secret detection (emails, SSNs, private keys, AWS/API keys, JWTs,
  Luhn-valid cards — masked samples), token-length outliers, and
  category-imbalance warnings, with project-level quality history
- leakage-checked train/validation/test splits (exact and near-duplicate rows
  shared across splits are reported before they inflate eval scores)
- export with an optional cleaning pass (dedupe / drop low-information) that
  writes a removal manifest; verbatim exports warn when duplicates remain
- export as JSONL (default, model-ready) or CSV/TSV for flat schemas (a
  chat/nested-object schema is refused rather than lossy-flattened)
- preference exports to DPO/KTO/reward with a pair-integrity gate
  (identical/empty/low-contrast pairs reported, `--drop-degenerate` opt-in)
- an inspectable dataset card summarizing metadata, schema, splits, quality,
  and the latest evaluation
- a graded **dataset debt** ledger: the quality signals normalized by dataset
  size, ranked by severity, and graded A–F so you know what to fix first
  (secrets/PII are graded by presence — a single leaked key is critical), each
  with a concrete remediation, surfaced in a desktop Debt tab whose grade
  invalidates the moment the dataset changes. See [`docs/DEBT.md`](docs/DEBT.md)

**Version & restore**
- durable dataset version history: capture the dataset's identity at a moment in
  time (a streaming content fingerprint + row count) with pinned links to the
  runs, artifacts, and evaluations from that state; live drift detection reports
  whether the current dataset still matches a version (matches / drifted /
  unreadable), and a live version card renders the lineage
- compare two versions (added / removed / common rows) and **restore** a
  version's exact rows. `examples.jsonl` now has a **single writer in the engine** —
  the `examples-append` command (#546); the WPF/Avalonia desktop that previously
  owned it is being retired (#545). In-place restore (capture an undo point, atomically
  swap in the restored rows, refuse without a safe undo) is being re-homed to the engine
  CLI next (#546). See [`docs/VERSIONING.md`](docs/VERSIONING.md)

**Govern & gate**
- role-based provider policy enforced **in the engine** (not just the UI):
  OpenAI/Anthropic are evaluator-only by default; local models (Ollama, local
  OpenAI-compatible servers) may generate trainable rows only when explicitly
  approved; OpenRouter is route-aware. Surfaced in a Settings panel. See
  [`docs/PROVIDER_POLICY.md`](docs/PROVIDER_POLICY.md)
- a gate runner producing serializable pass/warn/block reports over the
  existing schema, quality, leakage, and PII/secret logic; the export gate
  blocks on schema/PII failures. Surfaced by a Run Gates button. See
  [`docs/GATES.md`](docs/GATES.md)
- chat conversation-structure gate for chat datasets (`chat-gate`, plus a
  Run Chat Gates button): flags assistant-first, missing/dangling turns,
  back-to-back roles, misplaced system messages, turn-count bounds, and empty
  turns — advisory by default, escalatable to a block

**Evaluate & compare**
- Evaluation Studio runs against local Ollama or OpenAI-compatible endpoints with
  health checks, model discovery, report history, two-report comparison,
  regression reruns, tag/failure/score-band summaries, failed-row edit loops,
  manual scoring, and saved failure filters. The default automatic score is
  **keyword-overlap** recall — a lexical proxy, *not* a quality judgment; for a
  real quality signal use the opt-in **LLM-judge** scorer (`eval-run --judge-model`,
  also selectable per suite case) or manual scoring
- multi-model benchmark: run one dataset across several models and rank them,
  with per-model deltas and the examples every model failed
- Model Arena: run a prompt suite across several models side by side, with an
  optional evaluator-only judge that scores responses and picks a winner, and
  saved comparison reports
- Evaluation Suites: named, reusable multi-case suites (dataset × model × metric)
  with a per-metric verdict and optional dataset-`version_id`-pinned cases, run from
  the `suite-*` CLI or the desktop Suites tab. See
  [`docs/EVALUATION_STUDIO.md`](docs/EVALUATION_STUDIO.md)
- review-first AI Assist with a persistent accept/reject queue, saved
  views, bulk triage with undo, and resumable rewrite batches — every AI
  suggestion is review-required and never auto-accepted. AI-generated candidate
  rows are run through the dataset gate runner (schema/quality/PII) before review
  and carry a `candidate_gate` verdict — a pre-review signal only: a clean gate is
  not approval, a block does not auto-reject, and provider policy is enforced
  before generation

**Train & track**
- training config export for the first-party `corpus_studio` backend or axolotl /
  TRL / Unsloth / Hugging Face / LLaMA-Factory, with compatibility warnings, a real
  token budget (tokens-per-epoch after truncation, over-length counts), a rough VRAM
  planning estimate, and a LoRA rank/alpha suggestion. External targets include the exact launch
  command; the first-party target deliberately requires a sealed Platform plan instead
- an **opt-in first-party QLoRA backend** (the `[train]` extra): `platform-plan` binds immutable
  model/tokenizer/dataset/objective/environment/capability evidence, and `platform-run` supervises
  the exact worker configuration. Every execution receives a fresh UUIDv7 run ID and writes under
  `<output-root>/runs/<run-id>/`; its adapter ID includes the run, role, and weight-content hash.
  `train-check`, `train-merge`, and `model-fetch` remain supporting tools. The low-level `train-run`
  entry point is development-only: it refuses unless explicitly acknowledged and labels its result
  `UNSEALED_DIRECT_EXECUTION`, `NON_REPRODUCIBLE`, and `NO_PLATFORM_LINEAGE`
- versioned `TraceRecord` artifacts preserve source rows, role context, reasoning/tool/final-answer
  boundaries, producer-policy evidence, validation, and separate human review. Generated candidates
  are pending by default; `trace-review` writes immutable successors and first-party trainer admission refuses
  pending/rejected/tampered records before model loading (see
  [`docs/TRACE_RECORDS.md`](docs/TRACE_RECORDS.md))
- in-app Platform plan/run integration in the Tauri/React client. (The removed WPF/Avalonia desktop
  (#545) also launched reviewed external trainers with live logs + a process-tree Stop; that UX
  returns in the Tauri client. No UI head owns the first-party path — that is `platform-plan` ->
  `platform-run`, never a mutable-config bypass of Platform lineage
- checkpoint tracking during and after runs, resume-from-latest for targets
  with a CLI resume flag, and before/after evaluation comparison against the
  baseline captured at launch
- a durable training run registry: every run is recorded (argv, config, output
  dir, status, pid, checkpoints, before-eval link) under `training_runs/`, a
  force-closed run reconciles to `interrupted` on load, and a read-only run
  history browses past runs
- a durable model artifact registry: the adapters/checkpoints a run produced are
  tracked by referenced path (never moved), with path-integrity re-checked on
  load (`modified`/`missing` if the weights change on disk), a live weight card,
  and a promote gate that refuses to keep a modified/missing or regressed
  artifact

**Plan & isolate**
- a dependency-light, torch-free platform control plane that profiles hardware and storage, seals
  immutable run plans, predicts fit without claiming measured safety, supervises kill-able worker
  subprocesses, records measured fit, and preserves artifact lineage
- a 3-layer dependency architecture: lightweight control plane, opt-in capability packs, and isolated
  heavy backend workers. The `backend-corpus-studio` reference lifecycle discovers Python runtimes,
  previews exact no-shell install commands and sources, requires a matching plan hash, creates an owned
  venv, records an exact package/source/hash lock, separates CPU functional proof from GPU hardware
  proof, detects drift, and safely recreates only owned environments
- managed run plans pin the environment's immutable lock hash and dispatch through that interpreter;
  additional frameworks remain separate backend slices rather than being piled into one environment
  (see [`docs/ENVIRONMENT_MANAGER.md`](docs/ENVIRONMENT_MANAGER.md))
- static, offline `ModelDescriptor` / `TokenizerDescriptor` inspection records source pins, portable
  inventory/hashes, trust findings, component-scoped representations, and tokenizer compatibility.
  A hash-pinned allowlist adds structural Mixtral/Qwen2-MoE/DeepSeek V2/V3 expert-instance evidence
  while keeping runtime capability unverified; no model code is imported and no loadability/backend/
  fit/residency claim is made (see
  [`docs/MODEL_TOKENIZER_CONTRACTS.md`](docs/MODEL_TOKENIZER_CONTRACTS.md) and
  [`docs/MOE_ARCHITECTURE.md`](docs/MOE_ARCHITECTURE.md))
- a 29-entry, hash-sealed `TrainingObjective` registry describes datasets, labels, masks, losses,
  model/update/backend requirements, artifacts, resume/evaluation, and MoE-safe router/expert intent
  independently from backend implementation; its checker keeps declarations separate from measured
  capability evidence (see [`docs/TRAINING_SYSTEMS_ARCHITECTURE.md`](docs/TRAINING_SYSTEMS_ARCHITECTURE.md))
- a hash-sealed, MoE-safe parameter-accounting boundary keeps logical, active, resident, touched,
  updated, and exposed coordinates distinct; static descriptor/safetensors evidence and typed runtime
  reconciliation surface explicit gaps/conflicts without converting storage elements or allocator
  bytes into stronger claims (see [`docs/PARAMETER_ACCOUNTING.md`](docs/PARAMETER_ACCOUNTING.md))
- every new `RunPlan` seals an explicit physical-execution specification: concrete resources, state
  placements, offload rules, rank/group bindings, and exact parameter/storage evidence references.
  The current built-in workers prove only a single CPU or GPU resource and refuse non-trivial
  placement before launch; representation is not an execution claim (see
  [`docs/RUN_PLAN_PHYSICAL_EXECUTION.md`](docs/RUN_PLAN_PHYSICAL_EXECUTION.md))

Corpus Studio's dependency-light core never bundles CUDA, PyTorch, or trainer packages. Training deps
are opt-in via the `[train]` extra or installed into an explicitly reviewed managed reference-backend
environment; you can also launch your own installed trainer. It never hides the command it runs,
enforces who may generate trainable data, and does not publish datasets or auto-accept generated rows.

## License

MIT. See [`LICENSE`](LICENSE).

## Product principle

Every dataset example should be:

- valid
- inspectable
- traceable
- exportable
- versioned

## Repository Layout

```text
CorpusStudio
├── engine/     # Python: dependency-light control-plane engine (CLI) + opt-in [train] worker
├── apps/
│   └── web/        # Tauri 2 + React frontend — the UI (in progress)
├── schemas/    # Built-in dataset schema definitions
├── docs/       # Documentation (indexed by docs/README.md)
├── research/   # Opt-in IEEE native-Linux research overlay (supporting track)
├── examples/   # Example dataset rows
├── scripts/    # Developer scripts
├── tools/      # Developer utilities
├── assets/     # Branding / static assets
├── data/       # Local project data, ignored by git
├── output/     # Local run output, ignored by git
└── exports/    # Exported datasets, ignored by git
```

**Target architecture:** a **Rust authoritative core** + isolated Python ML workers, driven by the
**Tauri 2 / React** frontend over the Python engine. The WPF/Avalonia desktop is a **decommissioning
prototype** (#545), kept only until the engine CLI re-homes dataset authoring (#546); the Rust core is
a staged target (#522) and is not yet in the tree.

## Desktop preview

A walk through the workspace, front to back. An IDE-style activity bar toggles
between the **Start Center**, the file **Explorer**, and the **Studio**, with
**Problems** and **Output** panels docked at the bottom. See
[`docs/WORKSPACE_SYSTEM.md`](docs/WORKSPACE_SYSTEM.md).

### Design system (Nocturne)

The **Nocturne** design system — a quiet, dark-first look with a grouped workflow-phase sidebar
(**Overview · Author · Measure · Evaluate · Train**), Phosphor iconography, and a contextual quality
rail — is the framework-agnostic source of truth in [`docs/design/`](docs/design/) (tokens, icon set,
screen inventory). The removed WPF/Avalonia desktop prototype (#545) implemented it; the target
**Tauri 2/React** frontend (`apps/web`) carries the same design forward.
[`docs/WORKSPACE_SYSTEM.md`](docs/WORKSPACE_SYSTEM.md) describes the workspace behavior it specifies.

## Core Local Loop

Build a local desktop app that supports:

1. project creation
2. built-in schema templates
3. raw text, instruction, chat, and preference datasets
4. example authoring
5. schema validation
6. quality checks
7. train/validation/test split generation
8. JSONL export

## Development notes

The recommended stack is:

- Tauri 2 + React frontend (`apps/web`) — the target UI (a C# WPF/Avalonia desktop prototype is being retired, #545)
- Python dataset engine
- file-backed project state, with an optional SQLite index for fast project listing
- JSONL as the first export target
- Pydantic for schema validation
- Polars / DuckDB later for large datasets when needed

Tests: the Python engine has a pytest suite (with opt-in local Ollama
integration tests) that runs in CI (`.github/workflows/engine-tests.yml`); the
Tauri/React web client is built by `.github/workflows/web.yml`.

For what is implemented today, see [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md)
(the source of truth). For the product vision and staged roadmap, see
[`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md), [`docs/ROADMAP.md`](docs/ROADMAP.md),
and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

For hands-on setup, see [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md).
For every engine command (the desktop shells out to the same ones), see the
[`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md).
For copyable row formats, see [`docs/SCHEMA_SYSTEM.md`](docs/SCHEMA_SYSTEM.md) and
the per-schema reference in [`docs/schemas/`](docs/schemas/README.md).
For dataset card output, see [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md).
For provider generation policy and gates, see
[`docs/PROVIDER_POLICY.md`](docs/PROVIDER_POLICY.md) and
[`docs/GATES.md`](docs/GATES.md).
For dataset version history (capture/diff/restore) and the debt ledger, see
[`docs/VERSIONING.md`](docs/VERSIONING.md) and [`docs/DEBT.md`](docs/DEBT.md).
For Evaluation Studio, AI Assist, and training, see [`docs/EVALUATION_STUDIO.md`](docs/EVALUATION_STUDIO.md),
[`docs/AI_ASSIST.md`](docs/AI_ASSIST.md), and
[`docs/TRAINING.md`](docs/TRAINING.md) (config export, launcher architecture,
run tracking).
For dataset task walkthroughs, see [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md).
For public-release hygiene and known non-features, see
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).
