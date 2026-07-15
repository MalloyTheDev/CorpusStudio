# CLI Reference

The Corpus Studio engine is a dependency-light Python CLI. The desktop app shells out
to these same commands, so anything the app does you can also script.

**Invocation.** Installed (`pip install -e engine`): `corpus-studio <command> [args]`.
From source: `python -m corpus_studio.cli <command> [args]`. Every command supports
`--help` for its full option list; many emit JSON to **stdout** (parse it) and human
notes/progress to **stderr**.

**Conventions.** `--project-dir` / `<project_dir>` points at a dataset project folder
(the one holding `examples.jsonl`). Schemas are the built-in ids from `schemas`. Reports
are JSON-first. Live-backend commands (eval/suite/benchmark/ai-assist/backend-health)
make real calls to a running local model backend.

---

## Schemas & projects

| Command | What it does |
|---|---|
| `schemas` | List the built-in dataset schemas (id, fields). |
| `validate <file.jsonl> <schema>` | Validate a JSONL file against a built-in schema. |
| `new-project <id> <name> <schema>` | Create a local dataset project folder. |
| `project-list [--root <dir>]` | List local dataset projects (via the optional SQLite index). |
| `project-index-rebuild [--root <dir>]` | Rebuild the optional SQLite project index from `project.json` files. |

## Quality & debt

| Command | What it does |
|---|---|
| `quality <examples.jsonl>` | Build a basic quality report (empties, duplicates, low-information, first-pass synthetic patterns). |
| `dataset-debt <examples.jsonl> [--json]` | Summarize outstanding quality debt as a prioritized, graded (A–F) ledger. |

## Import (→ staging JSONL)

Corpus Studio is JSONL-canonical; tabular / Hub sources convert to a **staging JSONL**
that flows through the same preview → quarantine → commit path.

| Command | What it does |
|---|---|
| `import-preview <file.jsonl> <schema>` | Preview a JSONL import and report accepted / rejected rows. |
| `import-convert <file> <output.jsonl>` | Convert a CSV / TSV / **Parquet** file to a staging JSONL (Parquet needs the `[parquet]` extra). |
| `hf-inspect <dataset-id>` | Inspect a public Hugging Face dataset: configs/splits, columns, license. |
| `hf-import <dataset-id> …` | Import rows from a **public** Hugging Face dataset into a staging JSONL. |

## Splits

| Command | What it does |
|---|---|
| `split <input.jsonl> <output_dir> <schema> [--train-ratio 0.9 --validation-ratio 0.05 --seed 42]` | Validate + split JSONL into train / validation / test (deterministic seed; leakage-checked). |

## Evaluation & suites

All make **live backend calls**. The automatic score is keyword-overlap recall unless
`--judge-model` opts into an evaluator-model score (see [EVALUATION_LAB.md](EVALUATION_LAB.md)).

| Command | What it does |
|---|---|
| `eval-run <file.jsonl> <schema> --model … [--backend … --base-url … --judge-model … --judge-backend … --judge-base-url … --limit … --progress --reasoning]` | Run one Evaluation Lab pass; `--progress` streams `[k/N] evaluated` to stderr. **`--reasoning`** scores a reasoning model's **answer only** — the `<think>…</think>` block is stripped before scoring (the reasoning isn't the reference) while the full output is kept in the record; a "reasoning" model that emitted no reasoning is flagged. |
| `suite-init <name> [--project-dir … --force]` | Scaffold an example evaluation suite at `evaluation_suites/<name>.json` (`--force` overwrites an existing one). |
| `suite-list [--project-dir …] [--json]` | List the registered evaluation suites. |
| `suite-run <name-or-path> [--project-dir …] [--strict] [--json]` | Run a suite (each case = a live eval + gate); rolls up **per metric**. `--strict` exits 2 on a block. |
| `suite-history <name> [--project-dir …]` | Show a suite's run history (oldest → newest) for pass/warn/block trending. |
| `benchmark <file.jsonl> <schema> --model … (repeatable)` | Benchmark one dataset across several models and rank them. |
| `backend-health --backend … --model … [--base-url …]` | Check whether a configured model backend is reachable + the model is available. |
| `model-list --backend … [--base-url …]` | List models available from a configured local backend. |

## AI Assist

| Command | What it does |
|---|---|
| `ai-assist <draft.jsonl> <schema> --action … --model … [--backend … --base-url …]` | Run the AI Assist Lab on draft rows; returns **review-only** suggestions (never auto-saved). |

## Training

The engine executes the first-party QLoRA backend (opt-in `[train]` extra) through a sealed
`platform-plan` → `platform-run` chain, or generates an inspectable config for an installed external
trainer (axolotl / TRL / Unsloth / HF / LLaMA-Factory). The low-level `train-run` command is retained
only as an explicitly acknowledged, labeled development escape hatch; shipping clients do not use it.
The core stays dependency-light: torch is never imported until a worker invokes the optional stack. See
[TRAINING.md](TRAINING.md) and, for the supervised run lifecycle, [PLATFORM_RUN.md](PLATFORM_RUN.md).

| Command | What it does |
|---|---|
| `train-check [--json]` | Preflight the optional first-party training runtime (`[train]` extra): which deps are present, CUDA GPU + VRAM, and whether a real 4-bit QLoRA run — or only the CPU toy path — is possible. Reads only the Python env. |
| `dataset-tokens <dataset> --base-model … [--dataset-format chat\|instruction\|trace --seq-len N --sample N --json]` | Measure a dataset's token-length distribution and how many examples a given `sequence_len` would **truncate** (cutting the end — including the model's output). Run it BEFORE training so you never silently train on cut-off examples. Exit 3 when it truncates. |
| `trace-validate <dataset> [--json --show N --require-approved --project-dir …]` | Validate legacy trace rows and versioned `TraceRecord` rows. Verifies contract/hash, structure, current quality configuration, review counts, and answer leakage; `--require-approved` additionally enforces current external provider authority plus the trainer's approval/segment support gate. Provider authority loads from `--project-dir` or the dataset parent. Exit 3 on a block. |
| `trace-migrate <legacy.jsonl> --out <records.jsonl> [--source-ref …]` | Explicitly convert legacy prompt/thinking/answer rows into pending, hash-sealed `TraceRecord` rows with exact source-file and row identity. Never rewrites the input or `examples.jsonl`; exit 3 after writing if migrated records carry blocking quality evidence. |
| `trace-generate <prompts> --model … --out … [--backend ollama\|openai-compatible --project-dir … --base-url … --system … --limit N --report … --legacy-output]` | Generate self-filtered reasoning candidates. Resolves and authorizes the requested provider/model/route **before** constructing the backend, then separately reauthorizes the backend-reported model identity before preserving a candidate. Emits pending provenance-sealed records by default, binds response text/model/raw-response evidence in the response hash, preserves structured context, and writes an accepted/rejected attempt report. Input/output/report paths must be distinct. `--legacy-output` writes non-trainable flat rows plus a reviewable `<out>.trace-records.jsonl` sidecar. |
| `trace-review <records> --out … --reviewer … --decision approved\|rejected (--all\|--trace-id …) [--note … --project-dir …]` | Write immutable reviewed successors pinned to each predecessor hash. Approval recomputes engine validation and re-resolves requested/resolved models against the external project policy (`--project-dir` or input parent), including unknown-provider default deny and frontier restrictions. A blocking validation cannot be approved, and approval does not relabel generated reasoning as ground truth. |
| `train-run <config-path> --allow-unsealed-direct-execution […]` | **Development-only compatibility path.** It refuses by default because a mutable config and CLI overrides bypass RunPlan/backend/environment/input lineage. The explicit escape hatch runs the legacy in-process TRL/PEFT path and labels its JSON result `UNSEALED_DIRECT_EXECUTION`, `NON_REPRODUCIBLE`, and `NO_PLATFORM_LINEAGE`. `--save-steps`, `--save-total-limit`, and legacy step-checkpoint configs are refused until exact resume semantics exist. Shipping clients never invoke it; use `platform-plan` and `platform-run` for authoritative first-party execution. |
| `train-merge <adapter-path> [--base-model … --output-dir … --strategy auto\|gpu\|cpu\|adapter-only]` | Merge a trained LoRA adapter into its base. A 7B fp16 merge (~14 GB) won't fit a 12 GB card, so `auto` tries GPU → CPU-offload → adapter-only (serve base+adapter unmerged). JSON result→stdout; exit 2 if every strategy fails. |
| `model-fetch <repo-id> [--local-dir … --revision … --allow '*.safetensors']` | RELIABLY download a base model from the Hugging Face Hub (resumable — survives dropped connections) and report its LICENSE (read from the downloaded card). Prefer MIT/Apache/permissive models — the base model's license governs what you can do with the trained result. JSON→stdout; exit 2 on failure. |
| `model-card <adapter-path> [--base-model … --config <training-config.json> --output …]` | Render the adapter's Markdown model card: base model (+ the reminder that ITS license governs the result), the LoRA hyper-parameters (from `adapter_config.json`), the training settings, and honesty notes. Reads only local files. |
| `training-config <input.jsonl> <schema> --output-path … --base-model … [--target … --seed … --sequence-len … …]` | Generate an inspectable training config + token budget + VRAM estimate + **pre-flight** verdict. External targets include launch argv; `corpus_studio` intentionally emits no direct argv and requires a sealed Platform plan. |
| `training-compat --schema <id> --target <target> [--format …]` | Report training-config compatibility warnings without generating a config. |
| `training-checkpoints <output_dir> [--target …]` | List checkpoints in an output dir and build a resume command. |
| `run-provenance <project_dir> <config_path>` | Build a run's reproducibility manifest (dataset fingerprint + rows, config SHA-256, engine/platform). |
| `training-run-list <project_dir>` | List durable training run records (newest first; reconciles dead `running` records). |
| `training-run-update <project_dir> --run-id … [--status … --after-eval-path … --after-eval-model …]` | Headless status / eval-link update with transition validation. |
| `training-run-gate <project_dir> --run-id …` | Regression-gate a run using its linked before/after eval reports. |
| `training-eval-plan <project_dir> --run-id … [--json]` | Close the train→eval loop: print the ordered serve → eval → link → gate steps for a finished run. |

## Model artifacts

| Command | What it does |
|---|---|
| `artifact-register <project_dir> --run-id … --path …` | Register (idempotently) a model artifact a run produced (referenced, never moved). |
| `artifact-list <project_dir>` | List artifacts (newest first) with computed path integrity. |
| `artifact-update <project_dir> --artifact-id … --status candidate\|kept\|rejected` | Update an artifact's status (a transition to `kept` is promote-gated). |
| `artifact-card <project_dir> --artifact-id …` | Render a weight card (live projection; nothing stored). |
| `artifact-gate <project_dir> --artifact-id …` | Promote-gate an artifact (integrity + source-run regression) and save it. |

## Gates & provider policy

| Command | What it does |
|---|---|
| `gate-run <examples.jsonl> <schema>` | Run gates (schema/quality/leakage/PII/eval) → serializable pass/warn/block report. |
| `chat-gate <examples.jsonl>` | Gate a chat dataset's conversation structure (advisory; verdict in the report). |
| `provenance-gate <examples.jsonl> [--teacher-field meta.teacher] [--strict] [--allow-teacher …]` | Per-row provenance: read each row's `meta.teacher` and quarantine rows generated by a restricted provider (e.g. Anthropic/OpenAI) that can't be trained on. Licensing counterpart to `provider-policy`; trusts the declared teacher. |
| `gate-thresholds <project_dir>` | Show the effective gate thresholds for a project. |
| `gate-thresholds-set <project_dir> --values-json …` | Validate + write a project's `gate_thresholds.json`. |
| `provider-policy [--project-dir …]` | Show effective provider role policies (with project overrides). |
| `provider-approve <provider> <model> [--revoke] …` | Approve (or revoke) **trainable generation** for a specific local model/route. |

## Export, preference & dataset card

| Command | What it does |
|---|---|
| `export <input.jsonl> <output> <schema> [--format jsonl\|csv\|tsv\|parquet] [--dedupe --drop-low-information --redact-pii --check-provenance]` | Validate + export (optionally clean / mask PII / enforce provenance). `--check-provenance` BLOCKs the export (exit 2) if any row's `meta.teacher` is a restricted provider (`--provenance-strict` also blocks unknown; `--allow-teacher` clears one). CSV/TSV = flat schemas; JSONL/Parquet = all schemas. |
| `preference-export <input.jsonl> --output-path <out> [--format dpo\|kto\|reward] [--drop-degenerate]` | Export preference rows into a trainer-ready format (`--drop-degenerate` excludes empty/identical chosen-rejected pairs). |
| `dataset-card <project_dir> <schema>` | Build an inspectable dataset card from a project's existing artifacts. |

## Dataset versions (row-store)

| Command | What it does |
|---|---|
| `dataset-version-create <project_dir> …` | Capture a version: fingerprint + row count of `examples.jsonl` with pinned lineage. |
| `dataset-version-list <project_dir>` | List versions (newest first) with live integrity. |
| `dataset-version-show <project_dir> --version-id …` | Render a version card (live projection). |
| `dataset-version-diff <project_dir> --base-version-id … --other-version-id …` | Diff two versions by their stored row manifests (read-only). |
| `dataset-version-restore <project_dir> --version-id … --output …` | Reconstruct a version's exact rows from the row store (verified against the recorded fingerprint). |
| `dataset-version-gc <project_dir>` | Prune row-store rows no version references (fail-closed; never touches referenced rows). |

## Arena

| Command | What it does |
|---|---|
| `arena-run <prompts> --model … (repeatable) [--judge-model …]` | Run a prompt suite across several models and capture responses side by side. |

## Platform (headless run lifecycle)

The platform turns **goal + data + hardware** into a validated, reproducible run through
language-neutral contracts: profile the host → plan the run → predict the fit → run it → measure the
fit → account for the artifact. Dependency-light (torch is lazy-imported only by the runner). See
[PLATFORM_RUN.md](PLATFORM_RUN.md).

| Command | What it does |
|---|---|
| `platform-probe [--cache] [--json] [--out DIR]` | Profile the host + run the **functional capability probes**. `ready` requires one complete passing execution tuple (runtime/device, precision, quantization, adapter, attention, optimizer, loss, checkpoint, and export together), not package imports or a union of unrelated probes. The CapabilityReport embeds each probe's claims and tuple. On **native-Windows** Blackwell (sm_120) the standalone flash-attention probe short-circuits to `KERNEL_STALL` without executing (measured WDDM deadlock); it never upgrades another tuple. `--cache` reuses an unchanged host's report. |
| `platform-backends [--json]` | List registered backend manifests (`corpus_studio`, `unsloth`, …) and their declared capabilities. Registration is not execution support: only a backend with the complete required execution contract plus matching functional proof is admitted. |
| `platform-plan --base-model … --model-revision IMMUTABLE_COMMIT --dataset … [--tokenizer-revision COMMIT --sequence-len N --max-steps N\|--epochs N --allow-truncation --chat-template-sha256 HASH --backend ID --environment ID --manager-root DIR --allow-cpu-toy --json --out DIR]` | Resolve an immutable, **hash-sealed RunPlan** plus separately sealed `ResolvedExecutionConfiguration`. Hub inputs require immutable commits; local model/tokenizer directories and the dataset file are content-hashed. The configuration seals per-state precision, exact attention API/kernel/all SDP toggles, an explicit device map, semantic LoRA/trainer/checkpoint/data defaults, and the exact trainer interface, including `logging_strategy="steps"`, `logging_steps=1`, and `logging_nan_inf_filter=false`. New first-party plans explicitly disable intermediate checkpoints (`save_strategy="no"`, no cadence or retention); non-null checkpoint requests are refused until exact resume lineage exists. Admission requires one matching passing execution-combination probe plus trainer-surface proof; explicit overrides and independently passing axes cannot bypass it. A managed package claiming verified integrity must have complete positive RECORD/verified/installed-file counts. Unsloth currently lacks this Phase 9B declaration/proof and is refused on every host. `--environment` live-checks the managed lock and builds proof inside its interpreter. Prints the predicted **fit** (never `NATIVE_SAFE` from an estimate); `--json` emits `{run_plan, fit_classification}`. |
| `platform-run [PLAN.json \| --demo] [--runner auto\|echo\|cpu_toy\|training] [--subprocess --timeout S --preflight-timeout S] [--manager-root DIR --max-steps N --out DIR]` | Execute a RunPlan through the headless supervisor: stream RunEvents to stderr, a RunManifest to stdout. `auto` derives the only legal lane from the pinned backend; echo is restricted to explicit evaluation/demo plans. Both execution entry points recompute both seals before runner invocation. Explicit runner and `--max-steps` values are assertions only. Every invocation mints a fresh UUIDv7 `run_id`; resolved training writes to `<sealed-output-root>/runs/<run-id>/artifacts/adapter`. Success requires canonical trainable-state change, observed materialized-gradient coverage, a real optimizer, one finite loss per exact completed step, finite final tensors, exact trained-to-saved PEFT tensor/config identity, non-link output containment, independent weight/config integrity, durable records, and raw peak reconciliation before measured fit is proven. `--out` persists records under `<out>/runs/<run-id>/`. **`--subprocess`** launches a dedicated process group/session the parent can time out and terminate as a tree, while protocol 2.0 verifies backend/environment identity, run binding, ordering, required success evidence, and terminal lineage; it withholds the child terminal event until parent admission and persistence complete. `--preflight-timeout` gives resolved setup one non-extendable absolute budget (default 1800 seconds); bounded real progress is visible but cannot move it, and optimizer creation restores the ordinary `--timeout` silence rule. Heartbeats move neither deadline. Legacy plans remain readable but require regeneration for training. A managed plan also passes live health/drift/recipe-target checks and launches through its pinned interpreter. |
| `platform-profiles [--store DIR] [--json]` | List the cached host profiles (from `platform-probe --cache`). |
| `platform-storage [--path DIR --role R] [--diagnose "<error>"] [--recommend] [--json --out DIR]` | Characterize the host **storage** topology (mount / capacity / interface — NVMe/SATA/USB/network/virtual — non-destructively; no benchmark, no SMART read) and, for a `--path` + `--role` (`optimizer_offload`, `checkpoints`, `model_cache`, `dataset_cache`, `source_repo`, `python_env`, …), the **safe-spill suitability** verdict: refuses offload/checkpoint onto a USB bridge, a cloud-sync folder, a nearly-full disk, or inside the source repo, and flags a USB / WSL-`/mnt` home for the model cache / dataset / repo / venv (small-file & load-latency stalls). `--diagnose "<error>"` triages a failure (storage-implicated vs a VRAM/kernel failure); `--recommend` prints the per-role storage tier. See [HARDWARE_STORAGE_PROFILE.md](HARDWARE_STORAGE_PROFILE.md). |
| `model-inspect PATH [--model-id ID --tokenizer PATH --tokenizer-id ID --repository OWNER/REPO --requested-revision REV --resolved-commit HEX --tokenizer-repository OWNER/REPO --tokenizer-requested-revision REV --tokenizer-resolved-commit HEX --hash-weights --parameter-accounting --out DIR --json]` | Statically inspect a **local** model snapshot and optional tokenizer into versioned `ModelDescriptor` / `TokenizerDescriptor` records. Offline and dependency-light: no Hub fetch, torch/transformers/tokenizers import, link traversal, or repository-code execution; `trust_remote_code` is always false. A separate tokenizer directory never silently inherits the model repository identity; tokenizer source options are explicit (the same directory can inherit the same source evidence). Metadata/code files are hashed; large weights are streamed only with `--hash-weights`. A hash-pinned allowlist parses Mixtral, Qwen2-MoE, DeepSeek V2, and DeepSeek V3 config topology into structural expert-instance counts; malformed/unsupported metadata stays unknown, runtime capability stays unverified, and no load/backend/fit/residency claim is made. `--parameter-accounting` also writes a sealed static evidence report; safetensors elements become exact logical coordinates only with a matching content-pinned inventory, resolved handling, and a corroborating declaration. See [MODEL_TOKENIZER_CONTRACTS.md](MODEL_TOKENIZER_CONTRACTS.md), [MOE_MODEL_INSPECTION.md](MOE_MODEL_INSPECTION.md), and [PARAMETER_ACCOUNTING.md](PARAMETER_ACCOUNTING.md). |
| `parameter-account INPUT [--snapshot DIR --events JSONL --profile training_runtime\|inference_runtime\|checkpoint\|evaluation --report-id ID --artifact-ref ID@SHA256 --evaluation-ref ID@SHA256 --out FILE --json]` | Build a `ParameterAccountingReport` from a saved `ModelDescriptor`, verify an existing report, or reconcile typed `RunEvent.metrics.parameter_observations`. Dynamic evidence stays incomplete unless it is complete, exact, measured, hash-pinned, and run-anchored; allocator bytes are never converted into resident coordinates. Checkpoint/evaluation lineage refs must be hash-pinned. Output writes are atomic. See [PARAMETER_ACCOUNTING.md](PARAMETER_ACCOUNTING.md). |
| `training-objectives [OBJECTIVE_ID] [--json]` | List the 29 versioned, hash-sealed objective definitions or show one complete `TrainingObjective`. Definitions cover dataset fields, labels, masks, separately keyed losses, model/update/backend requirements, artifacts, resume/eval/hardware implications, limitations, and verification; registry presence is not backend support. See [TRAINING_OBJECTIVES.md](TRAINING_OBJECTIVES.md). |
| `training-objective-check OBJECTIVE_ID [--schema ID --schema-version VERSION --fields name:type,... --model-descriptor FILE --backend ID --capability-report FILE --json]` | Emit independent dataset/model/backend compatibility axes. Built-in schema versions and fields load automatically; custom/planned shapes require explicit version and field evidence. Multi-input objectives remain unverified until role-keyed evidence is supported. A static manifest can earn only `declared_compatible`; an exact backend-version match plus explicit effective objective capability evidence is required for `verified_compatible`. This is not a fit prediction. |
| `platform-schemas [--out DIR]` | Export the platform contracts as language-neutral JSON Schema (consumed by the Tauri/React client + the Rust core). |

**Environment Manager** — the 3-layer dependency model (control plane / capability profiles / isolated per-backend worker envs). See [ENVIRONMENT_MANAGER.md](ENVIRONMENT_MANAGER.md).

| Command | What it does |
|---|---|
| `env-recipes [--layer control_plane\|capability\|backend_worker] [--json]` | List the built-in **environment recipes** (declarations of what to install): the control-plane core, capability profiles, and isolated backend workers. `backend-corpus-studio-readiness-v2` is the exact-pinned math QLoRA readiness recipe; `backend-corpus-studio-readiness-flash-v1` is the exact-pinned forced-flash QLoRA readiness recipe. A recipe's declaration is not instance evidence; the current host has separate preserved math and flash tuple evidence, while the manager-1.1 flash instance requires replacement before a manager-1.2 health claim. |
| `env-runtimes [--recipe backend-corpus-studio] [--json]` | Discover and bounded-probe Python runtimes, recording version, implementation, architecture, platform, venv support, and recipe compatibility without creating anything. |
| `env-plan <recipe-id> [--env-id ID --runtime PYTHON --accelerator TAG --python 3.12 --manager-root DIR --worker-wheel FILE --out FILE --json]` | **Preview** a concrete environment without creating it. The hash seals exact argv, indexes, target, recipe/probe tuple, and (for readiness recipes) the worker wheel bytes. `--out` writes the canonical JSON plan. |
| `env-create [recipe-id] --confirm HASH [--env-id ID --runtime PYTHON --accelerator TAG --manager-root DIR --worker-wheel FILE --json]` | Create only when `HASH` matches the freshly resolved plan. Capture package/source and installed-file evidence, run required probes, and seal the lock only after they pass. Readiness recipes perform network installation and a bounded complete GPU QLoRA probe (math or forced-flash); each requires separate authorization. |
| `env-status [ID] [--refresh] [--manager-root DIR --json]` | Show durable descriptors and health. `--refresh` performs live lock, import, functional, hardware, and drift checks. |
| `env-probe ID [--manager-root DIR --json]` | Re-run all live health and drift checks. Importability, CPU functional proof, and GPU hardware proof remain distinct states. |
| `env-lock ID [--manager-root DIR]` | Print the immutable installed-package, source, metadata-hash, Python, torch/CUDA, recipe, and manager lock. |
| `env-remove ID --confirm ID [--manager-root DIR]` | Remove only a contained environment with a matching manager ownership marker; keep registry evidence and reserve the logical ID against silent reuse. |
| `env-recreate [recipe-id] --confirm HASH --confirm-remove ID [--env-id ID --runtime PYTHON --accelerator TAG --manager-root DIR --worker-wheel FILE --json]` | Recover an unsealed failed attempt from a newly reviewed plan. A sealed identity cannot be recreated; create its blue/green replacement under a new ID. No hidden repair or alternate-source retry occurs. |

---

*Cross-checked against the engine CLI (`corpus-studio --help`). Run any command with
`--help` for its complete, authoritative option list — that is always the source of truth.*
