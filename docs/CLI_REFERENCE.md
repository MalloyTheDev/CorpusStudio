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

The engine **runs the first-party QLoRA trainer in-process** (opt-in `[train]` extra) via `train-run`
— or generates an inspectable config to launch your own installed trainer (axolotl / TRL / Unsloth /
HF / LLaMA-Factory). The desktop can drive either path. The core stays dependency-light: torch is
never imported unless the `[train]` extra is installed and a training command is invoked. See
[TRAINING.md](TRAINING.md) and, for the supervised run lifecycle, [PLATFORM_RUN.md](PLATFORM_RUN.md).

| Command | What it does |
|---|---|
| `train-check [--json]` | Preflight the optional first-party training runtime (`[train]` extra): which deps are present, CUDA GPU + VRAM, and whether a real 4-bit QLoRA run — or only the CPU toy path — is possible. Reads only the Python env. |
| `dataset-tokens <dataset> --base-model … [--dataset-format chat\|instruction\|trace --seq-len N --sample N --json]` | Measure a dataset's token-length distribution and how many examples a given `sequence_len` would **truncate** (cutting the end — including the model's output). Run it BEFORE training so you never silently train on cut-off examples. Exit 3 when it truncates. |
| `trace-validate <dataset> [--json --show N]` | Validate a reasoning-**trace** corpus: each row must parse to a Trace (prompt/context + thinking + answer) and pass the structural checks (answer present, reasoning not a verbatim copy). Reports how many rows carry a real thinking trace. Exit 3 if any row is invalid. |
| `trace-generate <prompts> --model … [--backend ollama\|openai_compatible --base-url … --system … --limit N --show-thinking/--no-think --out …]` | **Create** reasoning traces from a prompt set via a local/OpenAI-compatible backend: each prompt is answered with a `<think>reasoning</think>answer`, self-filtered to keep only well-formed traces (drops malformed/no-answer generations). Provider policy is enforced (a teacher you can't train on is refused). Writes a trace JSONL you then `trace-validate`. |
| `train-run <config-path> [--dataset-path … --output-dir … --base-model … --cpu-toy --max-steps N --attn-implementation …]` | RUN the training in-process (first-party trainer): read a CorpusStudio config + dataset, build a TRL SFTTrainer + peft LoRA (4-bit QLoRA on GPU), train, save the adapter + tokenizer + checkpoints, and write `MODEL_CARD.md` next to the adapter. `--cpu-toy` runs a tiny CPU smoke path (no GPU). `--attn-implementation eager\|sdpa\|flash_attention_2` overrides the attention backend (auto: on **native-Windows** Blackwell/sm_120 the fused flash SDPA is disabled — it deadlocks the first backward under WDDM — and math SDPA is used; on **WSL/Linux** flash stays enabled). **Memory levers for a tight GPU:** `--optim paged_adamw_8bit` pages optimizer state to host RAM under pressure (spike-safe); `--use-liger` fuses the cross-entropy loss to drop the full-vocab logits memory spike at long `sequence_len` (needs `liger-kernel`; Blackwell support unverified); `--memory-efficient` is a shortcut enabling both. **Checkpoint retention:** `--save-steps N` (default 50) and `--save-total-limit N` (default 3 — keep only the N most recent checkpoints so a long run can't fill the disk; pass `0` to keep ALL). Progress `[step/total]`→stderr, JSON result→stdout; exit 2 if the runtime can't run it. |
| `train-merge <adapter-path> [--base-model … --output-dir … --strategy auto\|gpu\|cpu\|adapter-only]` | Merge a trained LoRA adapter into its base. A 7B fp16 merge (~14 GB) won't fit a 12 GB card, so `auto` tries GPU → CPU-offload → adapter-only (serve base+adapter unmerged). JSON result→stdout; exit 2 if every strategy fails. |
| `model-fetch <repo-id> [--local-dir … --revision … --allow '*.safetensors']` | RELIABLY download a base model from the Hugging Face Hub (resumable — survives dropped connections) and report its LICENSE (read from the downloaded card). Prefer MIT/Apache/permissive models — the base model's license governs what you can do with the trained result. JSON→stdout; exit 2 on failure. |
| `model-card <adapter-path> [--base-model … --config <training-config.json> --output …]` | Render the adapter's Markdown model card: base model (+ the reminder that ITS license governs the result), the LoRA hyper-parameters (from `adapter_config.json`), the training settings, and honesty notes. `train-run` writes this automatically; this regenerates it. Reads only local files. |
| `training-config <input.jsonl> <schema> --output-path … --base-model … [--target … --seed … --sequence-len … …]` | Generate an inspectable training config + token budget + VRAM estimate + **pre-flight** verdict. |
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
| `platform-probe [--cache] [--json] [--out DIR]` | Profile the host + run the **functional capability probes** (readiness = a kernel actually ran, not "the package imports"): emits an EnvironmentProfile + a CapabilityReport (per-probe `PASS / KERNEL_STALL / …`, `ready / cpu_toy_only / not_ready`). On **native-Windows** Blackwell (sm_120) the flash-attention probe short-circuits to `KERNEL_STALL` without executing (WDDM deadlock); on **WSL/Linux** it runs the flash backward and proves `sdpa`. `--cache` reuses an unchanged host's report. |
| `platform-backends [--json]` | List the registered training backends — "pick your framework" (`corpus_studio`, `unsloth`, …) — and their declared capabilities. |
| `platform-plan --base-model … --dataset … [--sequence-len N --backend corpus_studio\|unsloth --allow-cpu-toy --json --out DIR]` | Resolve an immutable, **hash-sealed RunPlan** — every ambiguous field decided against what PROVED to work on this host (bf16 only if proven, nf4 only if bitsandbytes passed, **math attention on native-Windows Blackwell / sdpa on WSL/Linux Blackwell**). The chosen backend is validated (on native-Windows sm_120 Unsloth is honestly refused; on WSL/Linux its sdpa runs). Prints the predicted **fit** (never `NATIVE_SAFE` from an estimate); `--json` emits `{run_plan, fit_classification}`. |
| `platform-run [PLAN.json \| --demo] [--runner echo\|cpu_toy\|training] [--subprocess --timeout S] [--max-steps N --out DIR]` | Execute a RunPlan through the headless supervisor: stream RunEvents to stderr, a RunManifest to stdout. `training` dispatches to the plan's backend. **`--subprocess`** runs it in a child process the parent can time out + KILL (a hung run → a real `KERNEL_STALL`; a crash is isolated). |
| `platform-profiles [--store DIR] [--json]` | List the cached host profiles (from `platform-probe --cache`). |
| `platform-storage [--path DIR --role R] [--json --out DIR]` | Characterize the host **storage** topology (mount / capacity / interface — NVMe/SATA/USB/network/virtual — non-destructively; no benchmark, no SMART read) and, for a `--path` + `--role` (e.g. `optimizer_offload`, `checkpoints`, `model_cache`, `python_env`), the **safe-spill suitability** verdict: refuses offload/checkpoint onto a USB bridge, a cloud-sync folder, a nearly-full disk, or inside the source repo. See [HARDWARE_STORAGE_PROFILE.md](HARDWARE_STORAGE_PROFILE.md). |
| `platform-schemas [--out DIR]` | Export the platform contracts as language-neutral JSON Schema (consumed by the Tauri/React client + the Rust core). |

---

*Generated from the engine CLI (`corpus-studio --help`, 62 commands). Run any command with
`--help` for its complete, authoritative option list — that is always the source of truth.*
