# Training

Everything about preparing, configuring, and launching training in Corpus
Studio. Consolidated from the former TRAINING_LAB, TRAINING_CONFIGS,
TRAINING_PREP, and TRAINING_LAUNCHER_DESIGN docs. The dependency-light core never
bundles CUDA/PyTorch; training is opt-in — either Corpus Studio's own first-party
QLoRA trainer (the `[train]` extra) or the user's installed external trainer.

A first-party run is now driven by the **platform run lifecycle** (profile → plan → predict-fit → run
→ measure-fit → artifacts) with a multi-backend "pick your framework" registry (`corpus_studio`,
`unsloth`) and a supervised subprocess worker that can kill a hung run — see
[PLATFORM_RUN.md](PLATFORM_RUN.md) and the `platform-*` commands in
[CLI_REFERENCE.md](CLI_REFERENCE.md).


---

## Training Lab

Training Lab is the Corpus Studio workspace for preparing training artifacts
now and eventually launching local fine-tuning jobs.

Training should come after dataset validation and evaluation. A training button
is not useful if the dataset is broken, leaky, duplicated, poorly split, or
untested.

Corpus Studio must not implement full training until dataset validation,
splitting, export, and Evaluation Lab workflows are stable.

### Why Training Comes Later

Training consumes time, disk, VRAM, and user attention. Bad datasets create bad
models faster than good tools can rescue them. Corpus Studio should first make
datasets valid, inspectable, split correctly, exported cleanly, and evaluated
against models.

The staged order is:

1. create and validate datasets
2. split and export datasets
3. evaluate examples against models
4. improve weak examples
5. generate training configs
6. launch local training jobs
7. compare checkpoints against the same eval set

### Training Lab Phases

#### v0.4 Config Generation

Corpus Studio generates training config files for established tools and then
launches the user's installed trainer with that config.

Current status: the Python engine has a `training-config` command and the
desktop Training tab writes a rendered config file under the configured export
directory, **then launches the trainer** with the generated command (after a
confirm) — the first-party `corpus_studio` trainer, or the external trainer you
already have — streaming its logs live, listing checkpoints, recording each run,
supporting resume-from-checkpoint, and running a post-training regression gate.
The dependency-light core bundles no CUDA/PyTorch/Transformers; those are opt-in
via the `[train]` extra (which delegates to TRL/peft rather than implementing a
raw loop). (Config-export came first; the launcher followed — see below.)

Config generation should include:

- dataset path selection
- eval dataset path selection
- format compatibility checks
- sequence length
- adapter settings
- learning-rate defaults
- batch and accumulation hints
- warnings for missing splits or unsupported schemas

#### v0.5 Local LoRA Launcher

Corpus Studio can later launch local LoRA or adapter jobs after config
generation is reliable.

Launcher scope should include:

- local command preview before launch
- training log viewer
- checkpoint tracking
- resume support
- stop/cancel behavior
- before/after eval comparison

#### Later Full Training Orchestration

Full orchestration can include job queues, hardware profiles, multiple
experiments, dataset versions, artifact tracking, and richer comparison views.

### Training backends

Registered training-backend manifests (selectable for admission via `platform-plan --backend`):

- **`corpus_studio`** — the first-party HF + TRL + PEFT QLoRA trainer (LoRA/QLoRA; math/eager/sdpa/
  flash attention). New plans require its exact Phase 9B execution-contract declaration and matching
  functional proof.
- **`unsloth`** — a registered accelerated 4-bit QLoRA manifest, not currently executable through new
  Phase 9B plans. It does not yet declare/prove the complete effective execution contract, so it is
  refused on every host. Native-Windows Blackwell/sm_120 also requires a math path Unsloth does not
  provide. WSL evidence does not verify bare Linux.

Planned additional config targets (external-launcher / config-generation): Axolotl, Hugging Face
Trainer, LLaMA-Factory, llama.cpp fine-tuning where applicable. Corpus Studio generates tool-specific
configs without embedding heavy ML frameworks into the core.

### Planned Features

- training config generation
- token budget estimate
- VRAM estimate
- LoRA parameter helper
- training log viewer
- checkpoint tracking
- resume training
- before/after eval comparison

### Current Boundary

The engine **core** stays dependency-light: dataset creation, schema templates,
editors, validation, JSONL export, train/validation/test splitting, evaluation,
review-first AI assistance, and inspectable training config export need **no**
CUDA, PyTorch, or Transformers, so the core installs and runs everywhere.
First-party training is **opt-in**: an isolated worker environment with the
`[train]` runtime adds the real QLoRA trainer, adapter merge, and model download
(see "First-party training" below). The dependency-light control plane still
imports no training framework. Without that runtime, none of those heavy deps
are pulled.


---

## Training Prep

Corpus Studio prepares datasets and inspectable config files for training. A
first-party QLoRA run goes through `platform-plan` -> sealed `RunPlan` ->
`platform-run` -> the opt-in `[train]` worker. The external config-export and
reviewed-launch path remains available for bring-your-own trainers.

### Later training integrations

Possible future targets:

- Hugging Face Trainer
- TRL
- Axolotl
- Unsloth
- llama.cpp
- MLX
- custom local LoRA scripts

### Training-prep outputs

- cleaned dataset files
- train/validation/test split
- dataset card
- schema report
- quality report
- training config draft

Two training paths exist. **(1) External:** export a config and launch the user's
own installed trainer after reviewing its argv (live logs, checkpoints, run
history, resume, regression gate); the core never pulls CUDA/PyTorch. **(2)
First-party:** generate a hash-sealed plan and dispatch it to the isolated
`corpus_studio` backend worker. These paths do not share execution authority.

## First-party training (opt-in `[train]` extra)

`pip install corpus-studio-engine[train]` adds `torch` / `transformers` / `peft` /
`trl` / `accelerate` / `datasets` / `bitsandbytes` (bitsandbytes is CUDA-only, so
it is skipped on macOS). The core stays dependency-light for everyone who does not
train. All heavy imports are lazy — importing the engine never pulls torch.

| Command | What it does |
|---|---|
| `train-check [--json]` | Preflight: which deps are present, the CUDA GPU + VRAM, and whether a real 4-bit QLoRA run — or only the CPU toy path — is possible. |
| `model-fetch <repo> [--local-dir …]` | Resumably download a base model from the HF Hub and report its **license** (read from the downloaded card). Prefer MIT/Apache/permissive models. |
| `platform-plan … --out <plan-dir>` | Resolve immutable inputs, capability/backend/environment evidence, trainer semantics, placement, and output policy into a hash-sealed `RunPlan`. |
| `platform-run <RunPlan.json> --subprocess --out <record-root>` | Dispatch the one runner authorized by the pinned backend. Each execution gets a fresh run ID and isolated final-adapter path. New first-party plans disable intermediate checkpoints. Setup has one absolute `--preflight-timeout`; optimizer execution uses the ordinary `--timeout` silence rule. |
| `train-run <config> --allow-unsealed-direct-execution …` | **Development-only compatibility escape hatch.** It refuses by default and labels an explicitly allowed result `UNSEALED_DIRECT_EXECUTION`, `NON_REPRODUCIBLE`, and `NO_PLATFORM_LINEAGE`. Intermediate checkpoint options are refused until exact resume exists. Shipping clients never invoke it. |
| `train-merge <adapter> [--strategy auto\|gpu\|cpu\|adapter-only]` | Merge the adapter into the base. A 7B fp16 merge (~14 GB) won't fit 12 GB, so `auto` walks GPU → CPU-offload → adapter-only. |
| `model-card <adapter> [--base-model … --config … --output …]` | (Re)render the adapter's Markdown model card - base model, the reminder that its license governs the result, LoRA hyperparameters, training settings, and honesty notes. |

**Authoritative first-party path:** `model-fetch Qwen/Qwen2.5-7B` -> create or
select immutable dataset/model/tokenizer/objective evidence -> `platform-plan` ->
review `RunPlan.json` -> `platform-run --subprocess` -> `train-merge <adapter>`.
The generated `RunManifest` and content-bound `ArtifactManifest` carry the run
lineage. See [`PLATFORM_RUN.md`](PLATFORM_RUN.md).

### First-party checkpoint and resume boundary

The first-party trainer does not yet implement exact resume semantics. New sealed plans therefore
bind `save_strategy="no"`, leave checkpoint cadence and retention unset, and omit unused TRL
`save_steps` / `save_total_limit` fields. Legacy step-checkpoint configurations remain parseable for
inspection, but both the runner and trainer refuse them before dataset or model work. If a worker
nevertheless reports an intermediate checkpoint, the run fails closed and preserves that failed-run
evidence. The final PEFT adapter remains the only planned training output.

This limitation is separate from checkpoint discovery and resume commands for supported external
trainers. Short first-party benchmark trials are checkpoint-free. Do not approve a first-party run
expected to exceed 30 minutes until checkpoint compatibility, source-run/checkpoint lineage, and exact
sealed resume semantics are implemented.

### Reasoning trace approval gate

The `trace` dataset format accepts legacy prompt/thinking/answer rows for compatibility and approved
versioned `TraceRecord` rows. Generated records are pending candidates, not ready-to-train data:

1. generate or explicitly migrate to a separate pending JSONL artifact;
2. inspect and write an approved successor with `trace-review`;
3. run `trace-validate --require-approved`;
4. bind the approved artifact and its content hash into a new `RunPlan`.

The first-party trainer admission path verifies record structure, canonical hash, recomputed engine
validation evidence, review decision, current project-local provider-policy authority/frontier
restrictions, and current segment supervision support **before**
runtime probing or model loading. It refuses pending, rejected, tampered, foreign-validator,
generated legacy-compatibility, tool-use, agent, verifier, and process-supervision records because
the first-party trainer currently implements only assistant-authored inline reasoning + final-answer
SFT rendering. Ordinary legacy rows remain usable but print an explicit warning that they carry no
versioned review provenance. See
[`TRACE_RECORDS.md`](TRACE_RECORDS.md).

### From the desktop

The WPF/Avalonia Training tab still exports and launches reviewed argv for
installed **external** trainers. For the `corpus_studio` target it deliberately
refuses the former mutable-config launch and directs the user to create and run a
sealed platform plan. Supporting actions such as runtime inspection and adapter
merge remain available, but they do not confer first-party execution authority.
The Tauri/React Platform surface is the current UI client for `platform-plan` and
`platform-run`.

### GPU notes: attention backend on Blackwell (RTX 50-series / sm_120)

On brand-new **Blackwell** GPUs (RTX 50-series, compute capability sm_120) the fused
**flash** SDPA attention kernel **deadlocks on the first backward pass — but only under the
native-Windows WDDM driver** — the training step hangs at 100% GPU util but ~55 W (a real step
pulls 150–250 W). Verified on a real RTX 5070: bitsandbytes 4-bit, the *mem-efficient* SDPA
kernel, and the *math* path all work — **only the fused flash kernel hangs, and only on native
Windows**. WSL2 testing on the 5070 showed the same flash kernel running ~1000× faster than the math
fallback. On the current native-Linux host, separate manager-1.2 research environments verified tiny
complete math and forced `SDPBackend.FLASH_ATTENTION` BF16/NF4/QLoRA tuples. Fresh matched 0.5B
sequence-256 attempts both verified model and post-adapter CUDA placement, forced their intended
attention kernel, and attached QLoRA. Both then failed before optimizer step 1 because production
checked a BF16 pre-accumulation autograd tensor as if it were the sealed FP32 materialized leaf
gradient. The verifier now checks post-accumulation `parameter.grad`, but that correction still needs
a new worker, new locks/plans, and separately approved hardware evidence. No real flash optimizer
step or sequence-4096 workload has passed. WSL evidence must not be reported as a native-Linux result.

CorpusStudio treats **WSL as its own platform** (`OperatingSystem.wsl`): it has separately measured
flash evidence but `wddm` memory-residency like Windows (it still spills - see below). A new platform
plan binds an exact attention API, effective kernel, and all three PyTorch SDPA toggles. The known
native-Windows + Blackwell hazard resolves to the math path; another host still needs a passing exact
execution-combination probe before that policy can be sealed. The managed flash tuple now supplies
environment-level evidence for this host, but the Phase 9B production path still needs a successful
eligible-hardware optimizer step. Bare-Linux RTX 5070 real-workload behavior and sequence length 4096
remain unverified.

The real ceiling (measured, and it is **not** an attention-kernel problem): on a **12 GB** card
the 7B 4-bit QLoRA training peak is ~10.8 GB @ `sequence_len` 1024 → 13.8 GB @ 2048, so above
~seq 1280 the run exceeds 12 GB. On Windows (and WSL2, which shares the same WDDM driver) the
driver then silently **spills to system RAM and thrashes over PCIe** — steps 10–25× slower and
*looks* frozen while crawling. Native-Linux over-VRAM behavior remains unverified here. The
pre-flight warns about the measured Windows/WSL risk.

**A faster attention kernel does NOT lift this ceiling.** We tested `flash_attention_2` (the
Dao flash-attn Blackwell wheel) against the real 7B on the actual sm_120 card: it is **faster**
but the peak is **identical to math** (13.8 GB @ seq2048). With **gradient checkpointing** the
peak is dominated by the checkpointed layer-boundary activations (linear in seq), not the
transient attention scores that flash/mem-efficient attention save — so a memory-efficient
kernel barely moves the peak. To fit **full-length rows on 12 GB**, reduce the memory that
actually dominates: keep `sequence_len` ≤ ~1280, shorten a long training system prompt (~700 →
~80 tokens — also the correct fine-tuning design; the student internalises the rules and doesn't
need the teacher preamble), or use a **smaller base** (Qwen2.5-3B fits full-length comfortably).
Attention is not a post-seal runtime override. Re-plan against capability evidence
for the desired backend; if the exact kernel policy was not proven, planning
refuses it.

Security: model loading uses `trust_remote_code=False` (a fetched repo can't execute
code), and `model-fetch` warns when a model ships only pickle (`.bin`) weights
instead of safetensors. Honesty: `train-check`'s verdict is a capability check, the
CPU-toy path proves the *pipeline* (not a trained model), the model card states a
completed run is not a quality signal, and license classification is fail-closed
(unknown/custom/non-commercial → restricted-until-verified).

### Rule

Training should only run after validation, quality checks, splitting, and
Evaluation Lab checks are stable.


---

## Training Configs

Training config export is the v0.4 Training Lab feature for turning a clean,
split, evaluated dataset into config files for established training tools.

Current status: Corpus Studio generates a config from the engine
`training-config` command and the desktop Training tab. The export path writes
an inspectable file and returns a JSON summary that now also includes a real
token budget, a rough VRAM planning estimate, a LoRA rank/alpha suggestion,
the exact per-target launch command, and a **pre-flight** verdict.

**Pre-flight (fail fast before a long run).** A trainer run can take hours and a lot
of GPU, so `training-config` runs cheap deterministic checks and returns a
`preflight` verdict (`pass` / `warn` / `block`): is the trainer command on `PATH`
(warn + which packages to install), are the config and referenced data files present
and non-empty (**block** if missing), does the dataset have a usable number of rows
(**block** if empty, warn if tiny), will rows be truncated at `sequence_len` (warn), and —
when an NVIDIA GPU is detected via `nvidia-smi` — does the config's most memory-efficient
(4-bit) VRAM estimate fit in the GPU's free memory, or is it **likely to OOM** (warn, with the
fix: smaller model / 4-bit / lower `sequence_len` or batch). The GPU probe is best-effort and
dependency-free — no `nvidia-smi` (CPU-only box, or generating the config to train elsewhere)
simply skips the OOM check. The desktop surfaces each check and **disables the Launch button when
a check blocks**, so a run that is certain to fail can't be started. It is a pre-flight, not a
guarantee — it never runs the trainer, so a green pre-flight means "nothing obviously wrong",
not "this run will succeed".

> **Token counts are an estimate by default, and the budget says which counter ran.**
> To stay dependency-light the engine picks the most exact counter available, in order,
> and reports it in the budget's `method` field so a figure is never shown as exact when
> it isn't:
> 1. the **target model's own tokenizer** — install `pip install corpus-studio-engine[model-tokenizer]`
>    (the light `tokenizers` library); `training-config` passes its `--base-model`, so the
>    budget is exact *for that model* when its tokenizer is on the Hub (`method: hf:<model>`);
> 2. **tiktoken** — `pip install corpus-studio-engine[tokenizer]`, exact BPE for the GPT-4
>    family (`method: tiktoken`);
> 3. a **Unicode-aware heuristic** (counts CJK/kana/Hangul directly, blends word/character
>    estimates otherwise) — good for *planning*, not exact (`method: heuristic`).
>
> Each tier falls silently to the next on any failure (library absent, no network,
> gated/unknown model). Treat the token budget and VRAM numbers as planning figures unless
> `method` names a real tokenizer.
>
> **Offline safety.** The model-specific tier is the only one that touches the network. Set
> `CORPUS_STUDIO_TOKENIZER_OFFLINE=1` (or the standard `HF_HUB_OFFLINE=1`) to skip it entirely —
> estimation stays fully offline and deterministic, so `training-config` can never stall on a
> slow/captive network while fetching a tokenizer (you trade the model-exact count for tiktoken /
> the heuristic).
>
> **Chat-aware counting.** A chat row (`messages` turns) is counted as each turn's content
> tokens *plus* the per-message role/turn markers and per-conversation BOS/EOS a chat template
> adds — so the budget doesn't under-count chat/instruction rows and under-predict truncation.
> That overhead is a dependency-free heuristic; exact per-model chat-template rendering (which
> needs `transformers`) is a follow-up. As of v0.5 the desktop can also launch the
user's installed trainer from that command (with explicit confirmation), stream
logs, stop it, track checkpoints, resume, and compare before/after evaluations —
see the [Training Launcher Design](#training-launcher-design-v05) section below.

**Reproducibility manifest.** Each run record (`training_runs/<run_id>.json`) now captures a
`provenance` manifest at run start: the engine's canonical **dataset fingerprint** + row count
(proves *which data* trained the model, independent of the path or a later edit), the config
**SHA-256** (proves *which config*, byte-for-byte), and the **engine version** + platform. Together
with the record's exact `argv`, `base_model`, dataset-version back-link, and before/after eval, that
is the auditable recipe behind a produced model, surfaced per run in the Training tab's run history.
The generated config now emits a fixed training **seed** (default 42, `--seed` to override), so weight
initialisation, data shuffling, and dropout are deterministic *by default* — and because the manifest
hashes the config, the seed is pinned along with the inputs. (Bit-exact reproducibility across
different hardware/library versions still depends on the trainer + CUDA/cuDNN determinism, which is
outside the config; the seed removes the run-to-run randomness the config controls.)

Configs are always generated before any launch, and the exact command is always
shown and confirmed first, so users keep inspectable files they can also run
manually.

### Close the train→eval loop

A finished run *produces* a model; the value comes from **evaluating** it against the
held-out set you baselined on, so the regression gate can compare before vs after. The
gate + linkage already exist (`training-run-update`, `training-run-gate`), but producing
the after-eval means **serving** the trained model — an external, format/stack-specific
step CorpusStudio deliberately does not automate. `training-eval-plan` closes that gap:

```
corpus-studio training-eval-plan <project_dir> --run-id <run_id> \
    [--eval-dataset held-out.jsonl] [--schema <id>] \
    [--backend ollama|openai-compatible] [--base-url <url>] [--served-model <name>]
```

It prints the ordered, run-specific recipe: **(1)** serve the produced model (external —
Ollama/vLLM/TGI, with a clearly-labelled example), **(2)** `eval-run` against the *same*
held-out set + schema + metric as the baseline (pre-filled from the run's before-eval when
omitted; writes `eval_reports/after-<run_id>.json`), **(3)** `training-run-update` to link
that after-eval + the served model to the run, **(4)** `training-run-gate` to compare
before/after and BLOCK a regressed promote. Honesty boundary: the serve step is a reminder,
not a guarantee (nothing here runs or serves a model); the eval/link/gate commands are exact
once served. Only a `succeeded` run is "ready" — you can't evaluate a model a run never produced.

The desktop **Training tab surfaces this plan in-app**: when a run finishes successfully (and on
each run-history refresh, for the newest run) the ordered steps + copy-pasteable commands appear
below the run history, so the one-stop-shop closes the loop without dropping to the CLI.

### Config Targets

Planned targets:

- Axolotl YAML
- TRL Python/JSON config
- Unsloth notebook/script config
- Hugging Face Trainer config
- LLaMA-Factory config

Each target should declare which dataset schemas and export formats it supports.

### Shared Config Inputs

The shared config model should capture:

- base model
- train dataset path
- eval dataset path
- dataset format
- sequence length
- adapter type
- LoRA rank and alpha
- micro batch size
- gradient accumulation
- learning rate
- output directory
- expected hardware profile

### Example Pseudo YAML

```yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
dataset_path: exports/coding_tutor_v0.1/train.jsonl
eval_dataset_path: exports/coding_tutor_v0.1/validation.jsonl
format: chat
sequence_len: 4096
adapter: lora
lora_r: 16
lora_alpha: 32
micro_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002
seed: 42
```

Each target's supported training styles are declared in
`engine/corpus_studio/training/compatibility.py`.

### Generation Rules

Config generation should:

- use explicit dataset paths
- include eval paths when available
- warn when no validation split exists
- preserve schema and export format metadata
- avoid hardware-specific claims unless measured or configured
- avoid hidden defaults that change training behavior dramatically

### Compatibility Warnings

The `training-config` command runs advisory schema/format/target checks and adds
the results to both the `warnings` and `compatibility_warnings` fields of its
JSON output. The checks are advisory only; they never block export. They flag:

- **Format vs schema mismatch** — a `--format` label that is unusual for the
  chosen schema (for example `sharegpt` on the `instruction` schema).
- **Preference datasets** — the `preference` schema needs a DPO/reward pipeline;
  targets that support it (Axolotl, TRL, Unsloth, LLaMA-Factory) are told to
  configure that trainer, and targets without a preference path are told a custom
  trainer is required. The LoRA template itself only renders SFT-shaped fields.
- **Non causal-LM schemas** — `image_caption`, `retrieval`, and `evaluation`
  data is not causal-LM fine-tuning data and needs a different trainer.
- **Classification / raw pretraining** — flagged when the chosen target does not
  express that training style with a LoRA causal-LM config.

### Non-Goals (for the dependency-light core)

The **core** must not require CUDA/PyTorch/Transformers or a trainer at import —
config export stays lightweight, inspectable, and safe to run without a GPU, and
those heavy deps live only in the opt-in `[train]` extra (loaded lazily, never at
import). The core is not itself a deep-learning framework: no distributed training,
no bespoke training loop — the `[train]` runtime delegates to TRL/peft/transformers.

Near-term hardening should improve target-specific config rendering depth,
clearer dataset/split path selection, and richer dataset-card context. It
should not start training processes.


---

## External Training Launcher Design (v0.5)

Scope and architecture for the Local Training Launcher. This is the biggest
shift in the app's life: from *generating inspectable training configs* to
*launching and observing real training runs*. The roadmap gated it behind the
hardening that is now done.

### Principles

- **Local-first.** No cloud orchestration by default.
- **Never hide the trainer command.** The user always sees exactly what runs.
- **No forced ML dependencies in the engine core.** The core is torch-free at import. The
  external-launcher path orchestrates the user's *installed* trainer (axolotl / TRL / Unsloth / HF /
  LLaMA-Factory) and never imports torch/CUDA. First-party training instead crosses the platform
  worker boundary; the worker imports torch/transformers/peft/trl only after plan and protocol
  validation. If the selected runtime is unavailable, the app shows an install/planning hint rather
  than silently changing lanes.
- **Launching is a big, machine-consuming action.** It requires explicit
  confirmation showing the exact command before every launch.

### The one architectural decision

For external tools, orchestrate the user's installed trainer CLI. The **engine**
produces the exact command (and resume variant) per target and discovers
checkpoints - pure string/path work, fully testable. The **desktop** spawns and
manages that external process. First-party execution is deliberately different:
the platform supervisor owns its isolated worker process, protocol, run ID,
lineage, and termination.

### The hard part: live log streaming

The desktop↔engine bridge (`RunEngineProcessAsync`) is request/response — it
runs a command to completion and reads all output with `ReadToEndAsync`. A
training run lasts minutes to hours and needs **live, line-by-line output**.
That is net-new infrastructure and where the risk concentrates:

1. **Streaming process runner** — spawn the trainer directly (not through the
   engine), read stdout/stderr incrementally, append to a bounded log buffer.
2. **Background job state machine** — a run must survive tab switches, update
   status without blocking the UI, and support cancel (process-tree kill,
   reusing the P2 cancellation work).
3. **App-close behavior** — the trainer is a child process. Closing the app
   kills it unless detached; the MVP kills on close and warns (detach/reattach
   is a large follow-up).

### Staged plan

- **v0.5.0 — guided command (done).** Engine emits the exact launch command per
  target plus the resume variant and dependencies (`launch` in the
  `training-config` output, copyable from the desktop); `training-checkpoints`
  lists checkpoints and builds a resume command for the latest. Nothing is
  executed. Dependency-free.
- **v0.5.1 — in-app launch + live log viewer + stop.** The streaming runner and
  job state machine. Engine adds an `argv` form of the command so the desktop
  spawns without shell parsing.
- **v0.5.2 — live checkpoint tracking + in-app resume (done).** Configs carry
  `output_dir`; the desktop refreshes the checkpoint list via the
  `training-checkpoints` CLI (slow poll during a run + on end/stop/error +
  manual), and "Resume latest" relaunches through the same confirmation for
  targets with a CLI resume flag. The directory is the persistence — no state
  file, latest-only resume.
- **v0.5.3 — before/after eval comparison (done).** The newest saved evaluation
  report is captured as the "before" baseline at launch. The app cannot serve
  the trained adapter itself, so the user loads it into their local backend and
  runs an evaluation; "Compare vs baseline" then reuses the existing two-report
  comparison (after − before deltas), with honest guidance for the
  no-baseline / no-after-eval cases.

### Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| No GPU / trainer not installed | Detect best-effort; degrade to the guided command; never fail silently. |
| App closed mid-run | MVP kills the child and warns; detach/reattach deferred. |
| Cross-platform (Windows-first; trainers often need Linux/WSL) | Command preview works everywhere; real launch may need WSL — detect and guide. |
| Runaway resource use | Explicit confirmation showing the exact command before every launch. |
| Corrupt/partial checkpoints | Checkpoint tracking is read-only/advisory; resume is opt-in. |
| Spawning arbitrary commands | Spawn a structured `argv` (no shell), from a command the engine generated and the user confirmed. |

### Testability

- Engine command/checkpoint logic is pure — unit-tested with crafted inputs.
- The streaming runner is tested against trivial deterministic processes
  (echo for streaming/exit, a long-lived process for stop/kill).
- Job-state and log-buffer logic live in the view-model and are tested without
  a UI thread; only the thin Dispatcher marshaling lives in code-behind.
