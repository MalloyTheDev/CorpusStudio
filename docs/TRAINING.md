# Training

Everything about preparing, configuring, and launching training in Corpus
Studio. Consolidated from the former TRAINING_LAB, TRAINING_CONFIGS,
TRAINING_PREP, and TRAINING_LAUNCHER_DESIGN docs. The dependency-light core never
bundles CUDA/PyTorch; training is opt-in — either Corpus Studio's own first-party
QLoRA trainer (the `[train]` extra) or the user's installed external trainer.


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

### Supported Future Training Tools

Planned config targets:

- Axolotl
- TRL
- Unsloth
- Hugging Face Trainer
- LLaMA-Factory
- llama.cpp fine-tuning where applicable

Corpus Studio should generate tool-specific configs without embedding heavy ML
frameworks into the core app.

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
First-party training is **opt-in**: installing the `[train]` extra adds a real
QLoRA trainer, adapter merge, and model download (see "First-party training"
below). Without the extra, none of those heavy deps are pulled.


---

## Training Prep

Corpus Studio prepares datasets and inspectable config files for training, **and**
— with the opt-in `[train]` extra — can run the QLoRA itself first-party
(`train-run` / `train-merge`, see "First-party training" below). The external
config-export path stays as an escape hatch for bring-your-own trainers.

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

Two training paths exist. **(1) External (default core):** export a config and
launch the user's own installed trainer (live logs, checkpoints, run history,
resume, regression gate) — the core never pulls CUDA/PyTorch. **(2) First-party
(opt-in `[train]` extra):** CorpusStudio runs the QLoRA itself — see below.

## First-party training (opt-in `[train]` extra)

`pip install corpus-studio-engine[train]` adds `torch` / `transformers` / `peft` /
`trl` / `accelerate` / `datasets` / `bitsandbytes` (bitsandbytes is CUDA-only, so
it is skipped on macOS). The core stays dependency-light for everyone who does not
train. All heavy imports are lazy — importing the engine never pulls torch.

| Command | What it does |
|---|---|
| `train-check [--json]` | Preflight: which deps are present, the CUDA GPU + VRAM, and whether a real 4-bit QLoRA run — or only the CPU toy path — is possible. |
| `model-fetch <repo> [--local-dir …]` | Resumably download a base model from the HF Hub and report its **license** (read from the downloaded card). Prefer MIT/Apache/permissive models. |
| `train-run <config> [--cpu-toy --max-steps N …]` | Run the QLoRA in-process (TRL `SFTTrainer` + peft LoRA, 4-bit on GPU); saves the adapter + tokenizer + checkpoints, and writes a `MODEL_CARD.md` next to the adapter. `--cpu-toy` is a tiny-model CPU smoke test, not a real train. |
| `train-merge <adapter> [--strategy auto\|gpu\|cpu\|adapter-only]` | Merge the adapter into the base. A 7B fp16 merge (~14 GB) won't fit 12 GB, so `auto` walks GPU → CPU-offload → adapter-only. |
| `model-card <adapter> [--base-model … --config … --output …]` | (Re)render the adapter's Markdown model card — base model + the reminder that ITS license governs the result, the LoRA hyper-parameters (from `adapter_config.json`), the training settings, and honesty notes. `train-run` already writes it; this regenerates it. |

**End-to-end (the config comes from the Training tab's `training-config`, target
`corpus_studio`):** `model-fetch Qwen/Qwen2.5-7B` → `train-check` → `train-run
<config>` → `train-merge <adapter>` → the adapter's `MODEL_CARD.md` documents the run.

### From the desktop

The **Training tab** runs all of this for you when the target is `corpus_studio`:
**Check runtime** (train-check) reports whether a real GPU QLoRA — or only the CPU
toy path — can run; **Launch** preflights and streams `[step/total]` progress into
the live log (a real GPU run needs `ready`; tick **CPU toy** for the smoke path);
**Merge adapter** runs `train-merge` with the small-VRAM fallback. The launcher runs
the trainer with the engine's own interpreter, so the run always matches what
train-check probed.

### GPU notes: attention backend on Blackwell (RTX 50-series / sm_120)

On brand-new **Blackwell** GPUs (RTX 50-series, compute capability sm_120) the fused
**flash / memory-efficient SDPA** attention kernels **deadlock on the first backward
pass** in current PyTorch — the training step hangs at 100% GPU util but ~55 W (a real
step pulls 150–250 W). bitsandbytes 4-bit and the *math* attention path are both fine;
only the fused kernels hang. `train-run` **detects sm_120 and automatically disables the
fused SDPA backends**, falling back to the sm_120-safe math SDPA kernel (`train-check`
notes this when it sees a Blackwell card). Trade-off: **math attention uses more VRAM
than flash**, so a long `sequence_len` is tighter on a 12 GB card — lower `sequence_len`
if you OOM. Override the backend explicitly with `train-run --attn-implementation
eager|sdpa|flash_attention_2` (or `attn_implementation` in the config).

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

## Training Launcher Design (v0.5)

Scope and architecture for the Local Training Launcher. This is the biggest
shift in the app's life: from *generating inspectable training configs* to
*launching and observing real training runs*. The roadmap gated it behind the
hardening that is now done.

### Principles

- **Local-first.** No cloud orchestration by default.
- **Never hide the trainer command.** The user always sees exactly what runs.
- **No forced ML dependencies in the engine.** Corpus Studio orchestrates the
  user's *installed* trainer (axolotl / TRL / Unsloth / Hugging Face /
  LLaMA-Factory). It never imports torch/CUDA. If the trainer is not installed,
  the app shows the command to run rather than failing silently.
- **Launching is a big, machine-consuming action.** It requires explicit
  confirmation showing the exact command before every launch.

### The one architectural decision

Orchestrate the user's installed trainer CLI. The **engine** produces the exact
command (and resume variant) per target and discovers checkpoints — pure
string/path work, fully testable. The **desktop** spawns and manages the
process. This keeps the engine dependency-free and puts OS process management
where it belongs.

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
