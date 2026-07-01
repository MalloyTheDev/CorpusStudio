# Training Lab

Training Lab is the Corpus Studio workspace for preparing training artifacts
now and eventually launching local fine-tuning jobs.

Training should come after dataset validation and evaluation. A training button
is not useful if the dataset is broken, leaky, duplicated, poorly split, or
untested.

Corpus Studio must not implement full training until dataset validation,
splitting, export, and Evaluation Lab workflows are stable.

## Why Training Comes Later

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

## Training Lab Phases

### v0.4 Config Generation

Corpus Studio should generate training config files for established tools. The
app should not run training yet.

Current MVP status: the Python engine has a `training-config` command and the
desktop app has a Training tab that writes a rendered config file under the
configured export directory. This is config export only.

Config generation should include:

- dataset path selection
- eval dataset path selection
- format compatibility checks
- sequence length
- adapter settings
- learning-rate defaults
- batch and accumulation hints
- warnings for missing splits or unsupported schemas

### v0.5 Local LoRA Launcher

Corpus Studio can later launch local LoRA or adapter jobs after config
generation is reliable.

Launcher scope should include:

- local command preview before launch
- training log viewer
- checkpoint tracking
- resume support
- stop/cancel behavior
- before/after eval comparison

### Later Full Training Orchestration

Full orchestration can include job queues, hardware profiles, multiple
experiments, dataset versions, artifact tracking, and richer comparison views.

## Supported Future Training Tools

Planned config targets:

- Axolotl
- TRL
- Unsloth
- Hugging Face Trainer
- LLaMA-Factory
- llama.cpp fine-tuning where applicable

Corpus Studio should generate tool-specific configs without embedding heavy ML
frameworks into the core app.

## Planned Features

- training config generation
- token budget estimate
- VRAM estimate
- LoRA parameter helper
- training log viewer
- checkpoint tracking
- resume training
- before/after eval comparison

## Current Boundary

The current app should stay focused on dataset creation, schema templates,
editors, validation, JSONL export, train/validation/test splitting, evaluation,
review-first AI assistance, and inspectable training config export. There
should be no CUDA, PyTorch, Transformers, trainer process launcher, checkpoint
manager, resume controller, or cloud-only requirement in the core app yet.
