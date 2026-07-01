# Training Configs

Training config export is the v0.4 Training Lab feature for turning a clean,
split, evaluated dataset into config files for established training tools.

Current MVP status: Corpus Studio can generate a first-pass config from the
engine `training-config` command and from the desktop Training tab. The export
path writes an inspectable file and returns a JSON summary; it does not launch
training.

Corpus Studio should generate configs before it launches trainers. This keeps
v0.4 safer and gives users inspectable files they can run manually.

## Config Targets

Planned targets:

- Axolotl YAML
- TRL Python/JSON config
- Unsloth notebook/script config
- Hugging Face Trainer config
- LLaMA-Factory config

Each target should declare which dataset schemas and export formats it supports.

## Shared Config Inputs

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

## Example Pseudo YAML

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
```

## Generation Rules

Config generation should:

- use explicit dataset paths
- include eval paths when available
- warn when no validation split exists
- preserve schema and export format metadata
- avoid hardware-specific claims unless measured or configured
- avoid hidden defaults that change training behavior dramatically

## Current Non-Goals

The current app should not add a trainer process launcher, CUDA dependency,
PyTorch, Transformers, or tool-specific package dependencies. Config export
should stay lightweight, inspectable, and safe to run without a GPU.

Near-term hardening should add target-specific compatibility warnings, clearer
dataset/split path selection, and dataset-card context. It should not start
training processes.
