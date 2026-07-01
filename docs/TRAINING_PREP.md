# Training Prep

Corpus Studio currently does not train models.

It prepares datasets and inspectable config files for training.

## Later training integrations

Possible future targets:

- Hugging Face Trainer
- TRL
- Axolotl
- Unsloth
- llama.cpp
- MLX
- custom local LoRA scripts

## Training-prep outputs

- cleaned dataset files
- train/validation/test split
- dataset card
- schema report
- quality report
- training config draft

The current engine and desktop app can export first-pass Training Lab config
files. They do not install CUDA, PyTorch, Transformers, or trainer-specific
packages, and they do not launch training processes.

## Rule

Training should only run after validation, quality checks, splitting, and
Evaluation Lab checks are stable.
