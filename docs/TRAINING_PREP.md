# Training Prep

Corpus Studio v0.1 does not train models.

It prepares datasets for training.

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

## Rule

Training should only run after validation and quality checks pass.
