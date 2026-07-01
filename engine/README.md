# Corpus Studio Engine

Python dataset engine for Corpus Studio.

Responsibilities:

- load built-in schemas
- validate examples
- import datasets
- clean datasets
- split datasets
- export datasets
- produce quality reports
- run local Evaluation Lab MVP passes
- run review-first AI Assist Lab MVP passes
- check configured model backend health
- list configured backend models
- generate Training Lab config exports without launching trainers

## Development

```bash
cd engine
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## CLI

Run these commands from the `engine` directory after installing the package:

```bash
python -m corpus_studio.cli schemas
python -m corpus_studio.cli new-project example_instruction_project "Example Instruction Project" instruction
python -m corpus_studio.cli validate ../examples/datasets/instruction/train.jsonl instruction
python -m corpus_studio.cli quality ../examples/datasets/instruction/train.jsonl
python -m corpus_studio.cli split ../examples/datasets/instruction/train.jsonl ../exports/instruction_splits instruction
python -m corpus_studio.cli model-list --backend ollama
python -m corpus_studio.cli backend-health --backend ollama --model qwen2.5-coder:7b
python -m corpus_studio.cli eval-run ../examples/datasets/instruction/train.jsonl instruction --backend ollama --model qwen2.5-coder:7b --limit 5
python -m corpus_studio.cli ai-assist ../examples/datasets/instruction/train.jsonl instruction --action review --backend ollama --model qwen2.5-coder:7b
python -m corpus_studio.cli ai-assist ../examples/datasets/preference/train.jsonl preference --action judge-preference-strength --backend ollama --model qwen2.5-coder:7b
python -m corpus_studio.cli training-config ../examples/datasets/instruction/train.jsonl instruction --output-path ../exports/instruction_axolotl.yaml --base-model Qwen/Qwen2.5-Coder-7B-Instruct --target axolotl_yaml
python -m corpus_studio.cli export ../examples/datasets/instruction/train.jsonl ../exports/instruction.jsonl instruction
```

The model-backed commands require the selected local backend to already be
running. AI Assist results remain review-only and may include local warnings
for validation issues, repetitive synthetic patterns, and weak preference-pair
contrast. Evaluation reports include a `run_settings` object so the desktop app
can rerun saved configurations as regression checks. Tests use fakes and do not
call real endpoints.

The `training-config` command writes inspectable config files only. It does not
install CUDA, PyTorch, Transformers, or trainer packages, and it does not launch
training jobs.
