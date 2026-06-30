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
python -m corpus_studio.cli export ../examples/datasets/instruction/train.jsonl ../exports/instruction.jsonl instruction
```

The export and split commands validate rows before writing output.
