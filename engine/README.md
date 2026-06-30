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
