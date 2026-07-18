# Development Setup

## Requirements

- Python 3.11+
- .NET 8 SDK
- Visual Studio 2022 (Windows WPF) or JetBrains Rider / the .NET SDK (cross-platform Avalonia)
- Node.js/npm and Rust/Tauri prerequisites for `apps/web`
- GitHub CLI, if creating the upstream GitHub repository from this checkout

## Current native-Linux host quick start

Work from the active checkout, not the historical Windows mounts:

```bash
cd /mnt/training-nvme/repos/CorpusStudio
cp .env.example .env
cd engine
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp
cd ..
dotnet build apps/desktop/CorpusStudio.Avalonia/CorpusStudio.Avalonia.csproj
cd apps/web
npm ci
npm run build
```

`CorpusStudio.Desktop` is the WPF head and is Windows-only. On Linux, build or run the Avalonia head;
the Tauri/React client lives under `apps/web`.

## GitHub setup

This checkout is initialized as a local Git repository on the `main` branch.
The upstream repository is:

```text
https://github.com/MalloyTheDev/CorpusStudio.git
```

Clone or connect to the public repository with:

```powershell
git remote add origin https://github.com/MalloyTheDev/CorpusStudio.git
```

If `origin` is already configured, verify it with:

```powershell
git remote -v
```

## Local environment file (Windows PowerShell)

Copy the example environment file for local defaults:

```powershell
Copy-Item .env.example .env
```

Do not put secrets or private dataset paths in committed files.

## Python engine setup (Windows PowerShell)

From the repository root:

```powershell
cd engine
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

## Desktop setup (Windows WPF only)

From the repository root:

```powershell
dotnet build apps\desktop\CorpusStudio.Desktop.sln
.\apps\desktop\CorpusStudio.Desktop\bin\Debug\net8.0-windows\CorpusStudio.Desktop.exe
```

Run the desktop unit tests (project-local JSON persistence logic — reviewed
fixes, rewrite batches, and saved failure filters):

```powershell
dotnet test apps\desktop\CorpusStudio.Desktop.Tests\CorpusStudio.Desktop.Tests.csproj
```

The desktop app uses the local Python engine. If the app cannot find Python or the engine, confirm `.env` contains:

```text
CORPUS_STUDIO_DATA_DIR=data/projects
CORPUS_STUDIO_EXPORT_DIR=exports
CORPUS_STUDIO_ENGINE_DIR=engine
```

## v0.1 desktop smoke test (Windows WPF only)

Use this checklist after a build:

1. Launch the desktop app.
2. Click **New Dataset Project**.
3. Enter a project name and choose one of the v0.1 schemas.
4. Click **Create**.
5. In **Writing Studio**, keep or edit the generated JSON example.
6. Click **Validate** and confirm the validation panel reports a valid row.
7. Click **Save Example** and confirm the row is appended to the active project.
8. Click **Run Quality** and confirm the quality panel reports the saved example count.
9. Open **Splits** and click **Generate Splits**.
10. Confirm `train.jsonl`, `validation.jsonl`, and `test.jsonl` appear under `exports/<project_id>/splits`.
11. Click **Export JSONL** and confirm an export appears under `exports/<project_id>/export.jsonl`.
12. Open **Settings** and confirm the repository, engine, Python, project, and export paths point to this checkout.

Or run the automated Windows-only WPF desktop smoke test:

```powershell
.\scripts\smoke_desktop_examples.ps1
```

The smoke script exercises the current desktop loop, including project
creation, validation, quality checks, splits, imports, Evaluation Studio shelling,
Evaluation report comparison, saved Evaluation regression rerun settings, AI
Assist queue behavior, persistent AI Assist rewrite batch resume, Evaluation
tag/failure/score-band summaries, failed Evaluation row edit handoff to Writing
Studio, Training Studio config export, and saved lab backend settings.

## Validate example datasets

On the current native-Linux host (venv `engine/.venv/bin/python`):

```bash
engine/.venv/bin/python -m corpus_studio.cli validate examples/datasets/instruction/train.jsonl instruction
engine/.venv/bin/python scripts/validate_examples.py
```

Windows (PowerShell):

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli validate examples\datasets\instruction\train.jsonl instruction
.\engine\.venv\Scripts\python.exe scripts\validate_examples.py
```

## Useful engine commands (Windows PowerShell examples)

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli schemas
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli new-project demo "Demo Dataset" instruction
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli quality examples\datasets\instruction\train.jsonl
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli split examples\datasets\instruction\train.jsonl exports\instruction_split instruction
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli export examples\datasets\instruction\train.jsonl exports\instruction.jsonl instruction
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli project-index-rebuild
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli project-list --schema instruction
```

`project-list` and `project-index-rebuild` use an optional SQLite index
(`data/projects/index.sqlite3`) for fast listing/filtering. The index is built
on first use, can be rebuilt from disk at any time, and never replaces the
inspectable `project.json`/`examples.jsonl` files.

## Local model-backed commands (Windows PowerShell examples)

These commands require Ollama, LM Studio, or another compatible local backend to
already be running. They do not pull models or install ML packages. (First-party
training is a separate **opt-in** feature - build the managed `[train]` worker and
use `platform-plan` / `platform-run`; `train-check`, `train-merge`, and
`model-fetch` remain supporting tools. See `CLI_REFERENCE.md` and `TRAINING.md`.)

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli model-list --backend ollama
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli backend-health --backend ollama --model qwen2.5-coder:7b
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli eval-run examples\datasets\instruction\train.jsonl instruction --backend ollama --model qwen2.5-coder:7b --limit 5
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli ai-assist examples\datasets\instruction\train.jsonl instruction --action review --backend ollama --model qwen2.5-coder:7b
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli training-compat --schema preference --target trl_config
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli preference-export examples\datasets\preference\train.jsonl --output-path exports\preference_dpo.jsonl --format dpo
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli training-config examples\datasets\instruction\train.jsonl instruction --output-path exports\instruction_axolotl.yaml --base-model Qwen/Qwen2.5-Coder-7B-Instruct --target axolotl_yaml
```

## Opt-in local integration tests (Windows PowerShell example)

The default `pytest` run stays offline: the Ollama integration tests
(`engine/tests/test_ollama_integration.py`) are skipped unless explicitly
enabled. With a running Ollama server and at least one model pulled:

```powershell
cd engine
$env:CORPUS_STUDIO_OLLAMA_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m pytest -m integration -q --basetemp .pytest-tmp
```

Optional overrides: `CORPUS_STUDIO_OLLAMA_MODEL` (default `llama3.2`) and
`CORPUS_STUDIO_OLLAMA_BASE_URL` (default `http://localhost:11434`). Each test
self-skips if the backend is unreachable or has no models, so an
enabled-but-unavailable environment reports skips rather than failures. These
tests cover model discovery, backend health, an Evaluation run, and an AI Assist
review against the real backend.
