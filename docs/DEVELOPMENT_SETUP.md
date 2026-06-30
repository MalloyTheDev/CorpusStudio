# Development Setup

## Requirements

- Python 3.11+
- .NET 8 SDK
- Visual Studio 2022 or JetBrains Rider for desktop development
- GitHub CLI, if creating the upstream GitHub repository from this checkout

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

## Local environment file

Copy the example environment file for local defaults:

```powershell
Copy-Item .env.example .env
```

Do not put secrets or private dataset paths in committed files.

## Python engine setup

From the repository root:

```powershell
cd engine
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

## Desktop setup

From the repository root:

```powershell
dotnet build apps\desktop\CorpusStudio.Desktop.sln
.\apps\desktop\CorpusStudio.Desktop\bin\Debug\net8.0-windows\CorpusStudio.Desktop.exe
```

The desktop app uses the local Python engine. If the app cannot find Python or the engine, confirm `.env` contains:

```text
CORPUS_STUDIO_DATA_DIR=data/projects
CORPUS_STUDIO_EXPORT_DIR=exports
CORPUS_STUDIO_ENGINE_DIR=engine
```

## v0.1 desktop smoke test

Use this checklist after a build:

1. Launch the desktop app.
2. Click **New Dataset Project**.
3. Enter a project name and choose one of the v0.1 schemas.
4. Click **Create**.
5. In **Writing Studio**, keep or edit the generated JSON example.
6. Click **Validate** and confirm the validation panel reports a valid row.
7. Click **Save Example** and confirm the row is appended to the active project.
8. Click **Export JSONL** and confirm an export appears under `exports/<project_id>/export.jsonl`.
9. Open **Settings** and confirm the repository, engine, Python, project, and export paths point to this checkout.

## Validate example datasets

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli validate examples\datasets\instruction\train.jsonl instruction
.\engine\.venv\Scripts\python.exe scripts\validate_examples.py
```

## Useful engine commands

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli schemas
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli new-project demo "Demo Dataset" instruction
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli quality examples\datasets\instruction\train.jsonl
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli split examples\datasets\instruction\train.jsonl exports\instruction_split instruction
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli export examples\datasets\instruction\train.jsonl exports\instruction.jsonl instruction
```
