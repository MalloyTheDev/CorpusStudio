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

## Local environment file

Copy the example environment file for local defaults:

```powershell
Copy-Item .env.example .env
```

Do not put secrets or private dataset paths in committed files.

## Python engine setup

```bash
cd engine
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Desktop setup

```powershell
cd apps/desktop
dotnet new sln -n CorpusStudio.Desktop
dotnet sln add CorpusStudio.Desktop/CorpusStudio.Desktop.csproj
dotnet build
```

## Validate example datasets

```bash
cd engine
python -m corpus_studio.cli validate ../examples/datasets/instruction/train.jsonl instruction
```
