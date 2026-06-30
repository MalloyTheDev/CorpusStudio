# Development Setup

## Requirements

- Python 3.11+
- .NET 8 SDK
- Visual Studio 2022 or JetBrains Rider for desktop development
- GitHub CLI, if creating the upstream GitHub repository from this checkout

## GitHub setup

This checkout is initialized as a local Git repository on the `main` branch.
The intended upstream repository is:

```text
https://github.com/MalloyTheDev/CorpusStudio.git
```

Create the GitHub repository only after choosing whether it should be public or private:

```powershell
gh repo create MalloyTheDev/CorpusStudio --private --source . --remote origin
```

Use `--public` instead of `--private` if this skeleton should be public from the first push.
After the remote exists, create the initial commit and push:

```powershell
git add .
git commit -m "Initial Corpus Studio skeleton"
git push -u origin main
```

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
