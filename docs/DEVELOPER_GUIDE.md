# Developer Guide

A hands-on walkthrough for working in the codebase: how the desktop talks to the
engine, how the view-models are structured, how to add a feature, and how the tests
fake it all. For the high-level picture see [ARCHITECTURE.md](ARCHITECTURE.md); for
setup see [the Development Setup section](#development-setup); for every engine command see
[CLI_REFERENCE.md](CLI_REFERENCE.md).

## The pieces

- **`engine/`** — a dependency-light Python CLI (`corpus_studio`), the source of truth
  for all dataset logic: schemas, validation, quality/debt, gates, splits, evaluation,
  training-config, artifacts, versions, import/export. No UI, no desktop knowledge. It also holds the
  **platform run-lifecycle substrate** (`corpus_studio/platform/`: language-neutral contracts +
  planner + calibrator + run supervisor + subprocess worker + backend manifests). The dependency-light
  control plane stays torch-free at import; a validated worker lazily enters the opt-in `[train]`
  runtime only after sealed-plan dispatch. Unsloth is declared but not Phase-9B executable.
- **`apps/web/`** — a Tauri 2 + React contract-first client (early-stage): TypeScript types generated
  from the engine's JSON-Schema contracts; the Rust shell shells out to the `corpus-studio platform-*`
  CLI. A *client* of the engine, like the desktop.
- **`apps/desktop/`** — the .NET desktop, split into four projects:
  - **`CorpusStudio.Core`** (`net8.0`, WPF-free) — the view-models, models, and service
    **seams**. This is where nearly all logic lives.
  - **`CorpusStudio.Desktop`** (`net8.0-windows`) — the WPF head: XAML views + the real
    platform adapters. This is the shipping app.
  - **`CorpusStudio.Avalonia`** (`net8.0`) — a cross-platform proof head over the *same*
    `Core` view-models, with Avalonia adapters.
  - **`CorpusStudio.Desktop.Tests`** (xUnit) — tests against `Core` with fakes.

Namespaces stay `CorpusStudio.Desktop.*` even inside `Core` — the split is by project,
not by namespace.

## The engine ⇄ desktop bridge

The desktop never imports Python. It **shells out** to the CLI:

`PythonEngineService` (in `Core/Services`) builds an argument list, runs
`python -m corpus_studio.cli <command> …` as a subprocess (UTF-8 forced), captures
**stdout** (usually JSON), and deserializes it into a C# model
(`Core/Models/*`, snake-case `[JsonPropertyName]` mirroring the engine's JSON). Errors
surface as a non-zero exit → thrown `InvalidOperationException`; **stderr** carries
human notes / progress. A long run can be cancelled (`CancelRunningEngineCommand`), and
one command can stream stderr line-by-line for live progress (see the eval progress bar).

`PythonEngineService` implements **`IEngineService`** — the seam the view-models depend
on. VMs call `_engine.SomethingAsync(...)`; they never touch the process. Tests supply a
`FakeEngine` instead. When you add an engine command a VM needs, add it to
`IEngineService` **and** `FakeEngine` (the concrete's full signature, including optional
params, or `PythonEngineService` won't satisfy the interface).

## View-model structure (seam-based MVVM)

- **`MainWindowViewModel`** is the shell: it owns the engine-run **commands**
  (`AsyncRelayCommand` for async, `RelayCommand` for sync — both guard re-entrancy) and
  composes the per-tab child view-models.
- **Per-tab view-models** (`Core/ViewModels/Tabs/*`, each behind an `I…ViewModel`
  interface) hold that tab's state + pure `Apply…`/`Set…` display logic — no engine
  access. Examples: `DebtViewModel`, `EvaluationViewModel`, `TrainingViewModel`,
  `AiAssistViewModel`, `ArtifactsViewModel`, `SuitesViewModel`, …
- **Head-agnostic seams** injected into the shell VM (each with a Core `Null…` default so
  the parameterless design-time/test ctor keeps working, and a real adapter per head):
  - `IEngineService` — the engine (above).
  - `IDialogService` — `ConfirmAsync` / `ShowAsync` (message boxes).
  - `IFilePickerService` — file/folder pickers.
  - `IHuggingFaceImportDialog` — the HF import modal (returns a staging path).
  - `IDispatcherTimerFactory` / `IDispatcherTimer` — UI-thread periodic timers.
  - `IProcessRunner` - spawns + streams reviewed **external** trainer processes. First-party worker
    ownership belongs to the platform supervisor, not the desktop.

The pattern: **every user action is a bindable command on a VM, driven through seams**, so
it runs identically on both heads and is unit-testable with fakes. The code-behind
(`MainWindow.xaml.cs`) is essentially pure View glue — event wiring, the process/window
lifecycle, and delegation to `ViewModel.*`.

## Both heads, one contract

The WPF `MainWindow.xaml` and the Avalonia `MainWindow.axaml` bind the *same* VM
commands/properties. The Avalonia head uses **compiled bindings**
(`AvaloniaUseCompiledBindingsByDefault=true` + `x:DataType`), so **a green Avalonia build
validates every binding path against the VM at compile time** — that is the standing
"both heads in sync" check. If you add a command and bind it in Avalonia, the build
proves the path exists.

DI wiring lives in each head's `App`: `CorpusStudio.Desktop/App.xaml.cs` registers the
WPF adapters (`MessageBoxDialogService`, `Win32FilePickerService`, `WpfDispatcherTimerFactory`,
`TrainingProcessRunner`, `HuggingFaceImportDialog`, `PythonEngineService`); the Avalonia
`App.axaml.cs` registers the Avalonia adapters (and `Null…` where a head has no real one yet).

## Recipe: add a feature (or convert a handler)

1. **Engine first.** If it's new dataset logic, add it to the Python engine + tests
   (`engine/.venv/bin/python -m pytest -q` on the current Linux host); expose a CLI command. Keep it
   dependency-light and honest (report what a number measures; never overclaim).
2. **Seam.** Add the method to `IEngineService` (+ the `FakeEngine` stub) if a VM needs it.
   A new C# model in `Core/Models` mirrors the engine JSON (snake-case `JsonPropertyName`).
3. **VM.** Add a `public async Task XAsync()` on the relevant VM using `_engine` +
   child-VM state; expose an `XCommand` (`AsyncRelayCommand`/`RelayCommand`) — declare the
   property, init it in the ctor. Put display state on the child VM (`Apply…`/`Set…`).
4. **View.** Bind the WPF button `Command="{Binding XCommand}"`; add the Avalonia binding
   too. Prefer commands + bindings over code-behind handlers.
5. **Test.** Drive the VM with `FakeEngine` (+ `FakeDialogService`, etc.) and assert the
   applied state. Long-running / process / timer flows use the fake runner / fake timer.

## Testing & mock setup

- **Construction.** `new MainWindowViewModel()` (parameterless) chains to the full ctor
  with all-`Null…` seams — ~130 tests use it directly. When a test needs a fake engine or
  dialog, `EngineCommandTests.VmWith(engine, dialogs?, filePicker?, hfImportDialog?,
  timerFactory?, trainingRunner?)` builds the full graph.
- **Fakes** live in the test project: `FakeEngine` (settable results + call-tracking, e.g.
  `SaveTrainingRunRecordCallCount`, `LastJudgeBackend`), `FakeDialogService(confirm:)`,
  `FakeFilePickerService(path)`, `FakeHuggingFaceImportDialog(staging)`,
  `FakeDispatcherTimer(Factory)` (tick on demand), `FakeTrainingProcessRunner` (streams
  configured lines + a configured exit code, no real spawn).
- **Pure logic is extracted to be testable.** Rather than assert on live processes/UI,
  the branching lives in pure helpers with their own tests — e.g.
  `TrainingRunClassifier.Classify` (run result → terminal status),
  `MainWindowViewModel.TryParseEvaluationProgress`, `PythonEngineService.NormalizeExportExtension`.
- **Progress that's marshaled asynchronously** (e.g. eval `Progress<string>`) is guarded so
  a late callback after the run cleared can't re-show stale state — see
  `EvaluationViewModel.SetEvaluationProgress`.

## Running the gates

- **Engine** (current Linux host): `cd engine` then
  `.venv/bin/python -m ruff check corpus_studio tests` · `.venv/bin/python -m mypy corpus_studio` ·
  `.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp` (with
  `--cov=corpus_studio` a coverage floor applies). Optional accuracy extras: `[tokenizer]`,
  `[model-tokenizer]`, `[parquet]`, and `[train]` (the first-party QLoRA trainer — heavy/GPU:
  torch/transformers/peft/trl, plus bitsandbytes which is CUDA-only, so skipped on macOS; the gate
  itself needs none of these).
- **Avalonia on Linux**: `dotnet build apps/desktop/CorpusStudio.Avalonia/CorpusStudio.Avalonia.csproj`.
- **WPF on Windows only**: `dotnet build apps/desktop/CorpusStudio.Desktop.sln` and
  `dotnet test apps/desktop/CorpusStudio.Desktop.Tests/CorpusStudio.Desktop.Tests.csproj`.
- **Web/Tauri client on Linux**: `cd apps/web`, then `npm ci` and `npm run build`.
- **CI** runs the engine gate, desktop build+tests, the web/Tauri build, and CodeQL (Python + C#) on every PR.

## Honesty invariants (don't weaken these)

The gates, provenance, and scoring language are deliberately honest — a suite/gate PASS
is a *structure/threshold* verdict, not proof of quality; keyword-overlap is a lexical
proxy, not a quality judgment; provider policy keeps cloud models evaluator-only; PII
redaction masks known patterns and is *not* de-identification. When you touch these,
preserve the verdict semantics and the "what this measures" wording.

---

## Development Setup

_Consolidated from the former `docs/DEVELOPMENT_SETUP.md`._

### Requirements

- Python 3.11+
- .NET 8 SDK
- Visual Studio 2022 (Windows WPF) or JetBrains Rider / the .NET SDK (cross-platform Avalonia)
- Node.js/npm and Rust/Tauri prerequisites for `apps/web`
- GitHub CLI, if creating the upstream GitHub repository from this checkout

### Current native-Linux host quick start

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

### GitHub setup

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

### Local environment file (Windows PowerShell)

Copy the example environment file for local defaults:

```powershell
Copy-Item .env.example .env
```

Do not put secrets or private dataset paths in committed files.

### Python engine setup (Windows PowerShell)

From the repository root:

```powershell
cd engine
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

### Desktop setup (Windows WPF only)

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

### v0.1 desktop smoke test (Windows WPF only)

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

### Validate example datasets

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

### Useful engine commands (Windows PowerShell examples)

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

### Local model-backed commands (Windows PowerShell examples)

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

### Opt-in local integration tests (Windows PowerShell example)

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
