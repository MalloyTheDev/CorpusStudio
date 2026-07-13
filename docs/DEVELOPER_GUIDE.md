# Developer Guide

A hands-on walkthrough for working in the codebase: how the desktop talks to the
engine, how the view-models are structured, how to add a feature, and how the tests
fake it all. For the high-level picture see [ARCHITECTURE.md](ARCHITECTURE.md); for
setup see [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md); for every engine command see
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
   (`engine/.venv/Scripts/python.exe -m pytest -q`); expose a CLI command. Keep it
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

- **Engine** (the CI gate): `cd engine` then
  `./.venv/Scripts/python.exe -m ruff check .` · `… -m mypy corpus_studio` ·
  `… -m pytest -q --basetemp=.pytest_tmp` (with `--cov=corpus_studio` a coverage floor
  applies). Optional accuracy extras: `[tokenizer]`, `[model-tokenizer]`, `[parquet]`, and
  `[train]` (the first-party QLoRA trainer — heavy/GPU: torch/transformers/peft/trl, plus
  bitsandbytes which is CUDA-only, so skipped on macOS; the gate itself needs none of these).
- **Desktop**: `dotnet build apps/desktop/CorpusStudio.Desktop.sln` (builds **both** heads)
  and `dotnet test apps/desktop/CorpusStudio.Desktop.sln`.
- **CI** runs the engine gate, desktop build+tests, and CodeQL (Python + C#) on every PR.

## Honesty invariants (don't weaken these)

The gates, provenance, and scoring language are deliberately honest — a suite/gate PASS
is a *structure/threshold* verdict, not proof of quality; keyword-overlap is a lexical
proxy, not a quality judgment; provider policy keeps cloud models evaluator-only; PII
redaction masks known patterns and is *not* de-identification. When you touch these,
preserve the verdict semantics and the "what this measures" wording.
