# Cross-Platform Assessment (Avalonia migration)

**Status:** the *whether/why* record for the cross-platform move. Its recommendation was
adopted and is now being executed — Phase 0/1 are done (seams + venv fix + the shared
`CorpusStudio.Core` + an Avalonia spike that passed GO) and Phase 2 (the per-tab
decomposition) is 13/15 done. For live progress see
[`AVALONIA_MIGRATION_PLAN.md`](AVALONIA_MIGRATION_PLAN.md); this doc is kept as the original
analysis of how the Windows-only WPF desktop (`apps/desktop`) could become cross-platform
(macOS/Linux), where the local trainers the app orchestrates actually run.

All figures below were measured against the code (not docs) on 2026-07-03 — a **pre-decomposition
snapshot** (e.g. the 5,549-line god object has since dropped to ~2,947); the *analysis* holds,
the raw counts are historical.

## TL;DR / recommendation

- **Worth doing, eventually.** The Python **engine already runs on macOS/Linux**, and the
  desktop's C# layer is portable except for the **WPF view layer**. The real value is letting
  users drive local training on the OS where their GPU/trainer lives.
- **Avalonia is the realistic target** (XAML + MVVM, same .NET, closest to WPF; MAUI drops
  Linux, Uno is heavier). No other realistic path reuses this much.
- **Do the god-object decomposition (backlog #4) FIRST.** The migration's dominant cost is the
  **~3,300-line `MainWindow.xaml` + a 5,549-line `MainWindowViewModel` + 103 code-behind Click
  handlers**. Porting that as-is is the hard part; porting per-tab view-models behind interfaces
  is tractable. #4 and #3 are the same work approached from two sides — sequence #4 before #3.
- **Rough effort:** a *runnable Avalonia slice* (shell + 1–2 tabs over the existing services) is
  **days**; a *full parity port* is **multiple weeks**, most of it re-authoring XAML views and the
  35 dialogs — not business logic.

## Why it's feasible: what already ports cleanly

- **The engine bridge is cross-platform.** `PythonEngineService` / `TrainingProcessRunner` drive
  the engine via `System.Diagnostics.Process` / `ProcessStartInfo` — portable .NET. The engine is
  dependency-light Python (pydantic/typer/orjson), already cross-platform.
- **No native interop.** `DllImport` in app code: **0**. (The PrintWindow screenshot harness is a
  Windows-only *developer* PowerShell script, not shipped app code.)
- **No third-party UI packages** (`PackageReference`: 0) and **no WinForms** (0). Nothing to re-source.
- **68 portable `.cs` files** — Models, Services, and ViewModels are plain .NET + MVVM
  (`INotifyPropertyChanged`), reusable verbatim by an Avalonia project.
- **The test suite (348 desktop + 455 engine) is UI-agnostic** — it exercises services and
  view-model logic, so it keeps protecting behavior through a port.

## WPF-specific surface that needs rework

| Area | Count | Avalonia equivalent | Effort |
|---|---|---|---|
| `MainWindow.xaml` (+ wizard, App) | 3 files, **3,378 lines** | Re-author as `.axaml` | High (bulk of the work) |
| `Trigger` / `DataTrigger` / `ControlTemplate.Triggers` | **40** | Style **selectors + pseudo-classes** (`:pointerover`, `:selected`, `:checked`) — WPF Triggers don't exist | High (conceptual rewrite) |
| `Visibility="{Binding …}"` via `BoolToVis`/`InverseBoolToVis` | **93** | Bind `IsVisible` (bool) directly; drop the converters (inverse = `!` or a tiny converter) | Medium (mechanical, many sites) |
| `ControlTemplate` (buttons/tabs/list items/activity bar) | **42** | Avalonia `ControlTemplate` — similar shape, different property/pseudo-class names | Medium |
| `MessageBox.Show` | **35** | **No built-in** — add `MessageBox.Avalonia` or a small custom dialog window; ideally route through one VM service so it's swappable | Medium (touches many handlers) |
| File dialogs (`OpenFolderDialog`/`OpenFileDialog`) | **4** | `TopLevel.StorageProvider` (async `OpenFolderPickerAsync` / `OpenFilePickerAsync`) | Low |
| `DispatcherUnhandledException` (App, from #83) | 1 | Avalonia global handler (`Dispatcher.UIThread` + `AppDomain`), no `DispatcherUnhandledException` | Low |
| `Dispatcher.BeginInvoke` | 8 | `Dispatcher.UIThread.Post/InvokeAsync` | Low |
| `DropShadowEffect` (dashboard cards) | 1 | `BoxShadow` on `Border` (or Avalonia `DropShadowEffect`) | Low |
| `TreeView` + `HierarchicalDataTemplate` (Explorer) | 9 | Avalonia `TreeView` + `TreeDataTemplate` | Low–Medium |
| `TabControl`/`TabItem` | 33 | Avalonia `TabControl` (close analog) | Low |
| `IValueConverter` | 1 | `IValueConverter` exists in Avalonia (near-identical) | Trivial |

Additional portability fixes (small but real, independent of Avalonia):
- **`ResolvePythonExecutable` was Windows-only** — it hardcoded `.venv/Scripts/python.exe` and
  silently bypassed a POSIX `.venv/bin/python`. **Fixed in Phase 0** (`PythonExecutableResolver`
  now checks both layouts); kept here as the original finding.
- **103 code-behind Click handlers, 0 `ICommand`.** Avalonia supports `Click` code-behind, so
  handlers *can* port as-is — but this is the same coupling that makes #4 (decomposition) the
  precondition. Moving to per-tab VMs + commands makes the view layer thin enough to re-author.

## Recommended migration path (when pursued)

1. **#4 first** — extract per-tab view-models behind interfaces + a DI container; prove one tab.
   This shrinks the code-behind and makes the logic Avalonia-ready (and is valuable on its own).
2. **New Avalonia project** in the solution that **references the existing Models/Services/VMs
   unchanged** and re-implements the shell + one tab as `.axaml`. Keep WPF shipping in parallel.
3. **Shim the platform seams once:** a dialog service (replaces the 35 `MessageBox.Show`), a
   file-picker service (`StorageProvider`), and the venv-path fix. Inject them so both heads share
   VMs.
4. **Port tabs incrementally**, converting Triggers→selectors and Visibility→`IsVisible` as each
   view moves. The test suite guards the VMs throughout.
5. Decide the **engine-shipping story per-OS** (still a runtime prerequisite; the "engine not
   found" screen from #2 already guides setup) — orthogonal to the UI port.

## Risks & non-goals

- **Not a mechanical XAML rename.** Triggers→selectors and Visibility→`IsVisible` are conceptual,
  spread across ~130 sites; budget for them.
- **Dual-maintenance window** while both WPF and Avalonia heads exist — mitigated by all logic
  living in shared VMs/services.
- **Non-goal:** bundling Python or a GPU/ML stack (unchanged hard boundary). The port is about the
  *UI*, not the engine.
- **Recommendation:** don't port yet. Land #4 (decomposition) first — it's the gating cost and pays
  off with or without the port — then do an Avalonia slice over the extracted VMs to validate the
  approach before committing to full parity.
