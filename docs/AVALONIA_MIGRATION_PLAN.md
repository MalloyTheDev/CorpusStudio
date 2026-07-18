# Avalonia Migration — Execution Plan

**The companion [Cross-Platform Assessment](#cross-platform-assessment-avalonia-migration) is consolidated at the end of this doc.** The assessment
answers *whether/why* (yes, eventually; Avalonia; decompose first). This document is the *how* — a
concrete, phased, slice-by-slice plan grounded in the current decomposition state.

## Progress

- **Phase 0 — DONE.** Cross-platform venv-path fix (`PythonExecutableResolver`) + the platform-seam
  shims behind interfaces: `IDialogService` (WPF `MessageBoxDialogService`) and `IFilePickerService`
  (WPF `Win32FilePickerService`), DI-registered, all dialogs/pickers routed through them.
- **Phase 1 — foundation + spike DONE.** Extracted a shared **`CorpusStudio.Core`** (`net8.0`) with
  all Models + view-models + the WPF-free services + seam interfaces (the WPF head keeps Views/`App`
  + the two WPF adapters and references Core). Then the **spike passed (GO):** a `CorpusStudio.Avalonia`
  (`net8.0`) head builds the **Debt + Arena** tabs in `.axaml` over the *unchanged* view-models —
  compiled bindings validate at build time (same DI as WPF; `IsVisible`←`bool`; TwoWay text; list
  templating), so a green build is the proof. Not shipped; WPF remains the product head.
- **Phase 2 — DONE.** Decomposed every tab out of `MainWindowViewModel` into `IXxxViewModel` +
  `XxxViewModel : ViewModelBase` + DI, one shippable slice per tab — **all 14 real tabs** (Debt, Arena,
  Settings, Versions, Artifacts, Suites, Splits, Preference Review, Quarantine, Examples, Writing Studio,
  AI Assist, Evaluation, Training; the last several via multi-PR splits through backend-connection
  sub-VMs) plus the **Quality panel**. (Dashboard is a composition view over the extracted children, not
  its own VM.) Tab extraction dropped the VM from **5,609 to ~2,165 lines** — what remained was legitimate
  shell orchestration (shell mode, project list, active-project, engine-unavailable, output log). The
  later command conversion (below) then intentionally grew it back to **~4,300 lines** by consolidating
  per-head run-orchestration *into* the VM as testable bindable commands, shrinking the WPF code-behind.
- **Phase 3 — DONE (structural).** Re-authored the whole app as `.axaml` on the Avalonia head over the
  *unchanged* Core VMs: all 14 tab views + the shell chrome (activity bar, Start Center, Universal
  Explorer, docked Problems/Output panels, the Studio hero + Quality panel), plus a `StringToBrush`
  converter for the VMs' hex-string status colours. Compiled bindings validate every path at build time,
  so the green build is the proof. Not shipped; WPF stays the product head.
- **`ICommand` conversion (#184) — partially complete / in progress.** Converted the eligible WPF
  code-behind `_Click` engine handlers into shared `AsyncRelayCommand`/`RelayCommand`s on the view-models,
  behind the `IEngineService`
  seam so run-orchestration is head-agnostic and unit-testable with a fake engine. The mechanical,
  dialog (`IDialogService`), file-picker (`IFilePickerService`), and named-control input-binding tiers are
  **done** (code-behind `_Click` handlers **108 → 59**, `ICommand` count **0 → ~55**). Remaining handlers
  need per-handler refactors first: a process-streaming seam (Launch/Resume training), timer decoupling
  (checkpoint polling), the AI-Assist bulk-undo state migration, and workspace-init extraction
  (Locate/Retry engine). Then: real Explorer file tree, Fluent-theme styling, and Phase 4 packaging.

The Avalonia head, running (Phase 3): the **full workspace shell** — activity bar, the Studio hero with
the colour-coded Debt grade badge, the Projects sidebar, all 14 Studio tabs, the docked Quality panel —
rendered over the *unchanged* `CorpusStudio.Core` view-models (same VMs, same DI, cross-platform toolkit).
What began as a two-tab spike now binds the whole app; compiled bindings validate every path at build time.


Ground-truth measured 2026-07-08 (Phases 0–3 done; `ICommand` conversion in progress):

| Fact | Value |
|---|---|
| Studio tabs | **15** (Dashboard, Writing Studio, Examples, Preference Review, Quarantine, Splits, Evaluation, AI Assist, Training, Arena, Artifacts, Suites, Versions, Debt, Settings) |
| Per-tab VMs extracted | **All done** — 18 `IXxxViewModel` + `XxxViewModel : ViewModelBase` (the 14 real tabs incl. Training + Evaluation, plus the Quality panel, and the AI-Assist/Evaluation connection + AI-Assist rewrite-batches sub-VMs). Dashboard stays a composition view over the extracted children, not its own VM. |
| `MainWindowViewModel.cs` | **~4,316 lines** — fell to ~2,165 after tab extraction, then grew back as the `ICommand` conversion consolidated per-head run-orchestration into it (as testable commands behind `IEngineService`) |
| Code-behind `_Click` handlers | **59** (down from 108), `ICommand` count: **~55** — the mechanical/dialog/picker/input-binding tiers are converted; remaining handlers need per-handler refactors (process streaming, timer, undo-state, workspace-init) |
| Head-agnostic seams on the VM | `IEngineService` (**57 methods**, faked in tests), `IDialogService` (confirm/message), `IFilePickerService` (file/folder pick) — all with Core `Null*` defaults for the parameterless design-time ctor |
| DI | `App.xaml.cs` (WPF) + `App.axaml.cs` (Avalonia) → `ServiceCollection` register all extracted `IXxxViewModel`s + sub-VMs + `IEngineService`/`IDialogService`/`IFilePickerService` + `<MainWindowViewModel>`; ctor injection with a parameterless design-time/test ctor |
| WPF-only surface (from the assessment) | ~3,383 XAML lines · 40 Triggers · 93 `Visibility` bindings · 42 ControlTemplates · `MessageBox.Show`/file dialogs now behind `IDialogService`/`IFilePickerService` |
| Already portable | engine bridge (`Process`), all Models/Services/VMs in `CorpusStudio.Core`, 0 `DllImport`, 0 third-party UI pkgs, the whole **571-test** desktop suite |

## Critique (read before committing)

> **Resolved (2026-07-05).** The inversion recommended below was adopted: the spike ran on the two
> already-extracted tabs *first* and passed (GO). The riskiest assumption — Avalonia rebinding the
> unchanged VMs — is now proven, so the remaining decomposition (Phase 2, 13/15 tabs done) is the
> de-risked grunt work this section predicted. Kept as the record of why the order was chosen.

**The riskiest assumption is not "can we decompose" — it's "will Avalonia actually reuse these
VMs cleanly."** The decomposition pattern is already proven (Debt + Arena behind interfaces, DI,
`ViewModelBase`, a parameterless test ctor keeping `new MainWindowViewModel()` alive). What is
*unproven* is that an Avalonia head can bind to those exact VMs and that the platform seams
(dialogs, file pickers, Triggers→selectors, the venv path) shim cleanly. The assessment's own
recommendation — decompose all 13 remaining tabs *first*, then port — front-loads **weeks** of work
before that assumption is tested. That's the wrong risk order.

**Recommended inversion: prove the port on the 2 tabs already extracted, before decomposing the
other 13.** A throwaway-friendly Avalonia spike (shell + Debt + Arena over the *unchanged* VMs)
costs days, not weeks, and answers the real question. If it works, the remaining decomposition
becomes de-risked grunt work with a clear target shape. If it surfaces a blocker (a VM leaking WPF
types, a binding Avalonia can't express), we learn it for the price of 2 tabs, not 15.

**Honest caveats:**
- Even the "done" tabs aren't fully extracted — `MainWindowViewModel` still holds `DebtTrend` and
  a lot of Debt/Arena-adjacent state. "2/15" overstates progress; call it ~1.5/15.
- The 5,609-line god object is the cost center, and it **grew** during v1.2/v1.3 (was ~5,549).
  Every feature added without extraction makes the port more expensive — there's a carrying cost
  to *not* deciding.
- This is a genuine **multi-week** effort with a **dual-maintenance window**. It should not start
  unless cross-platform is a real product goal (users on macOS/Linux who want to drive local
  training there). If it's speculative, the decomposition is *still* worth doing for
  maintainability — but the Avalonia head is not.

## Phased plan

Each phase is independently valuable and shippable. WPF keeps shipping throughout.

### Phase 0 — Platform-seam shims + the venv fix (small; valuable even without a port)
Make the seams injectable so both heads can share VMs later, and fix the one real
cross-platform bug now.
- **`IDialogService`** (`ConfirmAsync` / `MessageAsync`) — route the 35 `MessageBox.Show` calls
  through it. WPF impl wraps `MessageBox`; Avalonia impl comes later. (The unsaved-work guard from
  Tier-A #134 is already a step toward this — one confirm chokepoint.)
- **`IFilePickerService`** — wrap the 4 `OpenFolderDialog`/`OpenFileDialog` sites.
- **Fix `ResolvePythonExecutable`** (`PythonEngineService.cs`): it hardcodes `.venv/Scripts/python.exe`
  (Windows). Add a POSIX branch (`.venv/bin/python`) so a macOS/Linux venv isn't silently bypassed.
  This is a real bug independent of the port — do it regardless.
- *Effort:* ~1–2 slices. *Ships on WPF; no behavior change.*

### Phase 1 — Avalonia proof spike (the GO/NO-GO gate)
A new `apps/desktop/CorpusStudio.Avalonia` project in the solution that **references the existing
Models/Services/VMs unchanged** and re-authors **only the shell + the Debt and Arena tabs** as
`.axaml`.
- Validates: VMs reused verbatim; `Trigger`→style selectors/pseudo-classes; `Visibility`→`IsVisible`;
  one `MessageBox`→`IDialogService`; `StorageProvider` file picker; the DI bootstrap on Avalonia.
- **Explicitly a spike** — throwaway-friendly, not parity. Its output is a **decision**: the
  shared-VM approach holds (proceed) or it doesn't (stop / rethink).
- *Effort:* days. *Does not ship; lives behind the WPF head.*

### Phase 2 — Finish the decomposition (the gating grunt work) — IN PROGRESS
Extract each tab's logic out of `MainWindowViewModel` into per-tab VMs behind interfaces, following
the proven Debt/Arena pattern, one shippable slice per tab. Sequenced **simplest/most-isolated
first** to build momentum and keep each slice low-risk (✓ = extracted):
1. ✓ **Settings, Versions, Suites, Artifacts** — mostly read/act over existing services; small state.
2. ✓ **Splits, Preference Review, Quarantine, Examples** — moderate; some editor state.
3. **Dashboard** (remaining), ✓ **Writing Studio** — bind-heavy but logic-light.
4. ✓ **Evaluation, AI Assist** (each via a multi-PR split through a backend-connection sub-VM),
   **Training** (remaining) — the big, coupled ones, done with the pattern fully grooved. (Debt/Arena's
   residual state still to finish.)
- Each slice: `IXxxViewModel` + `XxxViewModel : ViewModelBase` + DI registration + move the
  code-behind handlers to VM methods + tests. The desktop suite (**571 tests** and growing) guards
  every move; add per-VM tests as logic lands in a testable seam.
- *Effort:* the bulk — multiple weeks, but each tab is a clean, independently-reviewable PR.

### Phase 3 — Port tabs to Avalonia incrementally
As each tab's VM is extracted (Phase 2), re-author its view as `.axaml` in the Avalonia head,
converting Triggers→selectors and Visibility→`IsVisible` per view. Dual-head until parity.
- *Effort:* proportional to XAML per tab; overlaps Phase 2.

### Phase 4 — Per-OS engine + packaging story
The engine stays a runtime prerequisite (unchanged hard boundary), and the "engine not found" setup
screen guides that step on every OS. **Packaging is shipped (#188):** `release.yml`'s `avalonia-dist`
job publishes self-contained Avalonia builds for `linux-x64` + `osx-arm64`/`osx-x64` on native
runners and attaches them to the GitHub Release beside the Windows (WPF) build; `desktop-tests.yml`
also builds the Avalonia head on Linux every PR so a platform break is caught early. The self-hosted
engine setup story per OS remains orthogonal to the UI port.

## Recommended first concrete slice

**Phase 0's `IDialogService` + the venv-path fix**, as one small PR — it's the lowest-risk, highest-
leverage starting point: it's useful on WPF today, it fixes a real cross-platform bug, and it
removes the single most-duplicated platform seam (35 `MessageBox.Show`) before the spike needs it.
Then **Phase 1 (the Avalonia spike)** as the very next step to hit the GO/NO-GO gate cheaply.

## Risks & non-goals
- **Not a mechanical XAML rename** — Triggers→selectors + Visibility→`IsVisible` are conceptual,
  spread across ~130 sites. Budget for them.
- **Dual-maintenance window** while both heads exist — mitigated by all logic in shared VMs.
- **Carrying cost** — every new feature added to the god object before Phase 2 makes the port dearer.
- **Non-goal:** bundling Python/CUDA/a GPU stack (unchanged). This is a *UI* port.
- **Non-goal:** a big-bang rewrite. Every phase ships; WPF never stops working.

## Acceptance criteria per phase
- **P0:** all 35 `MessageBox`/4 file-dialog sites route through the services; `ResolvePythonExecutable`
  resolves a POSIX venv; desktop tests green; WPF behavior unchanged.
- **P1:** the Avalonia head launches, shows the shell + Debt + Arena tabs bound to the *unchanged*
  VMs, over the real engine; a written GO/NO-GO with any surfaced blockers.
- **P2 (per tab):** the tab's logic lives in `IXxxViewModel`/`XxxViewModel` with tests; the god
  object shrinks by that tab; WPF still green.
- **P3 (per tab):** the tab renders + works in the Avalonia head at parity with WPF.

---

## Cross-Platform Assessment (Avalonia migration)

_Consolidated from the former `docs/CROSS_PLATFORM_ASSESSMENT.md`._

**Status:** the *whether/why* record for the cross-platform move. Its recommendation was
adopted and is now being executed — Phases 0–3 are done (seams + venv fix + the shared
`CorpusStudio.Core` + an Avalonia spike that passed GO + **all per-tab view-models extracted**
+ the whole app re-authored as `.axaml` on the Avalonia head), and the `ICommand` conversion
(issue #184) is in progress. For live progress see
the phased execution plan above; this section is kept as the original
analysis of how the Windows-only WPF desktop (`apps/desktop`) could become cross-platform
(macOS/Linux), where the local trainers the app orchestrates actually run.

All figures below were measured against the code (not docs) on 2026-07-03 — a **pre-decomposition
snapshot**; the decomposition is now complete and orchestration is being consolidated into the
view-models (see the plan for current line counts). The *analysis* holds; the raw counts here are
historical.

### TL;DR / recommendation

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

### Why it's feasible: what already ports cleanly

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

### WPF-specific surface that needs rework

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

### Recommended migration path (when pursued)

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

### Risks & non-goals

- **Not a mechanical XAML rename.** Triggers→selectors and Visibility→`IsVisible` are conceptual,
  spread across ~130 sites; budget for them.
- **Dual-maintenance window** while both WPF and Avalonia heads exist — mitigated by all logic
  living in shared VMs/services.
- **Non-goal:** bundling Python or a GPU/ML stack (unchanged hard boundary). The port is about the
  *UI*, not the engine.
- **Recommendation:** don't port yet. Land #4 (decomposition) first — it's the gating cost and pays
  off with or without the port — then do an Avalonia slice over the extracted VMs to validate the
  approach before committing to full parity.
