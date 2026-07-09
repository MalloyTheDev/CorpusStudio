using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;

using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels.Tabs;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>Which top-level view the workspace shell shows (v1.2.4 Workspace System).
/// StartCenter is a full-window screen (no activity bar); Files and Studio sit behind the
/// activity bar. Studio is the existing 14-tab app, preserved intact.</summary>
public enum WorkspaceShellMode
{
    StartCenter,
    Files,
    Studio,
}

/// <summary>The Studio tabs, in the WPF head's TabControl order. The values are the tab indices the
/// WPF head binds via <c>SelectedIndex="{Binding SelectedStudioTabIndex}"</c>, so cross-tab navigation
/// lives in the view-model (a shared command) instead of code-behind. Keep in sync with the XAML order.</summary>
public enum StudioTab
{
    Dashboard = 0,
    WritingStudio = 1,
    Examples = 2,
    PreferenceReview = 3,
    Quarantine = 4,
    Splits = 5,
    Evaluation = 6,
    AiAssist = 7,
    Training = 8,
    Arena = 9,
    Artifacts = 10,
    Suites = 11,
    Versions = 12,
    Debt = 13,
    Settings = 14,
}

public sealed class MainWindowViewModel : INotifyPropertyChanged
{
    private string _activeProjectTitle = "New Dataset Project";
    private string? _activeProjectPath;
    private string _activeSchemaId = "instruction";
    private string _activeSchemaDescription =
        "Choose a schema, write examples, validate rows, and export model-ready JSONL.";
    private string _validationSummary = "Create a project to start validation.";
    // Quality panel (report summary + metric grid + status banner + flagged-row detail + quality
    // history + debt-trend + synthetic-issue list/triage) extracted to QualityViewModel in Phase 2
    // (slice 5). The shell keeps the synthetic-triage AI-Assist handoffs reaching into the child.
    private string _gateSummary = "Run gates to check whether this dataset may move forward.";
    private bool _problemsPanelVisible;
    private string _problemsSummary = "Run gates to check this dataset for problems.";
    private string _problemsBadge = string.Empty;
    private string _problemsBadgeColor = GateReport.StatusColor(null);
    private bool _outputPanelVisible;
    private bool _searchPanelVisible;
    // Evaluation tab core (run + report panes + per-example results + failure filters + report
    // history) extracted to EvaluationViewModel in Phase 2 (backend-cluster slice 3, PR 3b). The
    // shell keeps the bridges (health methods, eval->AiAssist/ReviewedFix, Training-baseline) reaching in.
    private bool _exportRemoveDuplicates;
    private bool _exportRemoveLowInformation;
    private bool _exportRedactPii;
    private string _exportFormat = "jsonl";
    private string _benchmarkModelsInput = string.Empty;
    private string _benchmarkSummary =
        "Enter one model per line, then benchmark them against this project's examples.";
    // AI-Assist tab core (run + result panes + candidate gate + review queue + saved views)
    // extracted to AiAssistViewModel in Phase 2 (backend-cluster slice 2, PR 3/3). The shell
    // keeps the bridges (synthetic/eval/preference/resume writers, health methods) reaching in.
    private string _reviewedFixSummary =
        "Edited failed rows appear here so you can track which fixes were re-tested.";
    // Training tab (config export/preview/compatibility, launch + live run log, run registry +
    // regression gate, checkpoints/resume, baseline before/after comparison) extracted to
    // TrainingViewModel in Phase 2 (slice 4). It holds the shared Evaluation VM for the baseline
    // comparison; the shell keeps the launch-prep bridges (which read the engine/Evaluation history).
    private string _datasetCardSummary =
        "Generate a dataset card to summarize metadata, schema, splits, quality, and evaluation.";
    private string _datasetCardPreview = "Dataset card preview appears here.";
    private string _projectIndexSummary = "Projects list from local files. Rebuild the index to list from SQLite.";
    private bool _isBusy;
    private string _busyStatus = "Working...";
    private bool _hasError;
    private string _errorMessage = string.Empty;
    private string _labSettingsSummary = "Lab backend settings can be saved per project.";
    private DatasetProjectListItem? _selectedProject;
    private ValidationIssueNavigationItem? _selectedValidationIssue;
    private ReviewedFixRecord? _selectedReviewedFix;
    private ReviewedFixRecord? _lastPreparedEvaluationFix;

    public event PropertyChangedEventHandler? PropertyChanged;

    // ---- Workspace shell (v1.2.4 view layer) -------------------------------------
    // Default is Studio, so the app opens exactly as today (plus the activity bar);
    // the Start Center is reachable via the activity-bar brand button.
    private WorkspaceShellMode _shellMode = WorkspaceShellMode.Studio;

    /// <summary>The Start Center's own view-model (Recent Workspaces). Kept separate from
    /// this class per the workspace design.</summary>
    public StartCenterViewModel StartCenter { get; } = new();

    /// <summary>The Universal Workspace Explorer's own view-model (file tree + documents).
    /// Operates on the active project's folder.</summary>
    public WorkspaceExplorerViewModel Explorer { get; } = new();

    /// <summary>The workspace content-search ("find in files") view-model. Shares the active
    /// project's folder as its search root.</summary>
    public WorkspaceSearchViewModel Search { get; } = new();

    /// <summary>Extracted per-tab view-models (backlog #4). Injected by DI; the parameterless
    /// ctor supplies defaults so existing `new MainWindowViewModel()` call sites keep working.</summary>
    public IDebtViewModel Debt { get; }

    public IArenaViewModel Arena { get; }

    public ISettingsViewModel Settings { get; }

    public IVersionsViewModel Versions { get; }

    public IArtifactsViewModel Artifacts { get; }

    public ISuitesViewModel Suites { get; }

    public ISplitsViewModel Splits { get; }

    public IPreferenceReviewViewModel PreferenceReview { get; }

    public IQuarantineViewModel Quarantine { get; }

    public IExamplesViewModel Examples { get; }

    public IWritingStudioViewModel WritingStudio { get; }

    public IAiAssistRewriteBatchesViewModel RewriteBatches { get; }

    public IAiAssistConnectionViewModel AiAssistConnection { get; }

    public IAiAssistViewModel AiAssist { get; }

    public IEvaluationConnectionViewModel EvaluationConnection { get; }

    public IEvaluationViewModel Evaluation { get; }

    public ITrainingViewModel Training { get; }

    public IQualityViewModel Quality { get; }

    // Shell-navigation commands: both heads can bind Command="{Binding X}" instead of per-head
    // code-behind Click handlers (the cross-platform pattern the Avalonia port moves toward).
    public System.Windows.Input.ICommand ShowStartCenterCommand { get; }
    public System.Windows.Input.ICommand ShowFilesCommand { get; }
    public System.Windows.Input.ICommand ShowStudioCommand { get; }
    public System.Windows.Input.ICommand ToggleProblemsPanelCommand { get; }
    public System.Windows.Input.ICommand ToggleOutputPanelCommand { get; }
    public System.Windows.Input.ICommand DismissErrorCommand { get; }

    private readonly IEngineService _engine = null!;
    private readonly IDialogService _dialogs = new NullDialogService();
    private readonly IFilePickerService _filePicker = new NullFilePickerService();
    private readonly IHuggingFaceImportDialog _huggingFaceImportDialog = new NullHuggingFaceImportDialog();
    private readonly IDispatcherTimerFactory _dispatcherTimerFactory = new NullDispatcherTimerFactory();

    // Training-run launch state (moved from the desktop code-behind, #246). The runner spawns the
    // trainer + streams output; lines are enqueued off-thread and flushed to the UI by a timer.
    private readonly IProcessRunner _trainingRunner = new TrainingProcessRunner();
    private readonly System.Collections.Concurrent.ConcurrentQueue<string> _trainingLogQueue = new();
    private System.Threading.CancellationTokenSource? _trainingRunCts;
    private bool _trainingCancelRequested;

    /// <summary>Run the dataset-debt assessment and apply it to the Debt tab — the engine-run
    /// orchestration moved off the desktop code-behind into a shared async command.</summary>
    public System.Windows.Input.ICommand RunDatasetDebtCommand { get; }
    public System.Windows.Input.ICommand RunGatesCommand { get; }
    public System.Windows.Input.ICommand RunChatGatesCommand { get; }
    public System.Windows.Input.ICommand RunArenaCommand { get; }
    public System.Windows.Input.ICommand RunSuiteCommand { get; }
    public System.Windows.Input.ICommand ViewDatasetVersionCardCommand { get; }
    public System.Windows.Input.ICommand CaptureDatasetVersionCommand { get; }
    public System.Windows.Input.ICommand GenerateTrainingConfigCommand { get; }
    public System.Windows.Input.ICommand RunBenchmarkCommand { get; }
    public System.Windows.Input.ICommand RunEvaluationCommand { get; }
    public System.Windows.Input.ICommand RerunEvaluationReportCommand { get; }
    public System.Windows.Input.ICommand CheckEvaluationBackendCommand { get; }
    public System.Windows.Input.ICommand RefreshEvaluationModelsCommand { get; }
    public System.Windows.Input.ICommand CheckAiAssistBackendCommand { get; }
    public System.Windows.Input.ICommand RefreshAiAssistModelsCommand { get; }
    public System.Windows.Input.ICommand RunAiAssistCommand { get; }
    public System.Windows.Input.ICommand GenerateSplitsCommand { get; }
    public System.Windows.Input.ICommand GateTrainingRunCommand { get; }
    public System.Windows.Input.ICommand RefreshTrainingCheckpointsCommand { get; }
    public System.Windows.Input.ICommand LaunchTrainingCommand { get; }
    public System.Windows.Input.ICommand ResumeTrainingCommand { get; }
    public System.Windows.Input.ICommand StopTrainingCommand { get; }
    public System.Windows.Input.ICommand RunQualityCommand { get; }
    public System.Windows.Input.ICommand SaveGateThresholdsCommand { get; }
    public System.Windows.Input.ICommand RefreshProviderPoliciesCommand { get; }
    public System.Windows.Input.ICommand ApproveProviderGenerationCommand { get; }
    public System.Windows.Input.ICommand RevokeProviderGenerationCommand { get; }
    public System.Windows.Input.ICommand RebuildProjectIndexCommand { get; }
    public System.Windows.Input.ICommand ExportPreferenceRankingCommand { get; }
    public System.Windows.Input.ICommand RefreshDatasetVersionsCommand { get; }
    public System.Windows.Input.ICommand RestoreDatasetVersionCommand { get; }
    public System.Windows.Input.ICommand SaveExampleCommand { get; }
    public System.Windows.Input.ICommand ImportDatasetCommand { get; }
    public System.Windows.Input.ICommand ImportFromHuggingFaceCommand { get; }
    public System.Windows.Input.ICommand LocateEngineCommand { get; }
    public System.Windows.Input.ICommand RetryEngineCommand { get; }
    public System.Windows.Input.ICommand NewSuiteCommand { get; }
    public System.Windows.Input.ICommand ExportJsonlCommand { get; }
    public System.Windows.Input.ICommand RegisterArtifactFromRunCommand { get; }
    public System.Windows.Input.ICommand KeepArtifactCommand { get; }
    public System.Windows.Input.ICommand RejectArtifactCommand { get; }
    public System.Windows.Input.ICommand RefreshArtifactsCommand { get; }
    public System.Windows.Input.ICommand DiffVersionsCommand { get; }
    public System.Windows.Input.ICommand ViewArtifactCardCommand { get; }
    public System.Windows.Input.ICommand GenerateDatasetCardCommand { get; }
    public System.Windows.Input.ICommand ValidateDraftCommand { get; }
    public System.Windows.Input.ICommand CheckTrainingCompatibilityCommand { get; }
    public System.Windows.Input.ICommand RefreshSuitesCommand { get; }
    public System.Windows.Input.ICommand ExportPreferenceForTrainingCommand { get; }

    // Cross-tab navigation commands (replace the desktop's XxxTab.IsSelected = true code-behind).
    public System.Windows.Input.ICommand GoToDebtCommand { get; }
    public System.Windows.Input.ICommand GoToWritingStudioCommand { get; }
    public System.Windows.Input.ICommand GoToSplitsCommand { get; }
    public System.Windows.Input.ICommand GoToEvaluationCommand { get; }
    public System.Windows.Input.ICommand GoToTrainingCommand { get; }
    // Prepare-then-navigate: run the AI-Assist handoff bridge; on success it selects the AI Assist tab.
    public System.Windows.Input.ICommand PrepareEvaluationFailureReviewCommand { get; }
    public System.Windows.Input.ICommand PreparePreferenceJudgeReviewCommand { get; }
    public System.Windows.Input.ICommand PreparePreferenceBatchJudgeReviewCommand { get; }

    /// <summary>Design-time / test constructor.</summary>
    public MainWindowViewModel()
        : this(
            new DebtViewModel(), new ArenaViewModel(), new SettingsViewModel(),
            new VersionsViewModel(), new ArtifactsViewModel(), new SuitesViewModel(),
            new SplitsViewModel(), new PreferenceReviewViewModel(), new QuarantineViewModel(),
            new ExamplesViewModel(), new WritingStudioViewModel(), new AiAssistRewriteBatchesViewModel(),
            new AiAssistConnectionViewModel(), new EvaluationConnectionViewModel(), new QualityViewModel(),
            new PythonEngineService(), new NullDialogService(), new NullFilePickerService(),
            new NullHuggingFaceImportDialog(), new NullDispatcherTimerFactory(), new TrainingProcessRunner())
    {
    }

    public MainWindowViewModel(
        IDebtViewModel debt, IArenaViewModel arena, ISettingsViewModel settings,
        IVersionsViewModel versions, IArtifactsViewModel artifacts, ISuitesViewModel suites,
        ISplitsViewModel splits, IPreferenceReviewViewModel preferenceReview,
        IQuarantineViewModel quarantine, IExamplesViewModel examples,
        IWritingStudioViewModel writingStudio, IAiAssistRewriteBatchesViewModel rewriteBatches,
        IAiAssistConnectionViewModel aiAssistConnection, IEvaluationConnectionViewModel evaluationConnection,
        IQualityViewModel quality,
        IEngineService engine,
        IDialogService dialogs,
        IFilePickerService filePicker,
        IHuggingFaceImportDialog huggingFaceImportDialog,
        IDispatcherTimerFactory dispatcherTimerFactory,
        IProcessRunner trainingRunner)
    {
        _engine = engine;
        _dialogs = dialogs;
        _filePicker = filePicker;
        _dispatcherTimerFactory = dispatcherTimerFactory;
        _trainingRunner = trainingRunner;
        _huggingFaceImportDialog = huggingFaceImportDialog;
        Debt = debt;
        Arena = arena;
        Settings = settings;
        Versions = versions;
        Artifacts = artifacts;
        Suites = suites;
        Splits = splits;
        PreferenceReview = preferenceReview;
        Quarantine = quarantine;
        Examples = examples;
        WritingStudio = writingStudio;
        RewriteBatches = rewriteBatches;
        AiAssistConnection = aiAssistConnection;
        EvaluationConnection = evaluationConnection;
        Quality = quality;
        // The AI-Assist core is composed from the shared connection instance (its run reads the
        // backend/model), rather than DI-injected, so both share one AiAssistConnection.
        AiAssist = new AiAssistViewModel(aiAssistConnection);
        // Likewise the Evaluation core shares its connection instance (its run reads backend/model).
        Evaluation = new EvaluationViewModel(evaluationConnection);
        // Training holds the shared Evaluation instance so its before/after baseline comparison can
        // reuse Evaluation.BuildEvaluationReportComparison (shell-constructed, not DI, to share it).
        Training = new TrainingViewModel(Evaluation);
        // A run/split failure surfaces in the shell's shared error banner; the tabs keep no shell
        // reference, so they raise ErrorReported and the shell forwards it to ReportError.
        Splits.ErrorReported += ReportError;
        AiAssist.ErrorReported += ReportError;
        Evaluation.ErrorReported += ReportError;
        Training.ErrorReported += ReportError;
        Quality.ErrorReported += ReportError;
        // Load a suite's run-history trend when it's selected (best-effort; the engine call stays here).
        Suites.SuiteSelected += name => _ = LoadSuiteHistoryAsync(name);

        ShowStartCenterCommand = new RelayCommand(ShowStartCenter);
        ShowFilesCommand = new RelayCommand(ShowFiles);
        ShowStudioCommand = new RelayCommand(ShowStudio);
        ToggleProblemsPanelCommand = new RelayCommand(ToggleProblemsPanel);
        ToggleOutputPanelCommand = new RelayCommand(ToggleOutputPanel);
        DismissErrorCommand = new RelayCommand(DismissError);
        RunDatasetDebtCommand = new AsyncRelayCommand(RunDatasetDebtAsync);
        RunGatesCommand = new AsyncRelayCommand(RunGatesAsync);
        RunChatGatesCommand = new AsyncRelayCommand(RunChatGatesAsync);
        RunArenaCommand = new AsyncRelayCommand(RunArenaAsync);
        RunSuiteCommand = new AsyncRelayCommand(RunSuiteAsync);
        ViewDatasetVersionCardCommand = new AsyncRelayCommand(ViewDatasetVersionCardAsync);
        CaptureDatasetVersionCommand = new AsyncRelayCommand(CaptureDatasetVersionAsync);
        GenerateTrainingConfigCommand = new AsyncRelayCommand(GenerateTrainingConfigAsync);
        RunBenchmarkCommand = new AsyncRelayCommand(RunBenchmarkAsync);
        RunEvaluationCommand = new AsyncRelayCommand(RunEvaluationAsync);
        RerunEvaluationReportCommand = new AsyncRelayCommand(RerunEvaluationReportAsync);
        CheckEvaluationBackendCommand = new AsyncRelayCommand(CheckEvaluationBackendAsync);
        RefreshEvaluationModelsCommand = new AsyncRelayCommand(RefreshEvaluationModelsAsync);
        CheckAiAssistBackendCommand = new AsyncRelayCommand(CheckAiAssistBackendAsync);
        RefreshAiAssistModelsCommand = new AsyncRelayCommand(RefreshAiAssistModelsAsync);
        RunAiAssistCommand = new AsyncRelayCommand(RunAiAssistAsync);
        GenerateSplitsCommand = new AsyncRelayCommand(GenerateSplitsAsync);
        GateTrainingRunCommand = new AsyncRelayCommand(GateTrainingRunAsync);
        RefreshTrainingCheckpointsCommand = new AsyncRelayCommand(RefreshTrainingCheckpointsAsync);
        LaunchTrainingCommand = new AsyncRelayCommand(LaunchTrainingAsync);
        ResumeTrainingCommand = new AsyncRelayCommand(ResumeTrainingAsync);
        StopTrainingCommand = new RelayCommand(StopTraining);
        RunQualityCommand = new AsyncRelayCommand(() => RefreshQualityAsync());
        SaveGateThresholdsCommand = new AsyncRelayCommand(SaveGateThresholdsAsync);
        RefreshProviderPoliciesCommand = new AsyncRelayCommand(RefreshProviderPoliciesAsync);
        ApproveProviderGenerationCommand = new AsyncRelayCommand(() => ApplyProviderApprovalAsync(revoke: false));
        RevokeProviderGenerationCommand = new AsyncRelayCommand(() => ApplyProviderApprovalAsync(revoke: true));
        RebuildProjectIndexCommand = new AsyncRelayCommand(RebuildProjectIndexAsync);
        ExportPreferenceRankingCommand = new RelayCommand(ExportPreferenceRanking);
        RefreshDatasetVersionsCommand = new AsyncRelayCommand(RefreshDatasetVersionsAsync);
        RestoreDatasetVersionCommand = new AsyncRelayCommand(RestoreDatasetVersionAsync);
        SaveExampleCommand = new AsyncRelayCommand(SaveExampleAsync);
        ImportDatasetCommand = new AsyncRelayCommand(ImportDatasetAsync);
        ImportFromHuggingFaceCommand = new AsyncRelayCommand(ImportFromHuggingFaceAsync);
        LocateEngineCommand = new AsyncRelayCommand(LocateEngineAsync);
        RetryEngineCommand = new AsyncRelayCommand(RetryEngineAsync);
        NewSuiteCommand = new AsyncRelayCommand(CreateSuiteAsync);
        ExportJsonlCommand = new AsyncRelayCommand(ExportJsonlAsync);
        RegisterArtifactFromRunCommand = new RelayCommand(RegisterArtifactFromRun);
        KeepArtifactCommand = new AsyncRelayCommand(KeepArtifactAsync);
        RejectArtifactCommand = new RelayCommand(() => SetSelectedArtifactStatus("rejected"));
        RefreshArtifactsCommand = new RelayCommand(RefreshArtifacts);
        DiffVersionsCommand = new AsyncRelayCommand(DiffVersionsAsync);
        ViewArtifactCardCommand = new AsyncRelayCommand(ViewArtifactCardAsync);
        GenerateDatasetCardCommand = new AsyncRelayCommand(GenerateDatasetCardAsync);
        ValidateDraftCommand = new AsyncRelayCommand(ValidateDraftAsync);
        CheckTrainingCompatibilityCommand = new AsyncRelayCommand(CheckTrainingCompatibilityAsync);
        RefreshSuitesCommand = new AsyncRelayCommand(RefreshSuitesAsync);
        ExportPreferenceForTrainingCommand = new AsyncRelayCommand(ExportPreferenceForTrainingAsync);
        GoToDebtCommand = new RelayCommand(() => GoToStudioTab(StudioTab.Debt));
        GoToWritingStudioCommand = new RelayCommand(() => GoToStudioTab(StudioTab.WritingStudio));
        GoToSplitsCommand = new RelayCommand(() => GoToStudioTab(StudioTab.Splits));
        GoToEvaluationCommand = new RelayCommand(() => GoToStudioTab(StudioTab.Evaluation));
        GoToTrainingCommand = new RelayCommand(() => GoToStudioTab(StudioTab.Training));
        PrepareEvaluationFailureReviewCommand = new RelayCommand(() => PrepareEvaluationFailureReview());
        PreparePreferenceJudgeReviewCommand = new RelayCommand(() => PreparePreferenceJudgeReview());
        PreparePreferenceBatchJudgeReviewCommand = new RelayCommand(() => PreparePreferenceBatchJudgeReview());
    }

    // ---- Engine availability (v1.2.15 distributability) --------------------------
    // When the Python engine tree can't be found, the shell shows a setup screen instead
    // of crashing. Set from PythonEngineService.IsEngineAvailable at startup.

    private bool _isEngineUnavailable;
    private string _engineUnavailableMessage = string.Empty;

    public bool IsEngineUnavailable
    {
        get => _isEngineUnavailable;
        private set => SetField(ref _isEngineUnavailable, value);
    }

    public string EngineUnavailableMessage
    {
        get => _engineUnavailableMessage;
        private set => SetField(ref _engineUnavailableMessage, value);
    }

    public void SetEngineUnavailable(string? reason)
    {
        EngineUnavailableMessage = string.IsNullOrWhiteSpace(reason)
            ? "The Python engine could not be found."
            : reason!;
        IsEngineUnavailable = true;
    }

    public void ClearEngineUnavailable()
    {
        IsEngineUnavailable = false;
        EngineUnavailableMessage = string.Empty;
    }

    public WorkspaceShellMode ShellMode
    {
        get => _shellMode;
        private set
        {
            if (SetField(ref _shellMode, value))
            {
                OnPropertyChanged(nameof(IsStartCenter));
                OnPropertyChanged(nameof(IsFiles));
                OnPropertyChanged(nameof(IsStudio));
            }
        }
    }

    public bool IsStartCenter => _shellMode == WorkspaceShellMode.StartCenter;
    public bool IsFiles => _shellMode == WorkspaceShellMode.Files;
    public bool IsStudio => _shellMode == WorkspaceShellMode.Studio;

    /// <summary>Show the full-window Start Center (refreshes the recent list first so a
    /// workspace opened elsewhere and any missing-path flags are current).</summary>
    public void ShowStartCenter()
    {
        StartCenter.Refresh();
        ShellMode = WorkspaceShellMode.StartCenter;
    }

    public void ShowFiles()
    {
        // Point the explorer at the active project's folder (built lazily here so we don't
        // touch disk on every project select). Unchanged root keeps the tree + open tabs.
        if (HasActiveProject)
        {
            Explorer.SetWorkspaceRoot(ActiveProjectPath, ActiveProjectTitle);
        }
        else
        {
            Explorer.Reset();
        }

        ShellMode = WorkspaceShellMode.Files;
    }

    public void ShowStudio() => ShellMode = WorkspaceShellMode.Studio;

    private int _selectedStudioTabIndex;

    /// <summary>The selected Studio tab, bound by the WPF head's TabControl <c>SelectedIndex</c>. Cross-tab
    /// navigation sets this in the view-model (via <see cref="GoToStudioTab"/> + commands) rather than
    /// code-behind. See <see cref="StudioTab"/> for the index meanings.</summary>
    public int SelectedStudioTabIndex
    {
        get => _selectedStudioTabIndex;
        set => SetField(ref _selectedStudioTabIndex, value);
    }

    /// <summary>Switch to the Studio view and select <paramref name="tab"/>. The shared target for the
    /// cross-tab navigation commands (replacing the desktop's <c>XxxTab.IsSelected = true</c> code-behind).</summary>
    public void GoToStudioTab(StudioTab tab)
    {
        ShowStudio();
        SelectedStudioTabIndex = (int)tab;
    }

    /// <summary>Assess dataset debt via the engine and apply it to the Debt tab. Moved verbatim from the
    /// desktop code-behind (minus the WPF-only wait cursor) so the run is a shared, bindable async command
    /// (see <see cref="RunDatasetDebtCommand"/>). Read-only: the engine computes/grades, the VM parses.</summary>
    public async System.Threading.Tasks.Task RunDatasetDebtAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Debt.SetDebtError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ActiveProjectPath;
        try
        {
            SetBusy("Assessing dataset debt...");
            var report = await _engine.GetDatasetDebtAsync(projectPath);
            Debt.ApplyDebtReport(report);
        }
        catch (System.Exception ex)
        {
            Debt.SetDebtError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Run the dataset gate suite and apply the report. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RunGatesAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            SetGateError("Create or select a dataset project before running gates.");
            return;
        }

        try
        {
            SetBusy("Running gates...");
            SetGateInProgress();
            var report = await _engine.RunDatasetGatesAsync(ActiveProjectPath, ActiveSchemaId);
            ApplyGateReport(report);
        }
        catch (System.Exception ex)
        {
            SetGateError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Run the chat conversation-structure gate and apply the report. Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task RunChatGatesAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            SetGateError("Create or select a dataset project before running gates.");
            return;
        }

        try
        {
            SetBusy("Running chat gates...");
            SetGateInProgress();
            var report = await _engine.RunChatGatesAsync(ActiveProjectPath);
            ApplyGateReport(report);
        }
        catch (System.Exception ex)
        {
            SetGateError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Run the model arena across the entered prompts/models. Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task RunArenaAsync()
    {
        var models = ArenaViewModel.ParseModelList(Arena.ArenaModelsInput);
        if (string.IsNullOrWhiteSpace(Arena.ArenaPromptsInput))
        {
            Arena.SetArenaError("Enter at least one prompt (one per line).");
            return;
        }

        if (models.Count == 0)
        {
            Arena.SetArenaError("Enter at least one model (comma or newline separated).");
            return;
        }

        try
        {
            SetBusy("Running arena...");
            Arena.SetArenaInProgress();
            var judge = string.IsNullOrWhiteSpace(Arena.ArenaJudgeModelInput)
                ? null
                : Arena.ArenaJudgeModelInput.Trim();
            var projectPath = HasActiveProject ? ActiveProjectPath : null;
            var report = await _engine.RunArenaAsync(Arena.ArenaPromptsInput, models, judge, projectPath);
            Arena.ApplyArenaReport(report);
        }
        catch (System.Exception ex)
        {
            Arena.SetArenaError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Run the selected evaluation suite (live backend). Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task RunSuiteAsync()
    {
        if (!Suites.CanRunSuite || Suites.SelectedSuite is not { } suite
            || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return;
        }

        try
        {
            Suites.IsSuitesBusy = true;
            SetBusy($"Running suite '{suite.Name}' (live backend evaluations)...");
            Suites.ApplySuiteReport(await _engine.RunSuiteAsync(ActiveProjectPath, suite.Name));
            await LoadSuiteHistoryAsync(suite.Name);
        }
        catch (System.Exception ex)
        {
            Suites.SetSuitesError(ex.Message);
        }
        finally
        {
            Suites.IsSuitesBusy = false;
            ClearBusy();
        }
    }

    /// <summary>Load a suite's run-history trend into the Suites tab. Best-effort: a history read
    /// failure must not clobber the report or the tab with an error banner.</summary>
    private async System.Threading.Tasks.Task LoadSuiteHistoryAsync(string suiteName)
    {
        if (string.IsNullOrWhiteSpace(ActiveProjectPath) || string.IsNullOrWhiteSpace(suiteName))
        {
            return;
        }
        try
        {
            Suites.SetSuiteHistory(await _engine.GetSuiteHistoryAsync(ActiveProjectPath, suiteName));
        }
        catch (System.Exception)
        {
            // History is a secondary trend view; leave the previous history rather than surface an error.
        }
    }

    /// <summary>Generate a training config export. Moved from the desktop code-behind; the options
    /// are read + validated from the Training tab's own fields (no View coupling).</summary>
    public async System.Threading.Tasks.Task GenerateTrainingConfigAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Training.SetTrainingConfigError("Create or select a dataset project before generating a training config.");
            return;
        }

        if (!TryGetTrainingConfigOptions(
            out var target, out var baseModel, out var datasetFormat, out var sequenceLen,
            out var loraR, out var loraAlpha, out var microBatchSize, out var gradientAccumulationSteps,
            out var learningRate, out var errorMessage))
        {
            Training.SetTrainingConfigError(errorMessage);
            return;
        }

        try
        {
            SetBusy("Generating training config...");
            Training.SetTrainingConfigInProgress();
            var result = await _engine.GenerateTrainingConfigAsync(
                ActiveProjectPath, ActiveSchemaId, target, baseModel, datasetFormat, sequenceLen,
                loraR, loraAlpha, microBatchSize, gradientAccumulationSteps, learningRate);
            Training.ApplyTrainingConfigExportResult(result);
        }
        catch (System.Exception ex)
        {
            Training.SetTrainingConfigError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    private bool TryGetTrainingConfigOptions(
        out string target,
        out string baseModel,
        out string datasetFormat,
        out int sequenceLen,
        out int loraR,
        out int loraAlpha,
        out int microBatchSize,
        out int gradientAccumulationSteps,
        out double learningRate,
        out string errorMessage
    )
    {
        target = Training.TrainingTarget.Trim();
        baseModel = Training.TrainingBaseModel.Trim();
        datasetFormat = Training.TrainingFormat.Trim();
        sequenceLen = 0;
        loraR = 0;
        loraAlpha = 0;
        microBatchSize = 0;
        gradientAccumulationSteps = 0;
        learningRate = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(target))
        {
            errorMessage = "Training target is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(baseModel))
        {
            errorMessage = "Training base model is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(datasetFormat))
        {
            errorMessage = "Training format is required.";
            return false;
        }

        if (!int.TryParse(
            Training.TrainingSequenceLen,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out sequenceLen
        ) || sequenceLen <= 0)
        {
            errorMessage = "Training sequence length must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            Training.TrainingLoraR,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out loraR
        ) || loraR <= 0)
        {
            errorMessage = "Training LoRA r must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            Training.TrainingLoraAlpha,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out loraAlpha
        ) || loraAlpha <= 0)
        {
            errorMessage = "Training LoRA alpha must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            Training.TrainingMicroBatchSize,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out microBatchSize
        ) || microBatchSize <= 0)
        {
            errorMessage = "Training micro batch size must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            Training.TrainingGradientAccumulationSteps,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out gradientAccumulationSteps
        ) || gradientAccumulationSteps <= 0)
        {
            errorMessage = "Training gradient accumulation steps must be a positive whole number.";
            return false;
        }

        if (!double.TryParse(
            Training.TrainingLearningRate,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out learningRate
        ) || !double.IsFinite(learningRate) || learningRate <= 0)
        {
            errorMessage = "Training learning rate must be a positive number.";
            return false;
        }

        return true;
    }

    /// <summary>Run a multi-model benchmark. Moved from the desktop code-behind; the models and eval
    /// options come from the tab's own VM fields.</summary>
    public async System.Threading.Tasks.Task RunBenchmarkAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            SetBenchmarkError("Create or select a dataset project before benchmarking.");
            return;
        }

        if (ActiveSchemaId is not ("instruction" or "chat"))
        {
            SetBenchmarkError("Evaluation Lab supports instruction and chat projects.");
            return;
        }

        var models = GetBenchmarkModels();
        if (models.Count == 0)
        {
            SetBenchmarkError("Enter at least one model to benchmark (one per line).");
            return;
        }

        if (!TryGetEvaluationRunOptions(out var backend, out _, out var baseUrl, out var limit,
            out var scoreThreshold, out var timeoutSeconds, out var errorMessage, requireModel: false))
        {
            SetBenchmarkError(errorMessage);
            return;
        }

        try
        {
            SetBusy($"Benchmarking {models.Count} model(s)...");
            var report = await _engine.RunBenchmarkAsync(
                ActiveProjectPath, ActiveSchemaId, backend, models, baseUrl, limit, scoreThreshold, timeoutSeconds);
            ApplyBenchmarkReport(report);
        }
        catch (System.Exception ex)
        {
            SetBenchmarkError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    public bool TryGetEvaluationRunOptions(
        out string backend,
        out string model,
        out string? baseUrl,
        out int? limit,
        out double scoreThreshold,
        out int timeoutSeconds,
        out string errorMessage,
        bool requireModel = true
    )
    {
        backend = EvaluationConnection.EvaluationBackend.Trim();
        model = EvaluationConnection.EvaluationModel.Trim();
        baseUrl = string.IsNullOrWhiteSpace(EvaluationConnection.EvaluationBaseUrl)
            ? null
            : EvaluationConnection.EvaluationBaseUrl.Trim();
        limit = null;
        scoreThreshold = 0;
        timeoutSeconds = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = "Evaluation backend is required.";
            return false;
        }

        if (requireModel && string.IsNullOrWhiteSpace(model))
        {
            errorMessage = "Evaluation model is required.";
            return false;
        }

        if (!string.IsNullOrWhiteSpace(Evaluation.EvaluationLimit))
        {
            if (!int.TryParse(
                Evaluation.EvaluationLimit,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out var parsedLimit
            ) || parsedLimit <= 0)
            {
                errorMessage = "Evaluation limit must be a positive whole number or blank.";
                return false;
            }

            limit = parsedLimit;
        }

        if (!double.TryParse(
            Evaluation.EvaluationScoreThreshold,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out scoreThreshold
        ) || !double.IsFinite(scoreThreshold) || scoreThreshold < 0 || scoreThreshold > 100)
        {
            errorMessage = "Evaluation score threshold must be a number from 0 to 100.";
            return false;
        }

        if (!int.TryParse(
            EvaluationConnection.EvaluationTimeoutSeconds,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds
        ) || timeoutSeconds <= 0)
        {
            errorMessage = "Evaluation timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    /// <summary>Run an evaluation (preflight health check, then the graded run). Moved from the
    /// desktop code-behind; options come from the Evaluation tab's own VM fields.</summary>
    public async System.Threading.Tasks.Task RunEvaluationAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Evaluation.SetEvaluationError("Create or select a dataset project before running evaluation.");
            return;
        }

        if (ActiveSchemaId is not ("instruction" or "chat"))
        {
            Evaluation.SetEvaluationError("Evaluation Lab MVP supports instruction and chat projects.");
            return;
        }

        if (!TryGetEvaluationRunOptions(
            out var backend,
            out var model,
            out var baseUrl,
            out var limit,
            out var scoreThreshold,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            Evaluation.SetEvaluationError(errorMessage);
            return;
        }

        try
        {
            SetBusy("Running evaluation...");
            Evaluation.SetEvaluationPreflightInProgress();
            var healthReport = await _engine.CheckBackendHealthAsync(
                backend,
                model,
                baseUrl,
                timeoutSeconds
            );
            if (!IsEvaluationBackendReady(healthReport))
            {
                Evaluation.SetEvaluationError(FormatEvaluationPreflightError(healthReport));
                return;
            }

            Evaluation.SetEvaluationInProgress();
            Evaluation.BeginEvaluationProgress();
            // Stream per-example progress live: the engine writes '[k/N] evaluated' to stderr; the
            // Progress<string> (created here on the UI thread) marshals each line back to the UI thread.
            var progress = new Progress<string>(line =>
            {
                if (TryParseEvaluationProgress(line, out var completed, out var total))
                {
                    Evaluation.SetEvaluationProgress(completed, total);
                }
            });
            // Opt-in LLM-judge: when a judge model is set, the run scores with metric=llm_judge (the judge
            // reuses this run's backend/base-url). Blank = the default keyword-overlap scorer.
            var judgeModel = EvaluationConnection.EvaluationJudgeModel?.Trim();
            var result = await _engine.RunEvaluationAsync(
                ActiveProjectPath,
                ActiveSchemaId,
                backend,
                model,
                baseUrl,
                limit,
                scoreThreshold,
                timeoutSeconds,
                judgeModel: string.IsNullOrWhiteSpace(judgeModel) ? null : judgeModel,
                progress: progress
            );
            Evaluation.ApplyEvaluationRunResult(result);
            Evaluation.SetEvaluationReportHistory(
                _engine.LoadEvaluationReportHistory(ActiveProjectPath)
            );
            ReconcileReviewedFixesAfterRun(result);
        }
        catch (Exception ex)
        {
            Evaluation.SetEvaluationError(ex.Message);
        }
        finally
        {
            Evaluation.ClearEvaluationProgress();
            ClearBusy();
        }
    }

    private static readonly System.Text.RegularExpressions.Regex EvaluationProgressPattern =
        new(@"^\[(\d+)/(\d+)\]\s+evaluated$", System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>Parse the engine's <c>[k/N] evaluated</c> stderr progress line into (completed, total).
    /// Returns false for any other stderr line (so unrelated output is ignored).</summary>
    public static bool TryParseEvaluationProgress(string line, out int completed, out int total)
    {
        completed = 0;
        total = 0;
        if (string.IsNullOrWhiteSpace(line))
        {
            return false;
        }

        var match = EvaluationProgressPattern.Match(line.Trim());
        return match.Success
            && int.TryParse(match.Groups[1].Value, out completed)
            && int.TryParse(match.Groups[2].Value, out total);
    }

    /// <summary>Rerun the selected saved evaluation report's settings for a regression comparison.
    /// Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RerunEvaluationReportAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Evaluation.SetEvaluationError("Create or select a dataset project before rerunning evaluation.");
            return;
        }

        if (!Evaluation.TryGetSelectedEvaluationRunSettings(
            out var settings,
            out var errorMessage
        ))
        {
            Evaluation.SetEvaluationError(errorMessage);
            return;
        }

        if (settings.SchemaId is not ("instruction" or "chat"))
        {
            Evaluation.SetEvaluationError("Evaluation regression reruns support instruction and chat reports.");
            return;
        }

        if (!string.Equals(settings.SchemaId, ActiveSchemaId, StringComparison.OrdinalIgnoreCase))
        {
            Evaluation.SetEvaluationError(
                $"The selected report uses schema '{settings.SchemaId}', but the active project uses '{ActiveSchemaId}'."
            );
            return;
        }

        var baselineReportPath = Evaluation.SelectedEvaluationReportHistoryItem?.ReportPath;
        try
        {
            SetBusy("Rerunning evaluation...");
            Evaluation.SetEvaluationRegressionRerunPreflightInProgress(settings);
            var healthReport = await _engine.CheckBackendHealthAsync(
                settings.Backend,
                settings.Model,
                settings.BaseUrl,
                settings.TimeoutSeconds
            );
            if (!IsEvaluationBackendReady(healthReport))
            {
                Evaluation.SetEvaluationError(FormatEvaluationPreflightError(healthReport));
                return;
            }

            Evaluation.SetEvaluationRegressionRerunInProgress(settings);
            Evaluation.BeginEvaluationProgress();
            var progress = new Progress<string>(line =>
            {
                if (TryParseEvaluationProgress(line, out var completed, out var total))
                {
                    Evaluation.SetEvaluationProgress(completed, total);
                }
            });
            var result = await _engine.RunEvaluationAsync(
                ActiveProjectPath,
                settings.SchemaId,
                settings.Backend,
                settings.Model,
                settings.BaseUrl,
                settings.Limit,
                settings.ScoreThreshold,
                settings.TimeoutSeconds,
                progress: progress
            );
            Evaluation.ApplyEvaluationRunResult(result);
            Evaluation.SetEvaluationReportHistory(
                _engine.LoadEvaluationReportHistory(ActiveProjectPath)
            );
            ReconcileReviewedFixesAfterRun(result);

            var newItem = Evaluation.EvaluationReportHistory
                .FirstOrDefault(item => item.ReportPath == result.ReportPath);
            var baselineItem = string.IsNullOrWhiteSpace(baselineReportPath)
                ? null
                : Evaluation.EvaluationReportHistory
                    .FirstOrDefault(item => item.ReportPath == baselineReportPath);

            if (newItem is not null)
            {
                Evaluation.SelectedEvaluationReportHistoryItem = newItem;
            }

            if (baselineItem is not null)
            {
                Evaluation.SecondaryEvaluationReportHistoryItem = baselineItem;
                Evaluation.CompareSelectedEvaluationReports();
            }
        }
        catch (Exception ex)
        {
            Evaluation.SetEvaluationError(ex.Message);
        }
        finally
        {
            Evaluation.ClearEvaluationProgress();
            ClearBusy();
        }
    }

    private void ReconcileReviewedFixesAfterRun(EvaluationRunResult result)
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return;
        }

        try
        {
            SetReviewedFixes(
                _engine.ReconcileReviewedFixes(ActiveProjectPath, result.Report.Results)
            );
            ApplyReviewedFixesReconciled();
        }
        catch (Exception ex)
        {
            SetReviewedFixError(ex.Message);
        }
    }

    private static bool IsEvaluationBackendReady(BackendHealthReport report)
    {
        return report.Reachable && report.ModelAvailable;
    }

    private static string FormatEvaluationPreflightError(BackendHealthReport report)
    {
        var lines = new List<string>
        {
            "Pre-run backend health check failed.",
            $"Backend: {report.ProviderName}",
            $"Model: {report.ModelName}",
            $"Base URL: {report.BaseUrl}",
        };

        if (!report.Reachable)
        {
            lines.Add("Backend is not reachable.");
        }
        else if (!report.ModelAvailable)
        {
            lines.Add("The configured model was not listed by the backend.");
        }

        if (report.AvailableModels.Count > 0)
        {
            lines.Add($"Available models: {string.Join(", ", report.AvailableModels.Take(5))}");
        }

        if (!string.IsNullOrWhiteSpace(report.Error))
        {
            lines.Add($"Error: {report.Error}");
        }

        return string.Join(Environment.NewLine, lines);
    }

    /// <summary>Preflight-check the Evaluation backend/model (reachability + model availability). Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task CheckEvaluationBackendAsync()
    {
        if (!TryReadBackendOptions(
            EvaluationConnection.EvaluationBackend,
            EvaluationConnection.EvaluationModel,
            EvaluationConnection.EvaluationBaseUrl,
            EvaluationConnection.EvaluationTimeoutSeconds,
            "Evaluation",
            out var backend,
            out var model,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            Evaluation.SetEvaluationError(errorMessage);
            return;
        }

        try
        {
            SetBusy("Checking evaluation backend...");
            SetEvaluationHealthCheckInProgress();
            var report = await _engine.CheckBackendHealthAsync(
                backend,
                model,
                baseUrl,
                timeoutSeconds
            );
            ApplyEvaluationBackendHealthReport(report);
        }
        catch (Exception ex)
        {
            Evaluation.SetEvaluationError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>List the Evaluation backend's available models. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RefreshEvaluationModelsAsync()
    {
        if (!TryReadModelListOptions(
            EvaluationConnection.EvaluationBackend,
            EvaluationConnection.EvaluationBaseUrl,
            EvaluationConnection.EvaluationTimeoutSeconds,
            "Evaluation",
            out var backend,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            SetEvaluationModelListError(errorMessage);
            return;
        }

        try
        {
            SetBusy("Loading models...");
            SetEvaluationModelListInProgress();
            var report = await _engine.ListBackendModelsAsync(
                backend,
                baseUrl,
                timeoutSeconds
            );
            ApplyEvaluationModelListReport(report);
        }
        catch (Exception ex)
        {
            SetEvaluationModelListError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Preflight-check the AI-Assist backend/model. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task CheckAiAssistBackendAsync()
    {
        if (!TryReadBackendOptions(
            AiAssistConnection.AiAssistBackend,
            AiAssistConnection.AiAssistModel,
            AiAssistConnection.AiAssistBaseUrl,
            AiAssistConnection.AiAssistTimeoutSeconds,
            "AI Assist",
            out var backend,
            out var model,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            AiAssist.SetAiAssistError(errorMessage);
            return;
        }

        try
        {
            SetBusy("Checking AI Assist backend...");
            SetAiAssistHealthCheckInProgress();
            var report = await _engine.CheckBackendHealthAsync(
                backend,
                model,
                baseUrl,
                timeoutSeconds
            );
            ApplyAiAssistBackendHealthReport(report);
        }
        catch (Exception ex)
        {
            AiAssist.SetAiAssistError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>List the AI-Assist backend's available models. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RefreshAiAssistModelsAsync()
    {
        if (!TryReadModelListOptions(
            AiAssistConnection.AiAssistBackend,
            AiAssistConnection.AiAssistBaseUrl,
            AiAssistConnection.AiAssistTimeoutSeconds,
            "AI Assist",
            out var backend,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            SetAiAssistModelListError(errorMessage);
            return;
        }

        try
        {
            SetBusy("Loading models...");
            SetAiAssistModelListInProgress();
            var report = await _engine.ListBackendModelsAsync(
                backend,
                baseUrl,
                timeoutSeconds
            );
            ApplyAiAssistModelListReport(report);
        }
        catch (Exception ex)
        {
            SetAiAssistModelListError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Run AI Assist over the current draft, queue the suggestion for review, and select it.
    /// Moved from the desktop code-behind (#247); the bulk-undo stack now lives on the AI-Assist VM, so
    /// a fresh run clears it (a prior bulk undo would no longer be coherent).</summary>
    public async System.Threading.Tasks.Task RunAiAssistAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            AiAssist.SetAiAssistError("Create or select a dataset project before running AI Assist.");
            return;
        }

        if (string.IsNullOrWhiteSpace(WritingStudio.DraftText))
        {
            AiAssist.SetAiAssistError("Add a draft example before running AI Assist.");
            return;
        }

        if (!TryReadAiAssistOptions(
            out var backend, out var model, out var baseUrl, out var action,
            out var timeoutSeconds, out var instruction, out var errorMessage))
        {
            AiAssist.SetAiAssistError(errorMessage);
            return;
        }

        try
        {
            SetBusy("Running AI Assist...");
            AiAssist.SetAiAssistInProgress();
            var result = await _engine.RunAiAssistAsync(
                WritingStudio.DraftText, ActiveSchemaId, action, backend, model, baseUrl, timeoutSeconds, instruction);
            AiAssist.ApplyAiAssistRunResult(result);
            var queuedItem = _engine.SaveAiAssistReviewQueueItem(ActiveProjectPath, WritingStudio.DraftText, result);
            AiAssist.SetAiAssistReviewQueue(_engine.LoadAiAssistReviewQueue(ActiveProjectPath));
            AiAssist.SelectedAiAssistReviewQueueItem = AiAssist.AiAssistReviewQueue
                .FirstOrDefault(item => item.ReviewId == queuedItem.ReviewId);
            AiAssist.ClearBulkUndoStack();
        }
        catch (Exception ex)
        {
            AiAssist.SetAiAssistError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Read + validate the AI-Assist run options from the connection/tab view-models.
    /// Moved verbatim from the code-behind (reads only VM state, so it is head-agnostic).</summary>
    private bool TryReadAiAssistOptions(
        out string backend, out string model, out string? baseUrl, out string action,
        out int timeoutSeconds, out string? instruction, out string errorMessage)
    {
        backend = AiAssistConnection.AiAssistBackend.Trim();
        model = AiAssistConnection.AiAssistModel.Trim();
        baseUrl = string.IsNullOrWhiteSpace(AiAssistConnection.AiAssistBaseUrl)
            ? null
            : AiAssistConnection.AiAssistBaseUrl.Trim();
        action = AiAssist.AiAssistAction.Trim();
        timeoutSeconds = 0;
        instruction = string.IsNullOrWhiteSpace(AiAssist.AiAssistInstruction)
            ? null
            : AiAssist.AiAssistInstruction.Trim();
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = "AI Assist backend is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(model))
        {
            errorMessage = "AI Assist model is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(action))
        {
            errorMessage = "AI Assist action is required.";
            return false;
        }

        if (!int.TryParse(
            AiAssistConnection.AiAssistTimeoutSeconds,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds) || timeoutSeconds <= 0)
        {
            errorMessage = "AI Assist timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    private static bool TryReadBackendOptions(
        string backendText,
        string modelText,
        string baseUrlText,
        string timeoutText,
        string label,
        out string backend,
        out string model,
        out string? baseUrl,
        out int timeoutSeconds,
        out string errorMessage
    )
    {
        backend = backendText.Trim();
        model = modelText.Trim();
        baseUrl = string.IsNullOrWhiteSpace(baseUrlText) ? null : baseUrlText.Trim();
        timeoutSeconds = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = $"{label} backend is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(model))
        {
            errorMessage = $"{label} model is required.";
            return false;
        }

        if (!int.TryParse(
            timeoutText,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds
        ) || timeoutSeconds <= 0)
        {
            errorMessage = $"{label} timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    private static bool TryReadModelListOptions(
        string backendText,
        string baseUrlText,
        string timeoutText,
        string label,
        out string backend,
        out string? baseUrl,
        out int timeoutSeconds,
        out string errorMessage
    )
    {
        backend = backendText.Trim();
        baseUrl = string.IsNullOrWhiteSpace(baseUrlText) ? null : baseUrlText.Trim();
        timeoutSeconds = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = $"{label} backend is required.";
            return false;
        }

        if (!int.TryParse(
            timeoutText,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds
        ) || timeoutSeconds <= 0)
        {
            errorMessage = $"{label} timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    /// <summary>Export the project's examples to JSONL with the two cleaning options, then report the
    /// result (rows written, cleaning, warnings) via the IDialogService seam. Moved from the code-behind.</summary>
    public async System.Threading.Tasks.Task ExportJsonlAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            await _dialogs.ShowAsync("Create a dataset project before exporting.", "Corpus Studio", DialogSeverity.Information);
            return;
        }

        try
        {
            SetBusy("Exporting JSONL...");
            var exportResult = await _engine.ExportProjectExamplesAsync(
                ActiveProjectPath, ActiveSchemaId, ExportRemoveDuplicates, ExportRemoveLowInformation, ExportRedactPii, ExportFormat);

            var message = $"Exported {exportResult.OutputRows} row(s) to:{System.Environment.NewLine}{exportResult.OutputPath}";
            if (exportResult.Cleaned && exportResult.RemovedRows > 0)
            {
                message += $"{System.Environment.NewLine}Removed {exportResult.RemovedRows} row(s) during cleaning.";
            }
            if (exportResult.Warnings.Count > 0)
            {
                message += $"{System.Environment.NewLine}{System.Environment.NewLine}Warnings:{System.Environment.NewLine}- "
                    + string.Join($"{System.Environment.NewLine}- ", exportResult.Warnings);
            }

            await _dialogs.ShowAsync(message, "Export Complete",
                exportResult.Warnings.Count > 0 ? DialogSeverity.Warning : DialogSeverity.Information);
        }
        catch (System.Exception ex)
        {
            SetValidationError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Scaffold a new evaluation suite from the name typed into the Suites tab, then refresh the
    /// list (and clear the box). Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task CreateSuiteAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Suites.SetSuitesError("Create or select a dataset project first.");
            return;
        }

        var name = Suites.NewSuiteName?.Trim() ?? string.Empty;
        if (name.Length == 0)
        {
            Suites.SetSuitesError("Enter a suite name to create.");
            return;
        }

        try
        {
            Suites.IsSuitesBusy = true;
            await _engine.NewSuiteAsync(ActiveProjectPath, name);
            Suites.NewSuiteName = string.Empty;
            Suites.ApplySuites(await _engine.ListSuitesAsync(ActiveProjectPath));
            Suites.SetSuitesError($"Created suite '{name}'. Open evaluation_suites/{name}.json in Files to edit its cases.");
        }
        catch (System.Exception ex)
        {
            Suites.SetSuitesError(ex.Message);
        }
        finally
        {
            Suites.IsSuitesBusy = false;
        }
    }

    /// <summary>Import rows from a public Hugging Face dataset: show the HF import dialog (via the
    /// head-agnostic seam) mapped to the active project's built-in schema, then run the staged JSONL
    /// through the SAME preview/confirm/quarantine flow as any import, cleaning up the temp file.
    /// Moved from the desktop code-behind (#250); the modal window is now behind IHuggingFaceImportDialog.</summary>
    public async System.Threading.Tasks.Task ImportFromHuggingFaceAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            await _dialogs.ShowAsync("Create or select a dataset project before importing.", "Corpus Studio", DialogSeverity.Information);
            return;
        }

        // Map HF columns to the ACTIVE project's schema so imported rows match the project.
        IReadOnlyList<DatasetSchema> schemas;
        try
        {
            SetBusy("Loading schemas...");
            schemas = await _engine.GetSchemasAsync();
        }
        catch (Exception ex)
        {
            SetImportError(ex.Message);
            return;
        }
        finally
        {
            ClearBusy();
        }

        var schema = schemas.FirstOrDefault(s => s.Id == ActiveSchemaId);
        if (schema is null)
        {
            await _dialogs.ShowAsync(
                $"The active project's schema ('{ActiveSchemaId}') is not a built-in schema, so Hugging Face import can't map to it.",
                "Import from Hugging Face", DialogSeverity.Information);
            return;
        }

        var staging = await _huggingFaceImportDialog.ShowAsync(schema.Id, schema.Name, schema.Fields);
        if (string.IsNullOrWhiteSpace(staging))
        {
            return; // cancelled or nothing staged
        }

        // Hand the staging file to the SAME preview/confirm/append+quarantine flow as any JSONL
        // import (the desktop is the single writer of examples.jsonl), then clean up the temp file.
        try
        {
            await PreviewAndImportJsonlAsync(staging);
        }
        finally
        {
            try
            {
                if (System.IO.File.Exists(staging))
                {
                    System.IO.File.Delete(staging);
                }
            }
            catch (System.IO.IOException)
            {
                // best-effort temp cleanup
            }
        }
    }

    /// <summary>Pick a JSONL file and run it through the shared preview/confirm/import flow.
    /// The picker + info dialog route through the head-agnostic seams. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task ImportDatasetAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            await _dialogs.ShowAsync("Create or select a dataset project before importing.", "Corpus Studio", DialogSeverity.Information);
            return;
        }

        var file = await _filePicker.PickFileAsync(
            "Import Dataset",
            new FilePickerFilter("Dataset files (JSONL, CSV, TSV, Parquet)", "jsonl", "csv", "tsv", "parquet"),
            new FilePickerFilter("JSONL files", "jsonl"),
            new FilePickerFilter("CSV / TSV files", "csv", "tsv"),
            new FilePickerFilter("Parquet files", "parquet"),
            new FilePickerFilter("All files", "*"));
        if (file is null)
        {
            return;
        }

        var extension = System.IO.Path.GetExtension(file).TrimStart('.').ToLowerInvariant();
        if (extension is "csv" or "tsv" or "tab" or "parquet")
        {
            await ImportConvertedFileAsync(file);
            return;
        }

        await PreviewAndImportJsonlAsync(file);
    }

    /// <summary>Convert a CSV/TSV/Parquet file to a temp staging JSONL via the engine, then run it through
    /// the same preview/confirm/quarantine/commit flow as any JSONL import (mirrors the Hugging-Face staging
    /// path). The engine routes by extension; a CSV cell imports as text (so a schema type mismatch
    /// quarantines like a bad JSONL row), while Parquet keeps its column types. Parquet needs the engine's
    /// optional [parquet] extra — a missing dependency surfaces as an import error with the install hint.</summary>
    private async System.Threading.Tasks.Task ImportConvertedFileAsync(string sourcePath)
    {
        var staging = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            $"corpus_studio_import_{System.Guid.NewGuid():N}.jsonl");
        try
        {
            try
            {
                SetBusy("Converting to JSONL for import...");
                await _engine.ConvertTabularToJsonlAsync(sourcePath, staging);
            }
            finally
            {
                ClearBusy();
            }

            await PreviewAndImportJsonlAsync(staging);
        }
        catch (System.Exception ex)
        {
            SetImportError(ex.Message);
        }
        finally
        {
            try
            {
                if (System.IO.File.Exists(staging))
                {
                    System.IO.File.Delete(staging);
                }
            }
            catch (System.IO.IOException)
            {
                // best-effort temp cleanup
            }
        }
    }

    /// <summary>Preview a JSONL import, confirm, then append (quarantining rejects) and snapshot the
    /// dataset. Shared by the file-import and Hugging-Face-staging paths (public so the HF code-behind
    /// flow can hand its staging file to the same logic). Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task PreviewAndImportJsonlAsync(string importPath)
    {
        try
        {
            SetBusy("Importing dataset...");
            SetImportInProgress(importPath);
            var report = await _engine.PreviewImportAsync(importPath, ActiveSchemaId);
            ApplyImportPreview(report);

            if (report.AcceptedRows == 0 && report.RejectedRows == 0)
            {
                await _dialogs.ShowAsync("No importable rows were found.", "Import Preview", DialogSeverity.Information);
                return;
            }

            var confirmed = report.RejectedRows > 0
                ? await _dialogs.ConfirmAsync(BuildPartialImportPrompt(report), "Import Preview", DialogButtons.YesNo, DialogSeverity.Warning)
                : await _dialogs.ConfirmAsync($"Import {report.AcceptedRows} row(s) into {ActiveProjectTitle}?", "Import Preview", DialogButtons.YesNo, DialogSeverity.Question);
            if (!confirmed)
            {
                return;
            }

            var importResult = _engine.CommitJsonlImportToProjectExamples(ActiveProjectPath!, importPath, report);
            SetExamples(_engine.LoadExamples(ActiveProjectPath!));
            Quarantine.SetItems(_engine.LoadImportQuarantineItems(ActiveProjectPath!));
            await RefreshQualityAsync();

            // Snapshot the dataset change so an import is never silent. Best-effort: the import
            // already succeeded, so a failed snapshot is a note, not a failure — never claim a
            // snapshot that didn't happen.
            var snapshotNote = await AutoCaptureAfterImportAsync(importResult);

            await _dialogs.ShowAsync(BuildImportCompleteMessage(importResult, snapshotNote), "Import Complete", DialogSeverity.Information);
        }
        catch (System.Exception ex)
        {
            SetImportError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    private string BuildPartialImportPrompt(ImportPreviewReport report)
    {
        if (report.AcceptedRows == 0)
        {
            return $"No rows can be imported. Save {report.RejectedRows} rejected row(s) to quarantine?";
        }

        return string.Join(
            System.Environment.NewLine,
            [
                $"Import {report.AcceptedRows} valid row(s) into {ActiveProjectTitle}?",
                $"The {report.RejectedRows} rejected row(s) will be saved to quarantine for repair.",
            ]
        );
    }

    private static string BuildImportCompleteMessage(ImportCommitResult result, string? snapshotNote = null)
    {
        var lines = new List<string>
        {
            $"Imported {result.ImportedCount} row(s).",
        };

        if (result.SkippedDuplicateCount > 0)
        {
            lines.Add($"Skipped {result.SkippedDuplicateCount} duplicate row(s) already in the dataset.");
        }

        if (result.QuarantinedCount > 0)
        {
            lines.Add($"Quarantined {result.QuarantinedCount} rejected row(s).");
            if (!string.IsNullOrWhiteSpace(result.QuarantinePath))
            {
                lines.Add(result.QuarantinePath);
            }
        }

        if (!string.IsNullOrWhiteSpace(snapshotNote))
        {
            lines.Add(snapshotNote);
        }

        return string.Join(System.Environment.NewLine, lines);
    }

    private async System.Threading.Tasks.Task<string?> AutoCaptureAfterImportAsync(ImportCommitResult importResult)
    {
        if (!importResult.ShouldAutoCapture || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return null;
        }

        string note;
        try
        {
            var version = await _engine.CreateDatasetVersionAsync(ActiveProjectPath!, importResult.AutoCaptureLabel, "import");
            note = $"Snapshotted this import as dataset version {version.VersionId}.";
        }
        catch (System.Exception ex)
        {
            note = $"Note: could not snapshot this import as a dataset version ({ex.Message}).";
        }

        await RefreshDatasetVersionsAsync();
        return note;
    }

    /// <summary>Validate the current draft and, if valid, append it to the project's examples (clearing a
    /// repaired quarantine row and refreshing quality). Info dialogs route through the IDialogService seam.
    /// Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task SaveExampleAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            await _dialogs.ShowAsync("Create a dataset project before saving examples.", "Corpus Studio", DialogSeverity.Information);
            return;
        }

        try
        {
            SetValidationInProgress();

            var report = await _engine.ValidateDraftAsync(WritingStudio.DraftText, ActiveSchemaId);
            ApplyValidationReport(report);
            if (!report.Valid)
            {
                return;
            }

            var savedCount = _engine.AppendDraftToProjectExamples(ActiveProjectPath, WritingStudio.DraftText);

            SetExamples(_engine.LoadExamples(ActiveProjectPath));
            WritingStudio.MarkDraftClean(); // the draft is now persisted — no longer unsaved work

            // If this save repaired a quarantined row, clear that record so it doesn't orphan.
            var retried = TakePendingRetryItem();
            if (retried is not null)
            {
                _engine.RemoveImportQuarantineItem(retried);
                Quarantine.SetItems(_engine.LoadImportQuarantineItems(ActiveProjectPath));
            }

            await RefreshQualityAsync();
            await _dialogs.ShowAsync($"Saved {savedCount} example(s).", "Example Saved", DialogSeverity.Information);
        }
        catch (System.Exception ex)
        {
            SetValidationError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Restore the selected dataset version in place (after a confirm), capturing the current
    /// dataset as an undo version first. The confirm routes through the head-agnostic IDialogService
    /// seam so the VM owns the flow (no View coupling). Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RestoreDatasetVersionAsync()
    {
        if (Versions.SelectedDatasetVersion is not { } selected)
        {
            Versions.SetDatasetVersionError("Select a version to restore.");
            return;
        }
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ActiveProjectPath;

        // Confirm — this overwrites the current dataset. The dialog is honest about the
        // undo capture and the canonical caveat.
        var confirmed = await _dialogs.ConfirmAsync(
            VersionsViewModel.BuildRestoreConfirmation(selected, Examples.Items.Count),
            "Restore version",
            DialogButtons.YesNo,
            DialogSeverity.Warning);
        if (!confirmed)
        {
            return;
        }

        try
        {
            SetBusy("Restoring version (capturing the current dataset first)...");

            // The service captures the current dataset as an undo version, then restores
            // the selected version to a verified temp and atomically swaps it in. Any
            // failure before the swap leaves examples.jsonl untouched.
            var result = await _engine.RestoreDatasetVersionInPlaceAsync(
                projectPath, selected.Record.VersionId, VersionsViewModel.BuildRestoreUndoLabel(selected));

            // Reflect the restored dataset (and the flipped integrity badges) in the UI.
            SetExamples(_engine.LoadExamples(projectPath));
            Versions.ApplyRestoreResult(result);
            await RefreshDatasetVersionsAsync();
        }
        catch (System.Exception ex)
        {
            Versions.SetDatasetVersionError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Export the visible preference-review ranking to a file. Moved from the desktop code-behind.</summary>
    public void ExportPreferenceRanking()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            PreferenceReview.SetPreferenceRankingExportError("Create or select a preference project before exporting rankings.");
            return;
        }

        try
        {
            var items = PreferenceReview.GetVisiblePreferenceReviewItems();
            var outputPath = _engine.ExportPreferenceRanking(ActiveProjectPath, items);
            PreferenceReview.ApplyPreferenceRankingExport(outputPath, items.Count);
        }
        catch (System.Exception ex)
        {
            PreferenceReview.SetPreferenceRankingExportError(ex.Message);
        }
    }

    /// <summary>Register the newest training run's output directory as a model artifact.
    /// Moved from the desktop code-behind.</summary>
    public void RegisterArtifactFromRun()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Artifacts.SetArtifactError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ActiveProjectPath;
        try
        {
            var run = _engine.LoadTrainingRunRecords(projectPath).FirstOrDefault();
            if (run is null)
            {
                Artifacts.SetArtifactError("No training run has been recorded yet.");
                return;
            }
            if (string.IsNullOrWhiteSpace(run.OutputDir))
            {
                Artifacts.SetArtifactError("The latest run has no output directory to register.");
                return;
            }

            _engine.RegisterArtifact(projectPath, run.RunId, run.OutputDir);
            RefreshArtifacts();
        }
        catch (System.Exception ex)
        {
            Artifacts.SetArtifactError(ex.Message);
        }
    }

    /// <summary>Promote-gate then keep the selected artifact — the engine re-enforces the gate
    /// authoritatively, so a keep can never bypass it. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task KeepArtifactAsync()
    {
        var selected = Artifacts.SelectedModelArtifact;
        if (selected is null)
        {
            Artifacts.SetArtifactError("Select an artifact first.");
            return;
        }
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return;
        }

        var projectPath = ActiveProjectPath;
        try
        {
            SetBusy("Promote-gating artifact...");

            // Preview the promote gate so the user sees the verdict/reason before writing.
            var report = await _engine.GateArtifactAsync(projectPath, selected.Record.ArtifactId);
            var allowed = Artifacts.ApplyPromoteGate(report);
            if (!allowed)
            {
                return;
            }

            // ...then write through the ENGINE, which re-enforces the gate authoritatively — the
            // keep can never bypass it (a block throws and is surfaced below).
            await _engine.PromoteArtifactAsync(projectPath, selected.Record.ArtifactId);
            RefreshArtifacts();
        }
        catch (System.Exception ex)
        {
            Artifacts.SetArtifactError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Set the selected artifact's status directly (candidate/rejected — never "kept",
    /// which must go through the gated promote path). Moved from the desktop code-behind.</summary>
    private void SetSelectedArtifactStatus(string status)
    {
        var selected = Artifacts.SelectedModelArtifact;
        if (selected is null)
        {
            Artifacts.SetArtifactError("Select an artifact first.");
            return;
        }
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return;
        }

        try
        {
            _engine.UpdateArtifactStatus(ActiveProjectPath, selected.Record.ArtifactId, status);
            RefreshArtifacts();
        }
        catch (System.Exception ex)
        {
            Artifacts.SetArtifactError(ex.Message);
        }
    }

    /// <summary>Reload the artifact list, resolving each artifact's base_model live via its run_id
    /// (never stored on the artifact). Moved from the desktop code-behind.</summary>
    public void RefreshArtifacts()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Artifacts.SetArtifactError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ActiveProjectPath;
        try
        {
            // Resolve base_model live through run_id (never stored on the artifact).
            var runs = _engine.LoadTrainingRunRecords(projectPath)
                .ToDictionary(r => r.RunId, r => r, StringComparer.Ordinal);
            var items = _engine.LoadArtifacts(projectPath)
                .Select(entry => new ArtifactDisplayItem(
                    entry.Record,
                    entry.Integrity,
                    runs.TryGetValue(entry.Record.RunId, out var run) ? run.BaseModel : string.Empty))
                .ToList();
            Artifacts.ApplyArtifacts(items);
        }
        catch (System.Exception ex)
        {
            Artifacts.SetArtifactError(ex.Message);
        }
    }

    /// <summary>Rebuild the project index from disk and reload the project list (preserving the
    /// current selection where possible). Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RebuildProjectIndexAsync()
    {
        var selectedId = SelectedProject?.Id;
        try
        {
            SetBusy("Rebuilding project index...");
            var result = await _engine.RebuildProjectIndexAsync();
            SetProjects(await _engine.LoadProjectsFromIndexAsync());
            SelectedProject = Projects.FirstOrDefault(project => project.Id == selectedId);
            ApplyProjectIndexRebuilt(result);
        }
        catch (System.Exception ex)
        {
            SetProjectIndexError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Save the edited gate thresholds (the engine validates ranges and rejects bad values).
    /// Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task SaveGateThresholdsAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Settings.SetGateThresholdsError("Create or select a dataset project first.");
            return;
        }

        try
        {
            SetBusy("Saving gate thresholds...");
            // The engine validates ranges and rejects a bad value, so an invalid edit surfaces here.
            await _engine.SetGateThresholdsAsync(ActiveProjectPath, Settings.GateThresholds);
            Settings.SetGateThresholdsSaved();
        }
        catch (System.Exception ex)
        {
            Settings.SetGateThresholdsError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Approve (or revoke) the Settings-tab provider/model for generating trainable rows, then
    /// refresh the policy list. Provider + model come from the tab's own bound fields. Moved from the
    /// desktop code-behind.</summary>
    public async System.Threading.Tasks.Task ApplyProviderApprovalAsync(bool revoke)
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Settings.SetProviderPolicyError("Create or select a dataset project first.");
            return;
        }

        var provider = Settings.ProviderApprovalProvider?.Trim() ?? string.Empty;
        var model = Settings.ProviderApprovalModel?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(provider) || string.IsNullOrWhiteSpace(model))
        {
            Settings.SetProviderPolicyError("Choose a provider and enter a model name.");
            return;
        }

        try
        {
            SetBusy(revoke ? "Revoking generation approval..." : "Approving generation...");
            await _engine.ApproveProviderGenerationAsync(ActiveProjectPath, provider, model, revoke);
            await RefreshProviderPoliciesAsync();
        }
        catch (System.Exception ex)
        {
            Settings.SetProviderPolicyError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Reload the project's provider generation policies into the Settings tab.
    /// Moved from the desktop code-behind (public so the approve/revoke flow can refresh too).</summary>
    public async System.Threading.Tasks.Task RefreshProviderPoliciesAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Settings.SetProviderPolicyError("Create or select a dataset project first.");
            return;
        }

        try
        {
            var policies = await _engine.GetProviderPoliciesAsync(ActiveProjectPath);
            Settings.ApplyProviderPolicies(policies);
        }
        catch (System.Exception ex)
        {
            Settings.SetProviderPolicyError(ex.Message);
        }
    }

    /// <summary>Run the training-run regression gate: link the newest post-training eval to the newest
    /// run (carrying the model id for provenance), then gate it. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task GateTrainingRunAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Training.SetTrainingRunGateError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ActiveProjectPath;
        try
        {
            SetBusy("Running regression gate...");

            // Link the newest post-training eval (not the baseline) to the newest
            // run, carrying the model id so provenance can be verified.
            var baseline = Training.TrainingBaselineReport;
            var after = _engine.LoadEvaluationReportHistory(projectPath).FirstOrDefault(item =>
                baseline is null
                || !string.Equals(item.ReportPath, baseline.ReportPath, StringComparison.OrdinalIgnoreCase));

            var runId = after is not null
                ? _engine.LinkAfterEvalToNewestRun(projectPath, after.ReportPath, after.Report.Model)
                : _engine.LoadTrainingRunRecords(projectPath).FirstOrDefault()?.RunId;

            if (runId is null)
            {
                Training.SetTrainingRunGateError("No training run has been recorded yet.");
                return;
            }

            var report = await _engine.RunTrainingRunGateAsync(projectPath, runId);
            Training.ApplyTrainingRunGate(report);
            Training.ApplyTrainingRunHistory(_engine.LoadTrainingRunRecords(projectPath));
        }
        catch (System.Exception ex)
        {
            Training.SetTrainingRunGateError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Create a UI-thread background timer (via the head's factory) that fires
    /// <paramref name="onTick"/> every <paramref name="interval"/> — used by the training-run launch to
    /// flush streamed log lines and poll checkpoints without the View owning a DispatcherTimer (#246).</summary>
    private IDispatcherTimer CreateBackgroundTimer(TimeSpan interval, EventHandler onTick)
    {
        var timer = _dispatcherTimerFactory.Create();
        timer.Interval = interval;
        timer.Tick += onTick;
        return timer;
    }

    /// <summary>List the checkpoints/adapters the current run's output dir contains and surface them.
    /// Moved from the desktop code-behind (#248); PUBLIC because the live-run poll timer + the run
    /// finalizer also call it. Advisory — swallows errors and no-ops without an output dir so it can
    /// never disrupt a run (no busy overlay: it polls silently while training streams).</summary>
    public async System.Threading.Tasks.Task RefreshTrainingCheckpointsAsync()
    {
        var outputDirectory = Training.TrainingOutputDirectory;
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            return;
        }

        try
        {
            var result = await _engine.GetTrainingCheckpointsAsync(
                outputDirectory,
                string.IsNullOrWhiteSpace(Training.TrainingTarget) ? "axolotl" : Training.TrainingTarget,
                Training.TrainingConfigPath
            );
            Training.ApplyTrainingCheckpoints(result);
        }
        catch
        {
            // Checkpoint refresh is advisory; never let it disrupt a run.
        }
    }

    /// <summary>Launch a fresh training run (the generated config's command).</summary>
    public System.Threading.Tasks.Task LaunchTrainingAsync() =>
        RunTrainingAsync(Training.TrainingLaunchArgv, Training.TrainingLaunchCommand);

    /// <summary>Resume a training run from a checkpoint (the resume command).</summary>
    public System.Threading.Tasks.Task ResumeTrainingAsync() =>
        RunTrainingAsync(Training.TrainingResumeArgv, Training.TrainingResumeCommand);

    /// <summary>Shared launch core for fresh runs and resume-from-checkpoint: the user confirms the
    /// exact command, then the trainer is spawned + streamed. Moved from the desktop code-behind (#246);
    /// the process spawn/stream is behind IProcessRunner and the log-flush / checkpoint-poll timers behind
    /// the IDispatcherTimer seam, so the launch orchestration is head-agnostic + testable. The trainer
    /// runs on the user's machine with their tools — the engine never runs it.</summary>
    private async System.Threading.Tasks.Task RunTrainingAsync(IReadOnlyList<string> argv, string command)
    {
        if (argv.Count == 0)
        {
            await _dialogs.ShowAsync(
                "Generate a training config first — the launch command is produced with it.",
                "Corpus Studio", DialogSeverity.Information);
            return;
        }

        if (Training.IsTrainingRunning)
        {
            return;
        }

        var confirmed = await _dialogs.ConfirmAsync(
            "This runs the trainer on your machine (it can use significant CPU/GPU for a long "
                + "time) with your installed tools. Corpus Studio only launches the command below "
                + "and streams its output.\n\n"
                + command
                + "\n\nRun it now?",
            "Launch training", DialogButtons.OkCancel, DialogSeverity.Warning);
        if (!confirmed)
        {
            return;
        }

        // Capture the pre-training baseline (newest saved eval report, if any) so
        // the run can be compared before/after once the trained model is evaluated.
        try
        {
            var baseline = HasActiveProject && !string.IsNullOrWhiteSpace(ActiveProjectPath)
                ? _engine.LoadEvaluationReportHistory(ActiveProjectPath).FirstOrDefault()
                : null;
            Training.SetTrainingBaseline(baseline);
        }
        catch
        {
            Training.SetTrainingBaseline(null);
        }

        var workingDirectory = Training.TrainingLaunchWorkingDirectory;
        var cts = new System.Threading.CancellationTokenSource();
        _trainingRunCts = cts;
        _trainingCancelRequested = false;
        while (_trainingLogQueue.TryDequeue(out _)) { } // discard any residual lines
        var runId = Training.BeginTrainingRun();

        // Durable run record (v0.8): recorded to the project's training_runs/.
        var runProjectPath = HasActiveProject ? ActiveProjectPath : null;
        // Reproducibility manifest (dataset fingerprint / config hash / engine+platform) captured at
        // run start. Best-effort: a manifest failure must not block the run — it just leaves it absent.
        RunProvenance? provenance = null;
        if (!string.IsNullOrWhiteSpace(runProjectPath)
            && !string.IsNullOrWhiteSpace(Training.TrainingConfigPath))
        {
            try
            {
                provenance = await _engine.BuildRunProvenanceAsync(runProjectPath, Training.TrainingConfigPath);
            }
            catch
            {
                provenance = null;
            }
        }
        var runRecord = CreateAndSaveRunRecord(runProjectPath, argv, provenance);
        var terminalStatus = "interrupted";
        int? terminalExitCode = null;
        string? terminalNote = null;

        // Coalesce log lines: background reader threads enqueue, and a UI-thread timer flushes at a
        // fixed rate so a chatty trainer can't flood the dispatcher.
        var logTimer = CreateBackgroundTimer(TimeSpan.FromMilliseconds(150), (_, _) => FlushTrainingLogQueue(runId));
        logTimer.Start();

        // Slow poll so checkpoints surface while the run is live (they appear minutes apart).
        var checkpointTimer = CreateBackgroundTimer(
            TimeSpan.FromSeconds(15), async (_, _) => await RefreshTrainingCheckpointsAsync());
        checkpointTimer.Start();

        try
        {
            int? cleanExitCode = null;
            Exception? runError = null;
            try
            {
                cleanExitCode = await _trainingRunner.RunAsync(
                    argv,
                    workingDirectory,
                    _trainingLogQueue.Enqueue,
                    cts.Token,
                    onStarted: (pid, startedAt) => RecordRunPid(runProjectPath, runRecord, pid, startedAt)
                );
            }
            catch (Exception ex)
            {
                runError = ex;
            }

            FlushTrainingLogQueue(runId);

            // Pure classification (unit-tested) drives the terminal status + which VM state to set.
            var outcome = TrainingRunClassifier.Classify(cleanExitCode, _trainingCancelRequested, runError);
            terminalStatus = outcome.Status;
            terminalExitCode = outcome.ExitCode;
            terminalNote = outcome.Note;

            if (outcome.Note is not null)
            {
                Training.SetTrainingRunError(outcome.Note);
            }
            else if (outcome.Status == TrainingRunOutcome.Cancelled)
            {
                Training.SetTrainingRunCancelled();
            }
            else
            {
                Training.CompleteTrainingRun(outcome.ExitCode ?? 0);
            }
        }
        finally
        {
            logTimer.Stop();
            checkpointTimer.Stop();
            if (ReferenceEquals(_trainingRunCts, cts))
            {
                _trainingRunCts = null;
            }

            cts.Dispose();

            // A stopped/crashed run is exactly when surviving checkpoints matter.
            await RefreshTrainingCheckpointsAsync();

            // Finalize the durable record with fresh checkpoints + terminal status.
            await FinalizeRunRecord(runProjectPath, runRecord, terminalStatus, terminalExitCode, terminalNote);
        }
    }

    /// <summary>Request cancellation of the live run (the finally block reconciles the record).</summary>
    public void StopTraining()
    {
        _trainingCancelRequested = true;
        _trainingRunCts?.Cancel();
    }

    /// <summary>Best-effort: cancel the run and synchronously kill the trainer tree so it is not
    /// orphaned when the app exits (the shell calls this from its window-closing hook).</summary>
    public void StopTrainingForShutdown()
    {
        _trainingCancelRequested = true;
        _trainingRunCts?.Cancel();
        _trainingRunner.TryKillCurrent();
    }

    private TrainingRunRecord? CreateAndSaveRunRecord(string? projectPath, IReadOnlyList<string> argv, RunProvenance? provenance = null)
    {
        if (string.IsNullOrWhiteSpace(projectPath))
        {
            return null;
        }

        var now = PythonEngineService.UtcNowIso();
        var record = new TrainingRunRecord
        {
            RunId = PythonEngineService.MintTrainingRunId(),
            CreatedAt = now,
            UpdatedAt = now,
            Status = "running",
            Target = Training.TrainingTarget,
            BaseModel = Training.TrainingBaseModel,
            ConfigPath = Training.TrainingConfigPath,
            OutputDir = Training.TrainingOutputDirectory,
            Argv = argv.ToList(),
            BeforeEvalPath = Training.TrainingBaselineReport?.ReportPath,
            Provenance = provenance,
        };
        TrySaveRunRecord(projectPath, record);
        return record;
    }

    private void RecordRunPid(string? projectPath, TrainingRunRecord? record, int pid, DateTime? startedAt)
    {
        if (string.IsNullOrWhiteSpace(projectPath) || record is null)
        {
            return;
        }

        record.Pid = pid;
        record.ProcessStartedAt = startedAt?.ToString("o", System.Globalization.CultureInfo.InvariantCulture);
        record.UpdatedAt = PythonEngineService.UtcNowIso();
        TrySaveRunRecord(projectPath, record);
    }

    private async System.Threading.Tasks.Task FinalizeRunRecord(
        string? projectPath, TrainingRunRecord? record, string status, int? exitCode, string? note)
    {
        if (string.IsNullOrWhiteSpace(projectPath) || record is null)
        {
            return;
        }

        // Enumerate checkpoints against THIS run's captured output dir/config, not the live VM.
        try
        {
            if (!string.IsNullOrWhiteSpace(record.OutputDir))
            {
                var checkpoints = await _engine.GetTrainingCheckpointsAsync(
                    record.OutputDir,
                    string.IsNullOrWhiteSpace(record.Target) ? "axolotl_yaml" : record.Target,
                    record.ConfigPath
                );
                record.Checkpoints = checkpoints.Checkpoints.ToList();
            }
        }
        catch
        {
            // Leave checkpoints as-is if enumeration fails.
        }

        record.Status = status;
        record.ExitCode = exitCode;
        record.UpdatedAt = PythonEngineService.UtcNowIso();
        if (!string.IsNullOrWhiteSpace(note))
        {
            record.Notes = note;
        }
        TrySaveRunRecord(projectPath, record);

        // Close the train→eval loop: a succeeded run produced a model, so surface the plan to evaluate it.
        if (status == "succeeded")
        {
            await ShowEvalHandoffAsync(projectPath, record.RunId);
        }
    }

    /// <summary>Surface the close-the-loop plan for a finished run. Public because the run finalizer and
    /// the run-history refresh both use it.</summary>
    public async System.Threading.Tasks.Task ShowEvalHandoffAsync(string projectPath, string runId)
    {
        try
        {
            var plan = await _engine.BuildEvalHandoffAsync(projectPath, runId);
            Training.ApplyEvalHandoff(plan);
        }
        catch (Exception ex)
        {
            Training.SetEvalHandoffError(ex.Message);
        }
    }

    private void TrySaveRunRecord(string projectPath, TrainingRunRecord record)
    {
        try
        {
            _engine.SaveTrainingRunRecord(projectPath, record);
        }
        catch
        {
            // Recording must never break or interrupt the training run.
        }
    }

    private void FlushTrainingLogQueue(int runId)
    {
        if (_trainingLogQueue.IsEmpty)
        {
            return;
        }

        var batch = new List<string>();
        while (_trainingLogQueue.TryDequeue(out var line))
        {
            batch.Add(line);
        }

        Training.AppendTrainingRunLogBatch(runId, batch);
    }

    /// <summary>Generate deterministic train/validation/test splits and persist the settings.
    /// Moved from the desktop code-behind; the ratios/seed come from the Splits tab's own fields.</summary>
    public async System.Threading.Tasks.Task GenerateSplitsAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Splits.SetSplitError("Create a dataset project before generating splits.");
            return;
        }

        try
        {
            if (!TryGetSplitOptions(
                out var trainRatio,
                out var validationRatio,
                out var seed,
                out var errorMessage
            ))
            {
                Splits.SetSplitError(errorMessage);
                return;
            }

            SetBusy("Generating splits...");
            Splits.SetSplitInProgress(trainRatio, validationRatio, seed);
            var report = await _engine.GenerateProjectSplitsAsync(
                ActiveProjectPath,
                ActiveSchemaId,
                trainRatio,
                validationRatio,
                seed
            );
            _engine.SaveProjectSplitSettings(
                ActiveProjectPath,
                new SplitSettings
                {
                    TrainRatio = trainRatio,
                    ValidationRatio = validationRatio,
                    Seed = seed,
                }
            );

            Splits.ApplySplitReport(report);
        }
        catch (System.Exception ex)
        {
            Splits.SetSplitError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Parse + validate the Splits tab's train/validation percentages and seed.
    /// Moved from the desktop code-behind.</summary>
    private bool TryGetSplitOptions(
        out double trainRatio,
        out double validationRatio,
        out int seed,
        out string errorMessage
    )
    {
        trainRatio = 0;
        validationRatio = 0;
        seed = 0;
        errorMessage = string.Empty;

        if (!double.TryParse(
            Splits.SplitTrainPercent,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out var trainPercent
        ))
        {
            errorMessage = "Train split must be a number from 1 to 98.";
            return false;
        }

        if (!double.TryParse(
            Splits.SplitValidationPercent,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out var validationPercent
        ))
        {
            errorMessage = "Validation split must be a number from 0 to 98.";
            return false;
        }

        if (!int.TryParse(
            Splits.SplitSeed,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out seed
        ))
        {
            errorMessage = "Seed must be a whole number.";
            return false;
        }

        if (!double.IsFinite(trainPercent) || !double.IsFinite(validationPercent))
        {
            errorMessage = "Split percentages must be finite numbers.";
            return false;
        }

        trainRatio = trainPercent / 100;
        validationRatio = validationPercent / 100;
        var testRatio = 1 - trainRatio - validationRatio;

        if (trainRatio <= 0 || validationRatio < 0 || testRatio <= 0)
        {
            errorMessage = "Split percentages must leave at least some room for train and test rows.";
            return false;
        }

        return true;
    }

    /// <summary>Run the quality checks and refresh the Quality tab (optionally recording a history
    /// entry for the debt-trend chart). Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RefreshQualityAsync(bool recordHistory = true)
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Quality.SetQualityError("Create or select a dataset project before running quality checks.");
            return;
        }

        try
        {
            SetBusy("Running quality checks...");
            Quality.SetQualityInProgress();
            var report = await _engine.BuildQualityReportAsync(ActiveProjectPath);
            if (recordHistory)
            {
                _engine.SaveQualityHistoryEntry(ActiveProjectPath, report);
            }

            // Load a wider window than the 5-line text summary uses so the debt-trend chart
            // has enough points; the summary still shows only its most recent few internally.
            var history = _engine.LoadQualityHistory(ActiveProjectPath, maxEntries: 30);
            Quality.ApplyQualityReport(report, history);
        }
        catch (Exception ex)
        {
            Quality.SetQualityError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Render the selected dataset-version card. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task ViewDatasetVersionCardAsync()
    {
        if (Versions.SelectedDatasetVersion is not { } selected)
        {
            Versions.SetDatasetVersionError("Select a version first.");
            return;
        }

        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return;
        }

        try
        {
            SetBusy("Rendering version card...");
            var markdown = await _engine.GetDatasetVersionCardAsync(ActiveProjectPath, selected.Record.VersionId);
            Versions.SetDatasetVersionDetail(markdown);
        }
        catch (System.Exception ex)
        {
            Versions.SetDatasetVersionError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Reload the project's dataset versions into the Versions tab. Shared by the capture
    /// command and the project-switch load (the desktop code-behind delegates here).</summary>
    public async System.Threading.Tasks.Task RefreshDatasetVersionsAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        try
        {
            Versions.ApplyDatasetVersions(await _engine.LoadDatasetVersionsAsync(ActiveProjectPath));
        }
        catch (System.Exception ex)
        {
            Versions.SetDatasetVersionError(ex.Message);
        }
    }

    /// <summary>Load everything a project needs into the tabs (splits, lab settings, examples,
    /// quarantine, AI-Assist queue/views/batches, reviewed fixes, eval filters/history, versions,
    /// gate thresholds, quality). Moved from the desktop code-behind (#249) — PUBLIC because the
    /// startup workspace-init and the project-selection change both call it. Each loader reads
    /// project-local JSON via the engine seam and applies it to the owning tab view-model.</summary>
    public async System.Threading.Tasks.Task LoadProjectAsync(DatasetProjectListItem project)
    {
        SelectProject(project);
        Splits.ApplySplitSettings(_engine.LoadProjectSplitSettings(project.ProjectPath));
        ApplyLabSettings(_engine.LoadProjectLabSettings(project.ProjectPath));
        SetExamples(_engine.LoadExamples(project.ProjectPath));
        Quarantine.SetItems(_engine.LoadImportQuarantineItems(project.ProjectPath));
        AiAssist.SetAiAssistReviewQueue(_engine.LoadAiAssistReviewQueue(project.ProjectPath));
        AiAssist.SetAiAssistQueueViews(_engine.LoadAiAssistQueueViews(project.ProjectPath));
        RewriteBatches.SetAiAssistRewriteBatches(_engine.LoadAiAssistRewriteBatches(project.ProjectPath));
        SetReviewedFixes(_engine.LoadReviewedFixes(project.ProjectPath));
        Evaluation.SetEvaluationFailureFilters(_engine.LoadEvaluationFailureFilters(project.ProjectPath));
        AiAssist.ClearBulkUndoStack();
        Evaluation.SetEvaluationReportHistory(_engine.LoadEvaluationReportHistory(project.ProjectPath));
        await RefreshDatasetVersionsAsync();
        await RefreshGateThresholdsAsync();
        await RefreshQualityAsync(recordHistory: false);
    }

    /// <summary>App-startup entry: show the setup screen when the engine isn't available, else load
    /// the workspace. Called by the shell on load and after Locate/Retry succeed. Moved from the
    /// code-behind (#249) so startup + the engine-setup screen are testable.</summary>
    public async System.Threading.Tasks.Task StartWorkspaceAsync()
    {
        if (!_engine.IsEngineAvailable)
        {
            // Don't touch the engine — show the setup screen instead of crashing.
            SetEngineUnavailable(_engine.EngineUnavailableReason);
            return;
        }

        await InitializeWorkspaceAsync();
    }

    /// <summary>Load projects + settings from the engine and open the first project. Safe to call
    /// again after the engine is located via the setup screen.</summary>
    public async System.Threading.Tasks.Task InitializeWorkspaceAsync()
    {
        try
        {
            var projects = _engine.LoadProjects();
            SetProjects(projects);
            Settings.SetSettings(_engine.GetSettings());

            var firstProject = projects.FirstOrDefault();
            if (firstProject is not null)
            {
                await LoadProjectAsync(firstProject);
            }
        }
        catch (Exception ex)
        {
            await _dialogs.ShowAsync(ex.Message, "Corpus Studio", DialogSeverity.Error);
        }
    }

    /// <summary>Locate the engine via a folder picker; on success clear the setup screen + load the
    /// workspace, else warn. Moved from the code-behind (#249).</summary>
    public async System.Threading.Tasks.Task LocateEngineAsync()
    {
        var folder = await _filePicker.PickFolderAsync(
            "Select the Corpus Studio engine folder (or the repo root that contains it)");
        if (folder is null)
        {
            return;
        }

        if (_engine.TryLocateEngine(folder))
        {
            ClearEngineUnavailable();
            await InitializeWorkspaceAsync();
        }
        else
        {
            await _dialogs.ShowAsync(
                "That folder does not contain the Corpus Studio engine "
                + "(expected corpus_studio/cli.py, or an engine/ subfolder).",
                "Engine not found", DialogSeverity.Warning);
        }
    }

    /// <summary>Retry initialising the engine on the default paths; on success load the workspace,
    /// else re-show the setup screen. Moved from the code-behind (#249).</summary>
    public async System.Threading.Tasks.Task RetryEngineAsync()
    {
        if (_engine.TryReinitialize())
        {
            ClearEngineUnavailable();
            await InitializeWorkspaceAsync();
        }
        else
        {
            SetEngineUnavailable(_engine.EngineUnavailableReason);
        }
    }

    /// <summary>Load + surface the project's gate thresholds (or an error). Moved from the code-behind
    /// (single-caller of the project-load flow).</summary>
    private async System.Threading.Tasks.Task RefreshGateThresholdsAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return;
        }

        try
        {
            var thresholds = await _engine.GetGateThresholdsAsync(ActiveProjectPath);
            Settings.ApplyGateThresholds(thresholds);
        }
        catch (System.Exception ex)
        {
            Settings.SetGateThresholdsError(ex.Message);
        }
    }

    /// <summary>Capture the current dataset as a new version. Moved from the desktop code-behind — the
    /// engine computes the fingerprint (never reimplemented in C#), and a fingerprint-less record is
    /// confirmed honestly rather than as a verified success.</summary>
    public async System.Threading.Tasks.Task CaptureDatasetVersionAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        try
        {
            SetBusy("Capturing dataset version...");
            var record = await _engine.CreateDatasetVersionAsync(ActiveProjectPath, Versions.DatasetVersionLabel, "manual");
            Versions.DatasetVersionLabel = string.Empty;
            Versions.SetDatasetVersionDetail(Tabs.VersionsViewModel.FormatCaptureConfirmation(record));
            await RefreshDatasetVersionsAsync();
        }
        catch (System.Exception ex)
        {
            Versions.SetDatasetVersionError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Diff the selected dataset version against the pinned base. Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task DiffVersionsAsync()
    {
        if (Versions.SelectedDatasetVersion is not { } selected)
        {
            Versions.SetDatasetVersionError("Select a version to diff against the base.");
            return;
        }

        if (string.IsNullOrEmpty(Versions.DatasetDiffBaseId))
        {
            Versions.SetDatasetVersionError("Set a diff base first (select a version and click 'Set diff base').");
            return;
        }

        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        try
        {
            SetBusy("Diffing versions...");
            var markdown = await _engine.GetDatasetVersionDiffAsync(
                ActiveProjectPath, Versions.DatasetDiffBaseId, selected.Record.VersionId);
            Versions.SetDatasetVersionDetail(markdown);
        }
        catch (System.Exception ex)
        {
            Versions.SetDatasetVersionError(ex.Message);
            // Replace any prior successful diff so a failure never leaves a stale result.
            Versions.SetDatasetVersionDetail("Diff failed: " + ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Render the selected model-artifact weight card. Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task ViewArtifactCardAsync()
    {
        if (Artifacts.SelectedModelArtifact is not { } selected)
        {
            Artifacts.SetArtifactError("Select an artifact first.");
            return;
        }

        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            return;
        }

        try
        {
            SetBusy("Rendering weight card...");
            var markdown = await _engine.GetWeightCardAsync(ActiveProjectPath, selected.Record.ArtifactId);
            Artifacts.SetArtifactDetail(markdown);
        }
        catch (System.Exception ex)
        {
            Artifacts.SetArtifactError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Generate the dataset card and apply it. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task GenerateDatasetCardAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            SetDatasetCardError("Create or select a dataset project before generating a dataset card.");
            return;
        }

        try
        {
            SetBusy("Generating dataset card...");
            SetDatasetCardInProgress();
            var result = await _engine.GenerateDatasetCardAsync(ActiveProjectPath, ActiveSchemaId);
            ApplyDatasetCardResult(result);
        }
        catch (System.Exception ex)
        {
            SetDatasetCardError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Validate the current Writing Studio draft against the active schema. Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task ValidateDraftAsync()
    {
        try
        {
            SetValidationInProgress();
            var report = await _engine.ValidateDraftAsync(WritingStudio.DraftText, ActiveSchemaId);
            ApplyValidationReport(report);
        }
        catch (System.Exception ex)
        {
            SetValidationError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Check training-config compatibility for the current format/target. Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task CheckTrainingCompatibilityAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveSchemaId))
        {
            Training.SetTrainingConfigError(
                "Create or select a dataset project before checking training compatibility.");
            return;
        }

        try
        {
            SetBusy("Checking training compatibility...");
            var result = await _engine.CheckTrainingCompatibilityAsync(
                ActiveSchemaId, Training.TrainingFormat, Training.TrainingTarget);
            Training.ApplyTrainingCompatibility(result);
        }
        catch (System.Exception ex)
        {
            Training.SetTrainingConfigError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    /// <summary>Refresh the project's evaluation-suite registry. Moved from the desktop code-behind.</summary>
    public async System.Threading.Tasks.Task RefreshSuitesAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            Suites.SetSuitesError("Create or select a dataset project first.");
            return;
        }

        try
        {
            Suites.IsSuitesBusy = true;
            Suites.ApplySuites(await _engine.ListSuitesAsync(ActiveProjectPath));
        }
        catch (System.Exception ex)
        {
            Suites.SetSuitesError(ex.Message);
        }
        finally
        {
            Suites.IsSuitesBusy = false;
        }
    }

    /// <summary>Export the preference project for training (DPO/KTO/reward). Moved from code-behind.</summary>
    public async System.Threading.Tasks.Task ExportPreferenceForTrainingAsync()
    {
        if (!HasActiveProject || string.IsNullOrWhiteSpace(ActiveProjectPath))
        {
            PreferenceReview.SetPreferenceRankingExportError("Create or select a preference project before exporting.");
            return;
        }

        if (ActiveSchemaId != "preference")
        {
            PreferenceReview.SetPreferenceRankingExportError("Training export is available for preference projects.");
            return;
        }

        try
        {
            SetBusy("Exporting preference data...");
            var result = await _engine.ExportPreferenceForTrainingAsync(
                ActiveProjectPath, PreferenceReview.PreferenceExportFormat);
            PreferenceReview.ApplyPreferenceTrainingExport(result);
        }
        catch (System.Exception ex)
        {
            PreferenceReview.SetPreferenceRankingExportError(ex.Message);
        }
        finally
        {
            ClearBusy();
        }
    }

    public ObservableCollection<DatasetProjectListItem> Projects { get; } = [];

    private readonly List<DatasetProjectListItem> _allProjects = [];
    private string _projectSearch = string.Empty;

    public string ProjectSearch
    {
        get => _projectSearch;
        set
        {
            if (SetField(ref _projectSearch, value))
            {
                ApplyProjectFilter();
            }
        }
    }


    public ObservableCollection<ValidationIssueNavigationItem> ValidationIssues { get; } = [];












    public ObservableCollection<ReviewedFixRecord> ReviewedFixes { get; } = [];




    public string ActiveProjectTitle
    {
        get => _activeProjectTitle;
        private set => SetField(ref _activeProjectTitle, value);
    }

    public string ActiveSchemaDescription
    {
        get => _activeSchemaDescription;
        private set => SetField(ref _activeSchemaDescription, value);
    }

    public string ActiveSchemaId
    {
        get => _activeSchemaId;
        private set
        {
            if (SetField(ref _activeSchemaId, value))
            {
                OnPropertyChanged(nameof(IsChatProject));
            }
        }
    }

    /// <summary>True when the active project is a chat dataset — the point where the chat
    /// conversation-structure gates apply.</summary>
    public bool IsChatProject => ActiveSchemaId == "chat";

    public string? ActiveProjectPath
    {
        get => _activeProjectPath;
        private set => SetField(ref _activeProjectPath, value);
    }

    public bool HasActiveProject => !string.IsNullOrWhiteSpace(ActiveProjectPath);

    public DatasetProjectListItem? SelectedProject
    {
        get => _selectedProject;
        set => SetField(ref _selectedProject, value);
    }

    // The saved-examples list/selection/JSON pane moved to ExamplesViewModel in Phase 2 (the dataset
    // spine). The shell's SetExamples orchestrates the dataset-changed fan-out; cross-tab row
    // navigation (quality triage/synthetic/reviewed-fix) reads Examples.Items + sets
    // Examples.SelectedExample. Bindings use Examples.*, code-behind ViewModel.Examples.*.

    public ValidationIssueNavigationItem? SelectedValidationIssue
    {
        get => _selectedValidationIssue;
        set => SetField(ref _selectedValidationIssue, value);
    }


    // The draft editor buffer + dirty-tracking + LoadDraft/MarkDraftClean moved to
    // WritingStudioViewModel in Phase 2 (backend-cluster slice 1). LoadDraft is the shared
    // "load text into the editor" seam; call sites use WritingStudio.LoadDraft(...). The shell keeps
    // the aggregate HasUnsavedWork (draft OR a dirty Explorer document) and the draft-construction
    // helpers (BuildDraftTemplate / BuildJsonArrayDraft).

    /// <summary>Unsaved work that a project switch or app close would silently discard: an
    /// edited draft, or a dirty open document in the workspace explorer. Read imperatively by the
    /// switch/close guards.</summary>
    public bool HasUnsavedWork => WritingStudio.IsDraftDirty || Explorer.HasDirtyDocuments;

    public string ValidationSummary
    {
        get => _validationSummary;
        private set => SetField(ref _validationSummary, value);
    }


    // ---- Quality metric grid (v1.2.12) -------------------------------------------
    // A structured view of the core quality counts for the right panel, replacing the text
    // blob. Quality.QualitySummary (the full text) is still built for the dashboard card. The status
    // banner is PII-aware (unlike the legacy `health` line, which ignores PII).

    /// <summary>The core quality counts as scannable rows (Examples + the six issue metrics).</summary>





    /// <summary>The optional flagged-row detail (PII findings, token outliers, category
    /// imbalance, synthetic clusters/samples) — empty when the dataset is clean.</summary>



    public string GateSummary
    {
        get => _gateSummary;
        private set => SetField(ref _gateSummary, value);
    }

    // ---- Workspace Problems panel (v1.2.6) ---------------------------------------
    // A structured, workspace-level view of the latest gate findings (block/warn only),
    // populated by ApplyGateReport. The panel surfaces problems; it never approves or
    // auto-fixes anything — the human still reviews and decides.

    /// <summary>The latest gate findings that are actual problems (block/warn), block-first.
    /// Passes are not problems and are reported only in the summary line.</summary>
    public ObservableCollection<ProblemItem> Problems { get; } = [];

    public bool ProblemsPanelVisible
    {
        get => _problemsPanelVisible;
        private set
        {
            if (SetField(ref _problemsPanelVisible, value))
            {
                OnPropertyChanged(nameof(IsNoProblems));
                // Problems, Output, and Search share the bottom dock — only one is ever open.
                if (value)
                {
                    OutputPanelVisible = false;
                    SearchPanelVisible = false;
                }
            }
        }
    }

    /// <summary>True when the panel is open but has no problems to show (empty state).</summary>
    public bool IsNoProblems => Problems.Count == 0;

    public string ProblemsSummary
    {
        get => _problemsSummary;
        private set => SetField(ref _problemsSummary, value);
    }

    /// <summary>Activity-bar badge text — the count of problems (block+warn), or empty when
    /// there are none / gates have not run.</summary>
    public string ProblemsBadge
    {
        get => _problemsBadge;
        private set
        {
            if (SetField(ref _problemsBadge, value))
            {
                OnPropertyChanged(nameof(HasProblemsBadge));
            }
        }
    }

    public bool HasProblemsBadge => !string.IsNullOrEmpty(_problemsBadge);

    /// <summary>Badge colour: red when any block, amber when only warns (see StatusColor).</summary>
    public string ProblemsBadgeColor
    {
        get => _problemsBadgeColor;
        private set => SetField(ref _problemsBadgeColor, value);
    }

    public void ToggleProblemsPanel() => ProblemsPanelVisible = !ProblemsPanelVisible;

    public void ShowProblemsPanel() => ProblemsPanelVisible = true;

    /// <summary>Rebuild the Problems panel + activity-bar badge from a gate report. Blocks and
    /// warns become rows (block-first); passes are counted in the summary only. Kept in sync
    /// with GateSummary so the panel and the Studio gates tab never diverge.</summary>
    private void ApplyProblemsFromGateReport(GateReport report)
    {
        Problems.Clear();
        var rows = report.Results
            .Where(ProblemItem.IsProblem)
            .Select(ProblemItem.FromGateResult)
            .OrderBy(p => p.SeverityRank);
        foreach (var row in rows)
        {
            Problems.Add(row);
        }

        var problemCount = report.BlockCount + report.WarnCount;
        ProblemsBadge = problemCount > 0 ? problemCount.ToString(CultureInfo.InvariantCulture) : string.Empty;
        ProblemsBadgeColor = GateReport.StatusColor(report.BlockCount > 0 ? "block" : "warn");

        ProblemsSummary = problemCount == 0
            ? $"No problems — all {report.PassCount} checks passed. A clean gate is not approval; you still review before export."
            : $"{problemCount} problem{(problemCount == 1 ? string.Empty : "s")} "
              + $"({report.BlockCount} block, {report.WarnCount} warn) · {report.PassCount} passed.";

        OnPropertyChanged(nameof(IsNoProblems));
    }

    /// <summary>Reset the Problems panel to its empty state (e.g. on project switch), so stale
    /// findings from a previous project are never shown against the new one.</summary>
    public void ResetProblems()
    {
        Problems.Clear();
        ProblemsBadge = string.Empty;
        ProblemsBadgeColor = GateReport.StatusColor(null);
        ProblemsSummary = "Run gates to check this dataset for problems.";
        OnPropertyChanged(nameof(IsNoProblems));
    }

    // ---- Workspace Output / Logs panel (v1.2.7) ----------------------------------
    // An "Output channel" recording engine CLI activity (verb, outcome, duration, stderr on
    // failure). Ephemeral in-memory ring buffer; shares the bottom dock with Problems.

    /// <summary>Cap on retained log lines — a diagnostic tail, not a full history.</summary>
    public const int MaxOutputLogEntries = 200;

    /// <summary>Engine command log, newest last. Bounded to <see cref="MaxOutputLogEntries"/>.</summary>
    public ObservableCollection<EngineLogEntry> OutputLog { get; } = [];

    public bool OutputPanelVisible
    {
        get => _outputPanelVisible;
        private set
        {
            if (SetField(ref _outputPanelVisible, value))
            {
                OnPropertyChanged(nameof(IsNoOutput));
                if (value)
                {
                    ProblemsPanelVisible = false;
                    SearchPanelVisible = false;
                }
            }
        }
    }

    public bool IsNoOutput => OutputLog.Count == 0;

    public bool HasOutput => OutputLog.Count > 0;

    public string OutputSummary => OutputLog.Count == 0
        ? "Engine activity will appear here as you run commands."
        : $"{OutputLog.Count} engine command{(OutputLog.Count == 1 ? string.Empty : "s")} this session"
          + (OutputLog.Count >= MaxOutputLogEntries ? " (oldest trimmed)" : string.Empty);

    public void ToggleOutputPanel() => OutputPanelVisible = !OutputPanelVisible;

    public void ShowOutputPanel() => OutputPanelVisible = true;

    /// <summary>Workspace content-search panel — the third mutually-exclusive bottom-dock panel.</summary>
    public bool SearchPanelVisible
    {
        get => _searchPanelVisible;
        private set
        {
            if (SetField(ref _searchPanelVisible, value) && value)
            {
                ProblemsPanelVisible = false;
                OutputPanelVisible = false;
            }
        }
    }

    /// <summary>Open (or toggle) the Search panel, pointing it at the active project's folder
    /// so results reflect the current workspace.</summary>
    public void ToggleSearchPanel()
    {
        Search.SetWorkspaceRoot(HasActiveProject ? ActiveProjectPath : null);
        SearchPanelVisible = !SearchPanelVisible;
    }

    public void ShowSearchPanel()
    {
        Search.SetWorkspaceRoot(HasActiveProject ? ActiveProjectPath : null);
        SearchPanelVisible = true;
    }

    /// <summary>Append one engine-invocation log line, trimming the oldest past the cap. Must be
    /// called on the UI thread (the service raises its event on a background thread; the view
    /// marshals it).</summary>
    public void AppendEngineLog(EngineLogEntry entry)
    {
        if (entry is null)
        {
            return;
        }

        OutputLog.Add(entry);
        while (OutputLog.Count > MaxOutputLogEntries)
        {
            OutputLog.RemoveAt(0);
        }

        OnPropertyChanged(nameof(IsNoOutput));
        OnPropertyChanged(nameof(HasOutput));
        OnPropertyChanged(nameof(OutputSummary));
    }

    public void ClearOutputLog()
    {
        OutputLog.Clear();
        OnPropertyChanged(nameof(IsNoOutput));
        OnPropertyChanged(nameof(HasOutput));
        OnPropertyChanged(nameof(OutputSummary));
    }

    // Arena state + logic now live in the child Arena view-model (backlog #4). Bindings use
    // Arena.*, the code-behind uses ViewModel.Arena.* and ArenaViewModel.ParseModelList.
    // Provider generation policy + general app settings now live in the child Settings
    // view-model (Phase 2). Bindings use Settings.*, the code-behind uses ViewModel.Settings.*.

    public void SetGateInProgress()
    {
        GateSummary = "Running gates...";
    }

    public void SetGateError(string message)
    {
        GateSummary = $"Gates could not run.{Environment.NewLine}{message}";
    }

    /// <summary>Format a gate report into a readable pass/warn/block summary.</summary>
    public void ApplyGateReport(GateReport report)
    {
        var icon = report.OverallStatus switch
        {
            "block" => "⛔", // no-entry
            "warn" => "⚠",  // warning
            _ => "✅",       // check
        };
        var lines = new List<string>
        {
            $"{icon} {report.Scope} gates: {report.OverallStatus.ToUpperInvariant()} "
            + $"({report.PassCount} pass, {report.WarnCount} warn, {report.BlockCount} block)",
            string.Empty,
        };

        foreach (var result in report.Results)
        {
            var mark = result.Status switch
            {
                "block" => "[BLOCK]",
                "warn" => "[WARN]",
                _ => "[PASS]",
            };
            lines.Add($"{mark} {result.Name}: {result.Message}");
            if (result.Status != "pass" && !string.IsNullOrWhiteSpace(result.Repair))
            {
                lines.Add($"    fix: {result.Repair}");
            }
        }

        GateSummary = string.Join(Environment.NewLine, lines);
        ApplyProblemsFromGateReport(report);
    }


    // ---- Debt trend (v1.2.9) -----------------------------------------------------
    // A mini-chart of the quality issue rate across recorded quality runs, built from the
    // existing quality history (nothing new persisted). Not the A-F grade: presence-based
    // PII/secrets aren't in the history, so only the issue-rate trend is honest to plot.

    /// <summary>Ordered oldest → newest bars for the debt-trend mini-chart.</summary>

    /// <summary>True when there is at least one bar to draw (chart visibility).</summary>

    /// <summary>True when there are ≥2 runs — enough to state a direction.</summary>







    // Preference Review tab state + display/export logic extracted to PreferenceReviewViewModel in
    // Phase 2. The shell pushes ActiveSchemaId down, forwards SetItems(Examples) + Reset() on
    // project/example changes, and keeps the AI-Assist handoff actions (PreparePreference*JudgeReview)
    // which reach into the child. Bindings use PreferenceReview.*, code-behind ViewModel.PreferenceReview.*.


    // Splits tab state + logic extracted to SplitsViewModel in Phase 2. The shell wires the child's
    // ErrorReported to ReportError, pushes loaded settings, and forwards Reset() on project switch.
    // Bindings use Splits.*, code-behind ViewModel.Splits.*.





    public string BenchmarkModelsInput
    {
        get => _benchmarkModelsInput;
        set => SetField(ref _benchmarkModelsInput, value);
    }

    /// <summary>Export cleaning options (bound to the two Export checkboxes): drop exact-duplicate rows
    /// and drop low-information rows during a JSONL export.</summary>
    public bool ExportRemoveDuplicates
    {
        get => _exportRemoveDuplicates;
        set => SetField(ref _exportRemoveDuplicates, value);
    }

    public bool ExportRemoveLowInformation
    {
        get => _exportRemoveLowInformation;
        set => SetField(ref _exportRemoveLowInformation, value);
    }

    /// <summary>Opt-in: mask detected PII/secrets in the export with typed [REDACTED:kind] placeholders.
    /// Known high-precision patterns only — NOT a guarantee of de-identification (the engine records a
    /// manifest and appends an honest warning to the export result). See issue #222 / #194.</summary>
    public bool ExportRedactPii
    {
        get => _exportRedactPii;
        set => SetField(ref _exportRedactPii, value);
    }

    /// <summary>Export file format (bound to the Export format selector): "jsonl" (default, model-ready,
    /// all schemas), "csv"/"tsv" (flat schemas only — the engine refuses a chat/nested schema), or
    /// "parquet" (columnar, all schemas; needs the engine's optional [parquet] extra).</summary>
    public string ExportFormat
    {
        get => _exportFormat;
        set => SetField(ref _exportFormat, value);
    }

    public string BenchmarkSummary
    {
        get => _benchmarkSummary;
        private set => SetField(ref _benchmarkSummary, value);
    }

    /// <summary>Parse the benchmark models input (one per line or comma-separated,
    /// trimmed, de-duplicated, order preserved).</summary>
    public IReadOnlyList<string> GetBenchmarkModels()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var models = new List<string>();
        foreach (var token in _benchmarkModelsInput.Split(
            new[] { '\n', '\r', ',' },
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (seen.Add(token))
            {
                models.Add(token);
            }
        }

        return models;
    }

    public void ApplyBenchmarkReport(BenchmarkReport report)
    {
        var lines = new List<string>
        {
            $"Benchmarked {report.ModelCount} model(s) on {report.ExamplesTested} example(s)",
            $"Best: {report.BestModel} · Worst: {report.WorstModel} · Score spread: {report.ScoreSpread:0.##}",
            "",
            "Ranking:",
        };
        lines.AddRange(report.Models.Select(model => $"- {model.DisplayName}"));

        if (report.CommonlyFailedExamples.Count > 0)
        {
            lines.Add("");
            lines.Add($"Failed by every model ({report.CommonlyFailedExamples.Count}): "
                + string.Join(", ", report.CommonlyFailedExamples.Take(10)));
        }

        BenchmarkSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetBenchmarkError(string message)
    {
        BenchmarkSummary = $"Benchmark could not run.{Environment.NewLine}{message}";
    }
















    public ReviewedFixRecord? SelectedReviewedFix
    {
        get => _selectedReviewedFix;
        set
        {
            if (SetField(ref _selectedReviewedFix, value) && value is not null)
            {
                ReviewedFixSummary =
                    $"Selected reviewed fix for {value.ExampleId} (v{value.Version}): {value.StatusLabel}.";
            }
        }
    }

    public string ReviewedFixSummary
    {
        get => _reviewedFixSummary;
        private set => SetField(ref _reviewedFixSummary, value);
    }













    /// <summary>The exact launch command from the last training-config export
    /// (empty if none). Backs the "Copy launch command" action.</summary>




    /// <summary>Whether a run can be launched (a config was generated and none is running).</summary>

    /// <summary>The structured command to spawn (empty until a config is generated).</summary>



    /// <summary>Start a run and return its id; log appends are tagged with this
    /// id so a prior (cancelled) run's late output cannot contaminate this one.</summary>


    /// <summary>Append a batch of streamed lines for a specific run. Lines tagged
    /// with a stale run id (from a cancelled run) are dropped.</summary>







    // --- Model artifacts (v0.9; extracted to ArtifactsViewModel in Phase 2) ------
    // The Artifacts tab's state + logic now live in the child Artifacts view-model; the shell only
    // forwards Reset() on project switch. Bindings use Artifacts.*, code-behind ViewModel.Artifacts.*.

    // --- Evaluation suites (v1.3 M2; extracted to SuitesViewModel in Phase 2) ------
    // The Suites tab's state + logic now live in the child Suites view-model; the shell forwards
    // Reset() on project switch and pushes HasActiveProject down (CanRunSuite depends on it).
    // Bindings use Suites.*, code-behind ViewModel.Suites.*.

    // --- Dataset version history (v1.0; extracted to VersionsViewModel in Phase 2) --------
    // The Versions tab's state + logic now live in the child Versions view-model; the shell only
    // forwards Reset() on project switch. Bindings use Versions.*, code-behind ViewModel.Versions.*.

    // --- Dataset debt (v1.1; extracted to DebtViewModel in the #4 decomposition) -----
    // The Debt tab's state + logic now live in the child Debt view-model; the shell only
    // forwards the cross-cutting lifecycle (Reset on project switch, InvalidateDebt on dataset
    // change). Bindings use Debt.* (dashboard badge + Debt tab).


    /// <summary>Format a training-run regression gate verdict (pass/warn/block).</summary>


    /// <summary>Format the durable run registry (newest first). Reconciliation of
    /// stuck `running` records happens in the service before this is called.</summary>

    /// <summary>Where the trainer writes checkpoints (from the last config export).</summary>

    /// <summary>The rendered config path from the last config export.</summary>


    /// <summary>The exact resume command for the latest checkpoint (empty if none).</summary>


    /// <summary>Resume is available when the target supports a resume flag, a
    /// checkpoint exists, and no run is active.</summary>


    /// <summary>The "before" evaluation report captured at training launch (null if
    /// no evaluation had been saved yet).</summary>

    /// <summary>Capture the pre-training baseline (the newest saved evaluation
    /// report at launch time, or null when none exists).</summary>

    /// <summary>Compare the newest post-training evaluation report against the
    /// captured baseline. <paramref name="history"/> is newest-first.</summary>



    public string DatasetCardSummary
    {
        get => _datasetCardSummary;
        private set => SetField(ref _datasetCardSummary, value);
    }

    public string DatasetCardPreview
    {
        get => _datasetCardPreview;
        private set => SetField(ref _datasetCardPreview, value);
    }




    public string LabSettingsSummary
    {
        get => _labSettingsSummary;
        private set => SetField(ref _labSettingsSummary, value);
    }

    public void AddProject(string projectId, string name, string schemaName)
    {
        AddProject(projectId, name, "instruction", schemaName, null);
    }

    public void SetProjects(IEnumerable<DatasetProjectListItem> projects)
    {
        _allProjects.Clear();
        _allProjects.AddRange(projects);
        ApplyProjectFilter();
    }

    private void ApplyProjectFilter()
    {
        var search = _projectSearch?.Trim() ?? string.Empty;
        Projects.Clear();
        foreach (var project in _allProjects)
        {
            if (ProjectMatchesSearch(project, search))
            {
                Projects.Add(project);
            }
        }
    }

    private static bool ProjectMatchesSearch(DatasetProjectListItem project, string search)
    {
        if (string.IsNullOrEmpty(search))
        {
            return true;
        }

        return ContainsSearch(project.Name, search)
            || ContainsSearch(project.Id, search)
            || ContainsSearch(project.SchemaId, search);
    }

    public string ProjectIndexSummary
    {
        get => _projectIndexSummary;
        private set => SetField(ref _projectIndexSummary, value);
    }

    public void ApplyProjectIndexRebuilt(ProjectIndexRebuildResult result)
    {
        ProjectIndexSummary = result.Indexed == 0
            ? "No projects found to index."
            : $"Indexed {result.Indexed} project(s); listing from the SQLite index.";
    }

    public void SetProjectIndexError(string message)
    {
        ProjectIndexSummary = $"Project index update failed.{Environment.NewLine}{message}";
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set => SetField(ref _isBusy, value);
    }

    public string BusyStatus
    {
        get => _busyStatus;
        private set => SetField(ref _busyStatus, value);
    }

    /// <summary>Show the blocking busy overlay with a status message.</summary>
    public void SetBusy(string status)
    {
        HasError = false;
        BusyStatus = string.IsNullOrWhiteSpace(status) ? "Working..." : status;
        IsBusy = true;
    }

    public void ClearBusy()
    {
        IsBusy = false;
    }

    public bool HasError
    {
        get => _hasError;
        private set => SetField(ref _hasError, value);
    }

    public string ErrorMessage
    {
        get => _errorMessage;
        private set => SetField(ref _errorMessage, value);
    }

    /// <summary>Surface an operation error in the shared, dismissible error banner.</summary>
    public void ReportError(string message)
    {
        ErrorMessage = string.IsNullOrWhiteSpace(message)
            ? "An unexpected error occurred."
            : message.Trim();
        HasError = true;
    }

    public void DismissError()
    {
        HasError = false;
    }

    public void AddProject(
        string projectId,
        string name,
        string schemaId,
        string schemaName,
        string? projectPath
    )
    {
        var project = new DatasetProject(
            projectId,
            name,
            schemaId,
            DateTime.Now,
            DateTime.Now,
            SplitSettings.Default,
            LabBackendSettings.Default
        );
        var projectItem = new DatasetProjectListItem(project, projectPath ?? string.Empty);

        Projects.Add(projectItem);
        SelectProject(projectItem, schemaName);
        WritingStudio.LoadDraft(BuildDraftTemplate(schemaId));
    }

    public void SelectProject(DatasetProjectListItem project, string? schemaName = null)
    {
        SelectedProject = project;
        ActiveProjectTitle = project.Name;
        ActiveProjectPath = project.ProjectPath;
        ActiveSchemaId = project.SchemaId;
        // The Preference Review tab is gated on the schema; push it down so the child builds/shows
        // pairs only for a "preference" project.
        PreferenceReview.ActiveSchemaId = ActiveSchemaId;
        ActiveSchemaDescription = $"{schemaName ?? project.SchemaId} project. Ready for examples.";
        AiAssist.ApplyAiAssistActionPresets(project.SchemaId);
        Splits.ApplySplitSettings(project.Project.SplitSettings ?? SplitSettings.Default);
        ApplyLabSettings(project.Project.LabSettings ?? LabBackendSettings.Default);
        ValidationSummary = "No validation has run yet.";
        ClearValidationIssues();
        // Quality/debt-trend/synthetic panel state is per-project: reset the child.
        Quality.Reset();
        Splits.Reset();
        // Evaluation run/report/result state is per-project: reset the child.
        Evaluation.Reset();
        // AI-Assist run/queue/gate state is per-project: reset the child (clears panes, the queue +
        // views, the suggestion buffer, and the candidate-gate verdict).
        AiAssist.Reset();
        // AI-Assist rewrite batches are per-project: reset the child so nothing leaks across a switch.
        RewriteBatches.Reset();
        ReviewedFixes.Clear();
        SelectedReviewedFix = null;
        _lastPreparedEvaluationFix = null;
        ReviewedFixSummary =
            "Edited failed rows appear here so you can track which fixes were re-tested.";
        // Clear the Problems panel so a previous project's gate findings are never
        // shown against the newly selected (not-yet-gated) project.
        ResetProblems();
        // Training config inputs/preview are per-project (the format follows the schema).
        Training.Reset(project.SchemaId);
        // Saved-examples state is per-project: reset the child (clears the list + selection).
        Examples.Reset();
        // Import-quarantine state is per-project: reset the child; the retry-tracking (_pendingRetryItem)
        // is a shell-owned Writing-Studio bridge, cleared here too.
        Quarantine.Reset();
        _pendingRetryItem = null;
        // Preference-review state is per-project: reset the child so nothing leaks across a switch.
        PreferenceReview.Reset();
        // Dataset-version state is per-project: reset the child view-model so nothing leaks across
        // a project switch. LoadProjectAsync eagerly refreshes the new project's versions.
        Versions.Reset();
        // Model artifacts are per-project too. The project-load path does NOT eagerly refresh the
        // artifact list (the tab is refreshed on demand), so without this reset the previous
        // project's artifacts, selection, and panes would linger — and a Keep/Reject would act on a
        // stale artifact id against the newly selected project. Give the tab the same per-project
        // guard its siblings (Debt/Versions) already have.
        Artifacts.Reset();
        // Debt is per-project and per-dataset — reset to the neutral default so a fresh
        // project reads "run a debt check", never a leaked grade or a "dataset changed" note.
        Debt.Reset();
        // Clear the Explorer so a project switch can't show the previous project's tree or
        // open document tabs; ShowFiles rebuilds it lazily for the new project.
        Explorer.Reset();
        // Drop stale search results/root so they can't leak across a project switch.
        Search.SetWorkspaceRoot(null);
        // Drop stale evaluation suites + report so they can't leak across a project switch, then
        // push the new project-open state so the child's CanRunSuite (the Run gate) is correct.
        Suites.Reset();
        Suites.HasActiveProject = HasActiveProject;
        // Reset the Writing Studio draft to the new project's schema template. Without this, an
        // unsaved draft typed against the previous project would be written into THIS project's
        // examples.jsonl on the next save (a cross-project data leak).
        WritingStudio.LoadDraft(BuildDraftTemplate(project.SchemaId));
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(HasActiveProject)));
    }

    public void ApplyLabSettings(LabBackendSettings settings)
    {
        ApplyBackendSettings(settings.Evaluation, isEvaluation: true);
        ApplyBackendSettings(settings.AiAssist, isEvaluation: false);
        LabSettingsSummary = "Lab backend settings loaded for this project.";
    }

    public LabBackendSettings BuildCurrentLabSettings()
    {
        return new LabBackendSettings
        {
            Evaluation = new ModelBackendSettings
            {
                Backend = EvaluationConnection.EvaluationBackend.Trim(),
                Model = EvaluationConnection.EvaluationModel.Trim(),
                BaseUrl = EvaluationConnection.EvaluationBaseUrl.Trim(),
                TimeoutSeconds = ParsePositiveIntOrDefault(EvaluationConnection.EvaluationTimeoutSeconds, 120),
            },
            AiAssist = new ModelBackendSettings
            {
                Backend = AiAssistConnection.AiAssistBackend.Trim(),
                Model = AiAssistConnection.AiAssistModel.Trim(),
                BaseUrl = AiAssistConnection.AiAssistBaseUrl.Trim(),
                TimeoutSeconds = ParsePositiveIntOrDefault(AiAssistConnection.AiAssistTimeoutSeconds, 120),
            },
        };
    }

    public void ApplyLabSettingsSaved(string projectPath)
    {
        LabSettingsSummary =
            $"Saved lab backend settings to project metadata: {projectPath}";
    }

    public void SetLabSettingsError(string message)
    {
        LabSettingsSummary = $"Lab backend settings could not be saved.{Environment.NewLine}{message}";
    }

    /// <summary>Dataset-changed orchestrator: rebuild the saved-examples list (via the Examples tab),
    /// then fan the change out to the tabs that derive from the dataset — Preference pairs, the Quality
    /// summary, and the (now stale) Debt grade.</summary>
    public void SetExamples(IEnumerable<SavedExampleItem> examples)
    {
        Examples.SetItems(examples);
        PreferenceReview.SetItems(Examples.Items);
        Quality.QualitySummary = Examples.Items.Count == 0
            ? "No saved examples yet. Quality checks will run after examples are added."
            : $"{Examples.Items.Count} saved example(s). Run quality checks to inspect duplicates and empty rows.";
        // The dataset changed: any prior debt grade is now stale. Invalidate it so the
        // Debt tab can never show a verdict that no longer matches the data.
        Debt.InvalidateDebt();
    }



    /// <summary>Build the structured Quality metric rows (Examples + the six issue counts) and
    /// the status banner. The banner is PII-aware: any PII/secret finding is a red problem, even
    /// though the legacy `health` line — kept for Quality.QualitySummary — does not weigh PII.</summary>



    public bool PrepareSyntheticIssueRewrite()
    {
        if (Quality.SelectedSyntheticPatternIssue is null)
        {
            Quality.QualityTriageSummary = "Select a synthetic quality issue before preparing a rewrite.";
            return false;
        }

        var rowNumber = Quality.SelectedSyntheticPatternIssue.RowNumbers.FirstOrDefault(row => row > 0);
        if (rowNumber <= 0)
        {
            Quality.QualityTriageSummary = "Selected synthetic quality issue does not include an affected row number.";
            return false;
        }

        var example = Examples.Items.FirstOrDefault(item => item.RowNumber == rowNumber);
        if (example is null)
        {
            Quality.QualityTriageSummary = $"Affected row {rowNumber} is not loaded in the Examples list.";
            return false;
        }

        Examples.SelectedExample = example;
        WritingStudio.LoadDraft(example.Json);
        if (AiAssist.AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssist.AiAssistAction = "rewrite-output";
        }

        AiAssist.AiAssistInstruction = BuildSyntheticRewriteInstruction(Quality.SelectedSyntheticPatternIssue, rowNumber);
        Quality.QualityTriageSummary =
            $"Prepared row {rowNumber} for AI Assist rewrite. Review, run AI Assist, validate, and save only after editing.";
        return true;
    }

    public bool PrepareSyntheticBatchRewrite()
    {
        if (Quality.SyntheticPatternIssues.Count == 0)
        {
            Quality.QualityTriageSummary = "Run quality checks with synthetic warnings before preparing a batch rewrite.";
            return false;
        }

        var rowNumbers = Quality.SyntheticPatternIssues
            .SelectMany(issue => issue.RowNumbers)
            .Where(rowNumber => rowNumber > 0)
            .Distinct()
            .Order()
            .Take(12)
            .ToList();
        if (rowNumbers.Count == 0)
        {
            Quality.QualityTriageSummary = "Synthetic quality issues do not include affected row numbers.";
            return false;
        }

        var affectedRows = rowNumbers
            .Select(rowNumber => Examples.Items.FirstOrDefault(example => example.RowNumber == rowNumber))
            .Where(example => example is not null)
            .Cast<SavedExampleItem>()
            .ToList();
        if (affectedRows.Count == 0)
        {
            Quality.QualityTriageSummary = "Affected synthetic rows are not loaded in the Examples list.";
            return false;
        }

        WritingStudio.LoadDraft(BuildJsonArrayDraft(affectedRows.Select(row => row.Json)));
        if (AiAssist.AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssist.AiAssistAction = "rewrite-output";
        }

        AiAssist.AiAssistInstruction = BuildSyntheticBatchRewriteInstruction(Quality.SyntheticPatternIssues, rowNumbers);
        RewriteBatches.SetLastPrepared(new AiAssistRewriteBatch
        {
            SchemaId = ActiveSchemaId,
            Action = "rewrite-output",
            RowNumbers = rowNumbers,
            IssueCount = Quality.SyntheticPatternIssues.Count,
            IssueSummary = BuildSyntheticIssueSummary(Quality.SyntheticPatternIssues),
            SourceDraft = WritingStudio.DraftText,
            Instruction = AiAssist.AiAssistInstruction,
        });
        Quality.QualityTriageSummary =
            $"Prepared {affectedRows.Count} affected row(s) from {Quality.SyntheticPatternIssues.Count} synthetic issue(s) for batch rewrite.";
        return true;
    }

    public bool PrepareEvaluationFailureReview()
    {
        if (Evaluation.SelectedEvaluationExampleResult is null)
        {
            Evaluation.EvaluationReviewSummary = "Select an evaluation example before preparing failure triage.";
            return false;
        }

        if (Evaluation.SelectedEvaluationExampleResult.Passed)
        {
            Evaluation.EvaluationReviewSummary = "Selected evaluation example passed. Choose a failed example for failure triage.";
            return false;
        }

        if (ActiveSchemaId is not ("instruction" or "chat"))
        {
            Evaluation.EvaluationReviewSummary = "Evaluation failure triage supports instruction and chat projects.";
            return false;
        }

        WritingStudio.LoadDraft(BuildEvaluationFailureDraft(Evaluation.SelectedEvaluationExampleResult, ActiveSchemaId));
        if (AiAssist.AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssist.AiAssistAction = "rewrite-output";
        }

        AiAssist.AiAssistInstruction = BuildEvaluationFailureInstruction(Evaluation.SelectedEvaluationExampleResult);
        Evaluation.EvaluationReviewSummary =
            $"Prepared failed evaluation example {Evaluation.SelectedEvaluationExampleResult.ExampleId} for AI Assist triage.";
        GoToStudioTab(StudioTab.AiAssist);
        return true;
    }

    public bool PrepareEvaluationFailureEdit()
    {
        if (Evaluation.SelectedEvaluationExampleResult is null)
        {
            Evaluation.EvaluationReviewSummary = "Select an evaluation example before editing a failed row.";
            return false;
        }

        if (Evaluation.SelectedEvaluationExampleResult.Passed)
        {
            Evaluation.EvaluationReviewSummary = "Selected evaluation example passed. Choose a failed example to edit.";
            return false;
        }

        if (!TryParseEvaluationRowNumber(Evaluation.SelectedEvaluationExampleResult.ExampleId, out var rowNumber))
        {
            Evaluation.EvaluationReviewSummary =
                $"Evaluation example '{Evaluation.SelectedEvaluationExampleResult.ExampleId}' is not linked to a saved row.";
            return false;
        }

        var example = Examples.Items.FirstOrDefault(item => item.RowNumber == rowNumber);
        if (example is null)
        {
            Evaluation.EvaluationReviewSummary =
                $"Saved row {rowNumber} is not loaded in the Examples list.";
            return false;
        }

        var failure = Evaluation.SelectedEvaluationExampleResult;
        _lastPreparedEvaluationFix = new ReviewedFixRecord
        {
            ExampleId = failure.ExampleId,
            RowNumber = rowNumber,
            SchemaId = ActiveSchemaId,
            OriginalScore = failure.Score,
            FailureReason = Evaluation.ClassifyFailureReason(failure),
            SourceReport = Evaluation.SelectedEvaluationReportHistoryItem?.DisplayName ?? "current evaluation run",
        };

        Examples.SelectedExample = example;
        WritingStudio.LoadDraft(example.Json);
        ValidationSummary =
            $"Loaded evaluation failure row {rowNumber}. Validate before saving reviewed edits.";
        Evaluation.EvaluationReviewSummary =
            $"Loaded failed row {rowNumber} into Writing Studio. Edit, validate, save, then rerun evaluation.";
        return true;
    }

    // Preference → AI Assist handoff. These stay on the shell (they write the not-yet-extracted AI
    // Assist tab's state + LoadDraft); they reach into the extracted PreferenceReview tab for the
    // selection / visible pairs and to set the review summary. Resolve this bridge when AI Assist
    // decomposes (Phase 2 do-last cluster).
    public bool PreparePreferenceJudgeReview()
    {
        var selected = PreferenceReview.SelectedPreferenceReviewItem;
        if (selected is null)
        {
            PreferenceReview.SetReviewSummary("Select a saved preference pair before preparing AI Assist review.");
            return false;
        }

        WritingStudio.LoadDraft(selected.Json);
        if (AiAssist.AiAssistActionPresets.Contains("judge-preference-strength"))
        {
            AiAssist.AiAssistAction = "judge-preference-strength";
        }

        AiAssist.AiAssistInstruction = BuildPreferenceJudgeInstruction(
            selected.RowNumber,
            selected.Prompt,
            selected.Chosen,
            selected.Rejected
        );
        PreferenceReview.SetReviewSummary(
            $"Prepared Example {selected.RowNumber} for AI Assist preference-strength review.");
        GoToStudioTab(StudioTab.AiAssist);
        return true;
    }

    public bool PreparePreferenceBatchJudgeReview()
    {
        if (ActiveSchemaId != "preference")
        {
            PreferenceReview.SetReviewSummary("Preference batch review is available for preference projects.");
            return false;
        }

        var items = PreferenceReview.GetVisiblePreferenceReviewItems();
        if (items.Count == 0)
        {
            PreferenceReview.SetReviewSummary("No preference pairs match the current ranking filter.");
            return false;
        }

        WritingStudio.LoadDraft(BuildJsonArrayDraft(items.Select(item => item.Json)));
        if (AiAssist.AiAssistActionPresets.Contains("judge-preference-strength"))
        {
            AiAssist.AiAssistAction = "judge-preference-strength";
        }

        AiAssist.AiAssistInstruction = BuildPreferenceBatchJudgeInstruction(items);
        PreferenceReview.SetReviewSummary(
            $"Prepared {items.Count} visible preference pair(s) for AI Assist batch judging.");
        GoToStudioTab(StudioTab.AiAssist);
        return true;
    }














    public void SetEvaluationHealthCheckInProgress()
    {
        Evaluation.EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                "Checking evaluation backend...",
                $"Backend: {EvaluationConnection.EvaluationBackend}",
                $"Model: {EvaluationConnection.EvaluationModel}",
            ]
        );
    }

    public void ApplyEvaluationBackendHealthReport(BackendHealthReport report)
    {
        SetAvailableModels(EvaluationConnection.EvaluationAvailableModels, report.AvailableModels);
        Evaluation.EvaluationSummary = FormatBackendHealthReport("Evaluation backend", report);
    }

    public void SetEvaluationModelListInProgress()
    {
        EvaluationConnection.SetModelListSummary($"Refreshing models from {EvaluationConnection.EvaluationBackend}...");
    }

    public void ApplyEvaluationModelListReport(BackendModelListReport report)
    {
        ApplyModelListReport(
            report,
            EvaluationConnection.EvaluationAvailableModels,
            EvaluationConnection.EvaluationModel,
            model => EvaluationConnection.EvaluationModel = model,
            summary => EvaluationConnection.SetModelListSummary(summary),
            "Evaluation"
        );
    }

    public void SetEvaluationModelListError(string message)
    {
        EvaluationConnection.SetModelListSummary($"Evaluation model refresh failed.{Environment.NewLine}{message}");
        ReportError(message);
    }



    // Sets the short candidate-gate status label + color for the tab header, honest
    // across the three states: a real gate -> its status; content but no gate ->
    // "not run"; nothing to gate -> "n/a". Never green for null/unknown.

    // Revert the candidate-gate state to the neutral initial state (no current verdict).
    // Called whenever there is no meaningful current gate to show — a new run starting, a
    // run that failed, or a project switch — so a prior run's/project's verdict can never
    // linger in the header (or trigger a spurious block-confirm via ActiveAiAssistCandidateGate).







    // ResumeAiAssistRewriteBatch stays on the shell: it loads a saved batch's source draft into the
    // (not-yet-extracted) Writing Studio and sets the run-core action/instruction, reaching into the
    // extracted RewriteBatches tab for the selection. Resolve fully when the AI-Assist core extracts.
    public bool ResumeAiAssistRewriteBatch()
    {
        var selected = RewriteBatches.SelectedAiAssistRewriteBatch;
        if (selected is null)
        {
            RewriteBatches.SetRewriteBatchSummary("Select a saved rewrite batch before resuming.");
            return false;
        }

        WritingStudio.LoadDraft(selected.SourceDraft);
        AiAssist.AiAssistAction = AiAssist.AiAssistActionPresets.Contains(selected.Action)
            ? selected.Action
            : "rewrite-output";
        AiAssist.AiAssistInstruction = selected.Instruction;
        RewriteBatches.SetRewriteBatchSummary(
            $"Resumed rewrite batch for rows {AiAssistRewriteBatchesViewModel.FormatRowNumbers(selected.RowNumbers)}.");
        return true;
    }

    public void SetReviewedFixes(IEnumerable<ReviewedFixRecord> fixes)
    {
        var selectedFixId = SelectedReviewedFix?.FixId;
        ReviewedFixes.Clear();
        foreach (var fix in fixes)
        {
            ReviewedFixes.Add(fix);
        }

        SelectedReviewedFix = ReviewedFixes
            .FirstOrDefault(fix => string.Equals(fix.FixId, selectedFixId, StringComparison.Ordinal))
            ?? ReviewedFixes.FirstOrDefault();

        ReviewedFixSummary = BuildReviewedFixSummary();
    }

    public bool TryGetLastPreparedEvaluationFix(
        out ReviewedFixRecord fix,
        out string errorMessage
    )
    {
        if (_lastPreparedEvaluationFix is null)
        {
            fix = new ReviewedFixRecord();
            errorMessage = "Edit a failed evaluation row before tracking a reviewed fix.";
            return false;
        }

        fix = _lastPreparedEvaluationFix;
        errorMessage = string.Empty;
        return true;
    }

    public void ApplyReviewedFixRecorded(ReviewedFixRecord fix)
    {
        _lastPreparedEvaluationFix = null;
        ReviewedFixSummary =
            $"Tracked reviewed fix for {fix.ExampleId} (v{fix.Version}). Rerun evaluation to confirm the fix.";
    }

    public void ApplyReviewedFixesReconciled()
    {
        ReviewedFixSummary = BuildReviewedFixSummary();
    }

    public bool ResumeReviewedFix()
    {
        if (SelectedReviewedFix is null)
        {
            ReviewedFixSummary = "Select a tracked reviewed fix before reopening it.";
            return false;
        }

        var example = Examples.Items.FirstOrDefault(item => item.RowNumber == SelectedReviewedFix.RowNumber);
        if (example is null)
        {
            ReviewedFixSummary =
                $"Saved row {SelectedReviewedFix.RowNumber} for {SelectedReviewedFix.ExampleId} is not loaded in the Examples list.";
            return false;
        }

        Examples.SelectedExample = example;
        WritingStudio.LoadDraft(example.Json);
        ReviewedFixSummary =
            $"Reopened reviewed fix for {SelectedReviewedFix.ExampleId} (v{SelectedReviewedFix.Version}). Edit, validate, save, then rerun evaluation.";
        return true;
    }

    public void SetReviewedFixError(string message)
    {
        ReviewedFixSummary = $"Reviewed fix could not be updated.{Environment.NewLine}{message}";
    }

    private string BuildReviewedFixSummary()
    {
        if (ReviewedFixes.Count == 0)
        {
            return "No reviewed fixes are tracked for this project.";
        }

        var resolved = ReviewedFixes.Count(fix => fix.Status == ReviewedFixRecord.StatusResolved);
        var stillFailing = ReviewedFixes.Count(fix => fix.Status == ReviewedFixRecord.StatusStillFailing);
        var awaiting = ReviewedFixes.Count(fix => fix.Status == ReviewedFixRecord.StatusEdited);
        return $"Reviewed fixes: {ReviewedFixes.Count} tracked ({resolved} resolved, {stillFailing} still failing, {awaiting} awaiting re-test).";
    }

















    public void SetAiAssistHealthCheckInProgress()
    {
        AiAssist.AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                "Checking AI Assist backend...",
                $"Backend: {AiAssistConnection.AiAssistBackend}",
                $"Model: {AiAssistConnection.AiAssistModel}",
            ]
        );
    }

    public void ApplyAiAssistBackendHealthReport(BackendHealthReport report)
    {
        SetAvailableModels(AiAssistConnection.AiAssistAvailableModels, report.AvailableModels);
        AiAssist.AiAssistSummary = FormatBackendHealthReport("AI Assist backend", report);
    }

    public void SetAiAssistModelListInProgress()
    {
        AiAssistConnection.SetModelListSummary($"Refreshing models from {AiAssistConnection.AiAssistBackend}...");
    }

    public void ApplyAiAssistModelListReport(BackendModelListReport report)
    {
        ApplyModelListReport(
            report,
            AiAssistConnection.AiAssistAvailableModels,
            AiAssistConnection.AiAssistModel,
            model => AiAssistConnection.AiAssistModel = model,
            summary => AiAssistConnection.SetModelListSummary(summary),
            "AI Assist"
        );
    }

    public void SetAiAssistModelListError(string message)
    {
        AiAssistConnection.SetModelListSummary($"AI Assist model refresh failed.{Environment.NewLine}{message}");
        ReportError(message);
    }

    public bool MoveAiAssistSuggestionToDraft()
    {
        var suggestionJsonl = AiAssist.SelectedAiAssistReviewQueueItem?.SuggestedJsonl ?? AiAssist.AiAssistSuggestionJsonl;
        if (string.IsNullOrWhiteSpace(suggestionJsonl))
        {
            AiAssist.AiAssistSummary = "No AI Assist JSONL suggestion is available to move into the draft.";
            return false;
        }

        WritingStudio.LoadDraft(suggestionJsonl.TrimEnd());
        AiAssist.AiAssistSummary = "AI Assist suggestion moved to Writing Studio. Validate and edit before saving.";
        return true;
    }





    public void SetDatasetCardInProgress()
    {
        DatasetCardSummary = "Generating dataset card...";
        DatasetCardPreview = "Waiting for the dataset card.";
    }

    public void ApplyDatasetCardResult(DatasetCardResult result)
    {
        var lines = new List<string>();
        if (!string.IsNullOrWhiteSpace(result.OutputPath))
        {
            lines.Add($"Dataset card: {result.OutputPath}");
        }

        if (result.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(result.Warnings.Select(warning => $"- {warning}"));
        }
        else
        {
            lines.Add("No outstanding warnings.");
        }

        DatasetCardSummary = string.Join(Environment.NewLine, lines);
        DatasetCardPreview = string.IsNullOrWhiteSpace(result.Markdown)
            ? "The dataset card was empty."
            : result.Markdown;
    }

    public void SetDatasetCardError(string message)
    {
        DatasetCardSummary = $"Dataset card could not be generated.{Environment.NewLine}{message}";
        DatasetCardPreview = "No dataset card was generated.";
        ReportError(message);
    }

    public void SetValidationInProgress()
    {
        ValidationSummary = "Running validation...";
        ClearValidationIssues();
    }

    public void ApplyValidationReport(ValidationReport report)
    {
        var status = report.Valid ? "Valid" : "Invalid";
        var lines = new List<string>
        {
            $"{status}: {report.CheckedRows} row(s) checked against `{report.SchemaId}`.",
        };

        if (report.Errors.Count > 0)
        {
            lines.Add("");
            lines.Add("Errors:");
            lines.AddRange(report.Errors.Select(FormatIssue));
        }

        if (report.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(report.Warnings.Select(FormatIssue));
        }

        SetValidationIssues(report);
        ValidationSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetValidationError(string message)
    {
        ClearValidationIssues();
        ValidationSummary = $"Validation could not run.{Environment.NewLine}{message}";
        ReportError(message);
    }

    public void SetImportInProgress(string path)
    {
        ClearValidationIssues();
        ValidationSummary = $"Previewing import:{Environment.NewLine}{path}";
    }

    public void ApplyImportPreview(ImportPreviewReport report)
    {
        var lines = new List<string>
        {
            $"Import preview: {System.IO.Path.GetFileName(report.Path)}",
            $"Schema: {report.SchemaId}",
            $"Total rows: {report.TotalRows}",
            $"Accepted rows: {report.AcceptedRows}",
            $"Rejected rows: {report.RejectedRows}",
        };

        if (report.FailedRows.Count > 0)
        {
            lines.Add("");
            lines.Add("Failed rows:");
            foreach (var failedRow in report.FailedRows.Take(10))
            {
                var errors = failedRow.Errors.Count == 0
                    ? "Unknown error"
                    : string.Join("; ", failedRow.Errors.Select(FormatIssue));
                lines.Add($"- Row {failedRow.RowNumber}: {errors}");
            }

            if (report.FailedRows.Count > 10)
            {
                lines.Add($"...and {report.FailedRows.Count - 10} more failed row(s).");
            }
        }

        ClearValidationIssues();
        ValidationSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetImportError(string message)
    {
        ClearValidationIssues();
        ValidationSummary = $"Import preview could not run.{Environment.NewLine}{message}";
        ReportError(message);
    }

    // The Import Quarantine list/selection/detail moved to QuarantineViewModel in Phase 2. The retry
    // action stays here — it repairs a row by loading it into the not-yet-extracted Writing Studio
    // draft (LoadDraft) and tracks the pending row so a successful save can clear it — reaching into
    // the child for the selected row. Resolve this bridge when Writing Studio decomposes.
    private ImportQuarantineItem? _pendingRetryItem;

    public void RetrySelectedImportQuarantineItem()
    {
        var selected = Quarantine.SelectedImportQuarantineItem;
        if (selected is not null)
        {
            WritingStudio.LoadDraft(selected.Raw);
            // Remember which quarantine row is being repaired so a successful save can
            // remove it (otherwise the record orphans in quarantine forever).
            _pendingRetryItem = selected;
        }
    }

    /// <summary>The quarantine item currently being repaired (set by Retry), consumed once by
    /// a successful save so its record can be cleared. Null when no retry is in flight.</summary>
    public ImportQuarantineItem? TakePendingRetryItem()
    {
        var item = _pendingRetryItem;
        _pendingRetryItem = null;
        return item;
    }









    // The Preference Review display + contrast-metric helpers moved to PreferenceReviewViewModel in
    // Phase 2. Only the two Build*Instruction helpers remain here — they serve the shell's AI-Assist
    // handoff actions above.

    private static string BuildPreferenceJudgeInstruction(
        int rowNumber,
        string prompt,
        string chosen,
        string rejected
    )
    {
        return string.Join(
            " ",
            [
                $"Judge preference strength for row {rowNumber}.",
                "Explain whether the chosen response is clearly better than the rejected response for DPO or reward-model training.",
                "Flag weak contrast, identical claims, missing rationale, or cases where the rejected answer is also acceptable.",
                "Return review notes and suggested tags only; do not automatically accept the row.",
                $"Prompt: {TruncateForInstruction(prompt)}",
                $"Chosen: {TruncateForInstruction(chosen)}",
                $"Rejected: {TruncateForInstruction(rejected)}",
            ]
        );
    }

    private static string BuildPreferenceBatchJudgeInstruction(
        IReadOnlyList<PreferenceReviewItem> items
    )
    {
        var previewRows = items
            .Take(8)
            .Select(item => $"row {item.RowNumber}: {item.Contrast}, overlap {item.TokenOverlap:P0}")
            .ToList();
        return string.Join(
            " ",
            [
                $"Judge {items.Count} visible preference pair(s) for DPO or reward-model readiness.",
                "Prioritize weak or moderate contrast, rejected answers that remain acceptable, missing rationale, and duplicate chosen/rejected wording.",
                "Return concise review notes and suggested tags per row; do not automatically accept rows.",
                $"Visible ranking preview: {string.Join("; ", previewRows)}.",
            ]
        );
    }

    private static string TruncateForInstruction(string value, int maxLength = 360)
    {
        var trimmed = value.Trim();
        return trimmed.Length <= maxLength ? trimmed : $"{trimmed[..maxLength]}...";
    }











    private static bool ContainsSearch(string value, string search)
    {
        return value.Contains(search, StringComparison.OrdinalIgnoreCase);
    }


    private void SetValidationIssues(ValidationReport report)
    {
        ValidationIssues.Clear();
        foreach (var issue in report.Errors.Concat(report.Warnings))
        {
            ValidationIssues.Add(new ValidationIssueNavigationItem(issue));
        }

        SelectedValidationIssue = null;
    }

    private void ClearValidationIssues()
    {
        ValidationIssues.Clear();
        SelectedValidationIssue = null;
    }

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        return true;
    }

    private void OnPropertyChanged(string propertyName)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    private static string FormatIssue(ValidationIssue issue)
    {
        var location = issue.RowNumber is null ? "" : $"Row {issue.RowNumber}: ";
        var field = string.IsNullOrWhiteSpace(issue.Field) ? "" : $" [{issue.Field}]";
        return $"- {location}{issue.Message}{field}";
    }


    /// <summary>Describe the evaluation metric honestly so the score is never read as a
    /// quality judgment when it is only lexical overlap.</summary>
























    private static string BuildSyntheticRewriteInstruction(
        SyntheticPatternIssue issue,
        int rowNumber
    )
    {
        return string.Join(
            " ",
            [
                $"Rewrite row {rowNumber} to address this synthetic quality issue.",
                issue.Message,
                $"Repair guidance: {issue.Suggestion}",
                "Keep the current schema valid, preserve the user's intent, remove boilerplate synthetic phrasing, and return one corrected JSON object only.",
            ]
        );
    }

    private static string BuildSyntheticBatchRewriteInstruction(
        IEnumerable<SyntheticPatternIssue> issues,
        IReadOnlyList<int> rowNumbers
    )
    {
        var issueSummaries = issues
            .Take(8)
            .Select(issue =>
            {
                var severity = string.IsNullOrWhiteSpace(issue.Severity) ? "unknown" : issue.Severity;
                var kind = string.IsNullOrWhiteSpace(issue.Kind) ? "synthetic_pattern" : issue.Kind;
                return $"{severity} {kind}: {issue.Message}";
            });
        return string.Join(
            " ",
            [
                $"Rewrite affected rows {string.Join(", ", rowNumbers)} as a batch.",
                "Remove boilerplate synthetic phrasing, vary repeated openings and endings, and preserve each row's schema and user intent.",
                "Return corrected JSONL rows only, one JSON object per affected row, in the same order.",
                $"Issues: {string.Join(" | ", issueSummaries)}",
            ]
        );
    }

    private static string BuildSyntheticIssueSummary(IEnumerable<SyntheticPatternIssue> issues)
    {
        return string.Join(
            " | ",
            issues
                .Take(8)
                .Select(issue =>
                {
                    var severity = string.IsNullOrWhiteSpace(issue.Severity)
                        ? "unknown"
                        : issue.Severity;
                    var kind = string.IsNullOrWhiteSpace(issue.Kind)
                        ? "synthetic_pattern"
                        : issue.Kind;
                    return $"{severity} {kind}: {issue.Message}";
                })
        );
    }

    private static string BuildEvaluationFailureInstruction(EvaluationExampleResult result)
    {
        return string.Join(
            " ",
            [
                $"Triage failed evaluation example {result.ExampleId}.",
                $"Auto score: {result.Score:0.##}.",
                "Compare the model answer with the expected output, identify why it failed, and suggest a stronger dataset example or clearer expected answer.",
                "Return review notes and, if useful, one corrected JSON object only. Human review remains required.",
                $"Prompt: {TruncateForInstruction(result.Prompt)}",
                $"Expected: {TruncateForInstruction(result.ExpectedOutput)}",
                $"Model output: {TruncateForInstruction(result.ModelOutput)}",
            ]
        );
    }

    private static string BuildEvaluationFailureDraft(
        EvaluationExampleResult result,
        string schemaId
    )
    {
        object row = schemaId == "chat"
            ? new
            {
                messages = new object[]
                {
                    new { role = "user", content = result.Prompt },
                    new { role = "assistant", content = result.ExpectedOutput },
                },
                tags = result.Tags,
            }
            : new
            {
                instruction = result.Prompt,
                input = string.Empty,
                output = result.ExpectedOutput,
                tags = result.Tags,
            };

        return JsonSerializer.Serialize(row, new JsonSerializerOptions { WriteIndented = true });
    }

    private static bool TryParseEvaluationRowNumber(string exampleId, out int rowNumber)
    {
        rowNumber = 0;
        const string rowPrefix = "row-";
        if (string.IsNullOrWhiteSpace(exampleId)
            || !exampleId.StartsWith(rowPrefix, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var rawRowNumber = exampleId[rowPrefix.Length..];
        return int.TryParse(
                rawRowNumber,
                NumberStyles.None,
                CultureInfo.InvariantCulture,
                out rowNumber
            )
            && rowNumber > 0;
    }

    private static string BuildJsonArrayDraft(IEnumerable<string> jsonRows)
    {
        var rows = new List<JsonElement>();
        foreach (var jsonRow in jsonRows)
        {
            using var document = JsonDocument.Parse(jsonRow);
            rows.Add(document.RootElement.Clone());
        }

        return JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true });
    }


    private static string FormatBackendHealthReport(string label, BackendHealthReport report)
    {
        var status = report.Reachable ? "reachable" : "not reachable";
        var modelStatus = report.ModelAvailable ? "available" : "not listed";
        var lines = new List<string>
        {
            $"{label}: {status}",
            $"Provider: {report.ProviderName}",
            $"Base URL: {report.BaseUrl}",
            $"Model: {report.ModelName} ({modelStatus})",
            $"Available models: {report.AvailableModels.Count}",
        };

        if (!string.IsNullOrWhiteSpace(report.Error))
        {
            lines.Add($"Error: {report.Error}");
        }

        return string.Join(Environment.NewLine, lines);
    }

    private static void ApplyModelListReport(
        BackendModelListReport report,
        ObservableCollection<string> target,
        string currentModel,
        Action<string> setModel,
        Action<string> setSummary,
        string label
    )
    {
        SetAvailableModels(target, report.Models);

        if (!report.Reachable)
        {
            var lines = new List<string>
            {
                $"{label} model refresh: backend not reachable.",
                $"Provider: {report.ProviderName}",
                $"Base URL: {report.BaseUrl}",
            };
            if (!string.IsNullOrWhiteSpace(report.Error))
            {
                lines.Add($"Error: {report.Error}");
            }

            setSummary(string.Join(Environment.NewLine, lines));
            return;
        }

        if (target.Count == 0)
        {
            setSummary(
                $"{label} model refresh: backend reachable, but no models were returned by {report.ProviderName}."
            );
            return;
        }

        var selectedModel = currentModel.Trim();
        if (string.IsNullOrWhiteSpace(selectedModel)
            || !target.Any(model => string.Equals(model, selectedModel, StringComparison.Ordinal)))
        {
            selectedModel = target[0];
            setModel(selectedModel);
        }

        setSummary(
            $"{label} model refresh: {target.Count} model(s) loaded from {report.ProviderName}. Selected: {selectedModel}."
        );
    }

    private static void SetAvailableModels(
        ObservableCollection<string> target,
        IEnumerable<string> models
    )
    {
        target.Clear();
        foreach (var model in models
            .Where(model => !string.IsNullOrWhiteSpace(model))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(model => model, StringComparer.OrdinalIgnoreCase))
        {
            target.Add(model);
        }
    }

    private void ApplyBackendSettings(ModelBackendSettings settings, bool isEvaluation)
    {
        var backend = string.IsNullOrWhiteSpace(settings.Backend) ? "ollama" : settings.Backend.Trim();
        var model = string.IsNullOrWhiteSpace(settings.Model) ? "qwen2.5-coder:7b" : settings.Model.Trim();
        var baseUrl = string.IsNullOrWhiteSpace(settings.BaseUrl)
            ? DefaultBaseUrlForBackend(backend)
            : settings.BaseUrl.Trim();
        var timeoutSeconds = settings.TimeoutSeconds <= 0 ? 120 : settings.TimeoutSeconds;

        if (isEvaluation)
        {
            EvaluationConnection.EvaluationBackend = backend;
            EvaluationConnection.EvaluationModel = model;
            EvaluationConnection.EvaluationBaseUrl = baseUrl;
            EvaluationConnection.EvaluationTimeoutSeconds = timeoutSeconds.ToString();
            return;
        }

        AiAssistConnection.AiAssistBackend = backend;
        AiAssistConnection.AiAssistModel = model;
        AiAssistConnection.AiAssistBaseUrl = baseUrl;
        AiAssistConnection.AiAssistTimeoutSeconds = timeoutSeconds.ToString();
    }

    private static string DefaultBaseUrlForBackend(string backend)
    {
        return backend.Replace("_", "-", StringComparison.Ordinal).ToLowerInvariant() switch
        {
            "openai-compatible" or "lm-studio" => "http://localhost:1234/v1",
            _ => "http://localhost:11434",
        };
    }

    private static int ParsePositiveIntOrDefault(string value, int defaultValue)
    {
        return int.TryParse(value, out var parsed) && parsed > 0 ? parsed : defaultValue;
    }

    private static string BuildDraftTemplate(string schemaId)
    {
        return schemaId switch
        {
            "raw_text" => "{\n  \"text\": \"A compiler translates source code into machine instructions.\"\n}",
            "chat" => "{\n  \"messages\": [\n    {\"role\": \"user\", \"content\": \"What is recursion?\"},\n    {\"role\": \"assistant\", \"content\": \"Recursion is when a function calls itself.\"}\n  ]\n}",
            "preference" => "{\n  \"prompt\": \"Explain recursion simply.\",\n  \"chosen\": \"Recursion is when a function calls itself.\",\n  \"rejected\": \"Recursion is when code does things again.\"\n}",
            _ => "{\n  \"instruction\": \"Explain what a variable is.\",\n  \"input\": \"\",\n  \"output\": \"A variable stores a value.\"\n}",
        };
    }
}
