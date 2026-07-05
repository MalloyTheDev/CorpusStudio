using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;

using CorpusStudio.Desktop.Models;
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

public sealed class MainWindowViewModel : INotifyPropertyChanged
{
    private string _activeProjectTitle = "New Dataset Project";
    private string? _activeProjectPath;
    private string _activeSchemaId = "instruction";
    private string _activeSchemaDescription =
        "Choose a schema, write examples, validate rows, and export model-ready JSONL.";
    private string _validationSummary = "Create a project to start validation.";
    private string _qualitySummary = "Create or select a project to run quality checks.";
    private bool _hasQualityMetrics;
    private string _qualityStatusLine = string.Empty;
    private string _qualityStatusColor = "#64748B";
    private string _qualityStatusBackground = "#F1F5F9";
    private string _qualityDetail = string.Empty;
    private bool _hasQualityDetail;
    private string _gateSummary = "Run gates to check whether this dataset may move forward.";
    private bool _problemsPanelVisible;
    private string _problemsSummary = "Run gates to check this dataset for problems.";
    private string _problemsBadge = string.Empty;
    private string _problemsBadgeColor = GateReport.StatusColor(null);
    private bool _outputPanelVisible;
    private bool _searchPanelVisible;
    private string _qualityHistorySummary = "Quality history appears after quality checks run.";
    private bool _hasDebtTrend;
    private string _debtTrendDirection = string.Empty;
    private string _debtTrendDirectionColor = "#64748B";
    private string _debtTrendSummary = "Run quality checks to build a debt trend.";
    private string _qualityTriageSummary = "Synthetic quality issues appear here after quality checks run.";
    // Evaluation tab core (run + report panes + per-example results + failure filters + report
    // history) extracted to EvaluationViewModel in Phase 2 (backend-cluster slice 3, PR 3b). The
    // shell keeps the bridges (health methods, eval->AiAssist/ReviewedFix, Training-baseline) reaching in.
    private string _benchmarkModelsInput = string.Empty;
    private string _benchmarkSummary =
        "Enter one model per line, then benchmark them against this project's examples.";
    // AI-Assist tab core (run + result panes + candidate gate + review queue + saved views)
    // extracted to AiAssistViewModel in Phase 2 (backend-cluster slice 2, PR 3/3). The shell
    // keeps the bridges (synthetic/eval/preference/resume writers, health methods) reaching in.
    private string _reviewedFixSummary =
        "Edited failed rows appear here so you can track which fixes were re-tested.";
    private string _trainingTarget = "axolotl_yaml";
    private string _trainingBaseModel = "Qwen/Qwen2.5-Coder-7B-Instruct";
    private string _trainingFormat = "instruction";
    private string _trainingSequenceLen = "4096";
    private string _trainingLoraR = "16";
    private string _trainingLoraAlpha = "32";
    private string _trainingMicroBatchSize = "1";
    private string _trainingGradientAccumulationSteps = "8";
    private string _trainingLearningRate = "0.0002";
    private string _trainingSummary =
        "Generate a training config after validation, splits, and evaluation checks.";
    private string _trainingConfigPreview = "Training config preview appears here.";
    private string _trainingLaunchCommand = string.Empty;
    private IReadOnlyList<string> _trainingLaunchArgv = [];
    private string _trainingLaunchWorkingDirectory = string.Empty;
    private string _trainingOutputDirectory = string.Empty;
    private string _trainingConfigPath = string.Empty;
    private IReadOnlyList<string> _trainingCheckpointNames = [];
    private string _trainingRunHistorySummary = "Refresh to see past training runs recorded for this project.";
    private string _trainingRunGateSummary = "Gate a run to check for regression vs its baseline.";
    private string _trainingCheckpointsSummary =
        "Checkpoints appear here after a training run writes them.";
    private IReadOnlyList<string> _trainingResumeArgv = [];
    private string _trainingResumeCommand = string.Empty;
    private EvaluationReportHistoryItem? _trainingBaselineReport;
    private string _trainingComparisonSummary =
        "Run an evaluation before training to capture a baseline for before/after comparison.";
    private readonly List<string> _trainingRunLines = [];
    private int _trainingRunId;
    private string _trainingRunLog = "Launch training after generating a config; live logs appear here.";
    private string _trainingRunStatus = "Idle";
    private bool _isTrainingRunning;
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
    private SyntheticPatternIssue? _selectedSyntheticPatternIssue;
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

    /// <summary>Design-time / test constructor.</summary>
    public MainWindowViewModel()
        : this(
            new DebtViewModel(), new ArenaViewModel(), new SettingsViewModel(),
            new VersionsViewModel(), new ArtifactsViewModel(), new SuitesViewModel(),
            new SplitsViewModel(), new PreferenceReviewViewModel(), new QuarantineViewModel(),
            new ExamplesViewModel(), new WritingStudioViewModel(), new AiAssistRewriteBatchesViewModel(),
            new AiAssistConnectionViewModel(), new EvaluationConnectionViewModel())
    {
    }

    public MainWindowViewModel(
        IDebtViewModel debt, IArenaViewModel arena, ISettingsViewModel settings,
        IVersionsViewModel versions, IArtifactsViewModel artifacts, ISuitesViewModel suites,
        ISplitsViewModel splits, IPreferenceReviewViewModel preferenceReview,
        IQuarantineViewModel quarantine, IExamplesViewModel examples,
        IWritingStudioViewModel writingStudio, IAiAssistRewriteBatchesViewModel rewriteBatches,
        IAiAssistConnectionViewModel aiAssistConnection, IEvaluationConnectionViewModel evaluationConnection)
    {
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
        // The AI-Assist core is composed from the shared connection instance (its run reads the
        // backend/model), rather than DI-injected, so both share one AiAssistConnection.
        AiAssist = new AiAssistViewModel(aiAssistConnection);
        // Likewise the Evaluation core shares its connection instance (its run reads backend/model).
        Evaluation = new EvaluationViewModel(evaluationConnection);
        // A run/split failure surfaces in the shell's shared error banner; the tabs keep no shell
        // reference, so they raise ErrorReported and the shell forwards it to ReportError.
        Splits.ErrorReported += ReportError;
        AiAssist.ErrorReported += ReportError;
        Evaluation.ErrorReported += ReportError;
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

    public ObservableCollection<SyntheticPatternIssue> SyntheticPatternIssues { get; } = [];











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

    public SyntheticPatternIssue? SelectedSyntheticPatternIssue
    {
        get => _selectedSyntheticPatternIssue;
        set
        {
            if (SetField(ref _selectedSyntheticPatternIssue, value))
            {
                QualityTriageSummary = value is null
                    ? "Select a synthetic quality issue to prepare a rewrite."
                    : FormatSyntheticTriageSummary(value);
            }
        }
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

    public string QualitySummary
    {
        get => _qualitySummary;
        private set => SetField(ref _qualitySummary, value);
    }

    // ---- Quality metric grid (v1.2.12) -------------------------------------------
    // A structured view of the core quality counts for the right panel, replacing the text
    // blob. QualitySummary (the full text) is still built for the dashboard card. The status
    // banner is PII-aware (unlike the legacy `health` line, which ignores PII).

    /// <summary>The core quality counts as scannable rows (Examples + the six issue metrics).</summary>
    public ObservableCollection<QualityMetric> QualityMetrics { get; } = [];

    public bool HasQualityMetrics
    {
        get => _hasQualityMetrics;
        private set => SetField(ref _hasQualityMetrics, value);
    }

    public string QualityStatusLine
    {
        get => _qualityStatusLine;
        private set => SetField(ref _qualityStatusLine, value);
    }

    public string QualityStatusColor
    {
        get => _qualityStatusColor;
        private set => SetField(ref _qualityStatusColor, value);
    }

    public string QualityStatusBackground
    {
        get => _qualityStatusBackground;
        private set => SetField(ref _qualityStatusBackground, value);
    }

    /// <summary>The optional flagged-row detail (PII findings, token outliers, category
    /// imbalance, synthetic clusters/samples) — empty when the dataset is clean.</summary>
    public string QualityDetail
    {
        get => _qualityDetail;
        private set => SetField(ref _qualityDetail, value);
    }

    public bool HasQualityDetail
    {
        get => _hasQualityDetail;
        private set => SetField(ref _hasQualityDetail, value);
    }

    private void ResetQualityMetrics()
    {
        QualityMetrics.Clear();
        HasQualityMetrics = false;
        QualityStatusLine = string.Empty;
        QualityStatusColor = "#64748B";
        QualityStatusBackground = "#F1F5F9";
        QualityDetail = string.Empty;
        HasQualityDetail = false;
    }

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

    public string QualityHistorySummary
    {
        get => _qualityHistorySummary;
        private set => SetField(ref _qualityHistorySummary, value);
    }

    // ---- Debt trend (v1.2.9) -----------------------------------------------------
    // A mini-chart of the quality issue rate across recorded quality runs, built from the
    // existing quality history (nothing new persisted). Not the A-F grade: presence-based
    // PII/secrets aren't in the history, so only the issue-rate trend is honest to plot.

    /// <summary>Ordered oldest → newest bars for the debt-trend mini-chart.</summary>
    public ObservableCollection<DebtTrendPoint> DebtTrend { get; } = [];

    /// <summary>True when there is at least one bar to draw (chart visibility).</summary>
    public bool HasDebtTrendPoints => DebtTrend.Count > 0;

    /// <summary>True when there are ≥2 runs — enough to state a direction.</summary>
    public bool HasDebtTrend
    {
        get => _hasDebtTrend;
        private set => SetField(ref _hasDebtTrend, value);
    }

    public string DebtTrendDirection
    {
        get => _debtTrendDirection;
        private set => SetField(ref _debtTrendDirection, value);
    }

    public string DebtTrendDirectionColor
    {
        get => _debtTrendDirectionColor;
        private set => SetField(ref _debtTrendDirectionColor, value);
    }

    public string DebtTrendSummary
    {
        get => _debtTrendSummary;
        private set => SetField(ref _debtTrendSummary, value);
    }

    private void ApplyDebtTrend(IReadOnlyList<QualityHistoryEntry> history)
    {
        var result = Models.DebtTrend.Build(history);
        DebtTrend.Clear();
        foreach (var point in result.Points)
        {
            DebtTrend.Add(point);
        }
        HasDebtTrend = result.HasTrend;
        DebtTrendDirection = result.Direction;
        DebtTrendDirectionColor = result.DirectionColor;
        DebtTrendSummary = result.Summary;
        OnPropertyChanged(nameof(HasDebtTrendPoints));
    }

    private void ResetDebtTrend()
    {
        DebtTrend.Clear();
        HasDebtTrend = false;
        DebtTrendDirection = string.Empty;
        DebtTrendDirectionColor = "#64748B";
        DebtTrendSummary = "Run quality checks to build a debt trend.";
        OnPropertyChanged(nameof(HasDebtTrendPoints));
    }

    public string QualityTriageSummary
    {
        get => _qualityTriageSummary;
        private set => SetField(ref _qualityTriageSummary, value);
    }

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


    public string TrainingTarget
    {
        get => _trainingTarget;
        set => SetField(ref _trainingTarget, value);
    }

    public string TrainingBaseModel
    {
        get => _trainingBaseModel;
        set => SetField(ref _trainingBaseModel, value);
    }

    public string TrainingFormat
    {
        get => _trainingFormat;
        set => SetField(ref _trainingFormat, value);
    }

    public string TrainingSequenceLen
    {
        get => _trainingSequenceLen;
        set => SetField(ref _trainingSequenceLen, value);
    }

    public string TrainingLoraR
    {
        get => _trainingLoraR;
        set => SetField(ref _trainingLoraR, value);
    }

    public string TrainingLoraAlpha
    {
        get => _trainingLoraAlpha;
        set => SetField(ref _trainingLoraAlpha, value);
    }

    public string TrainingMicroBatchSize
    {
        get => _trainingMicroBatchSize;
        set => SetField(ref _trainingMicroBatchSize, value);
    }

    public string TrainingGradientAccumulationSteps
    {
        get => _trainingGradientAccumulationSteps;
        set => SetField(ref _trainingGradientAccumulationSteps, value);
    }

    public string TrainingLearningRate
    {
        get => _trainingLearningRate;
        set => SetField(ref _trainingLearningRate, value);
    }

    public string TrainingSummary
    {
        get => _trainingSummary;
        private set => SetField(ref _trainingSummary, value);
    }

    public string TrainingConfigPreview
    {
        get => _trainingConfigPreview;
        private set => SetField(ref _trainingConfigPreview, value);
    }

    /// <summary>The exact launch command from the last training-config export
    /// (empty if none). Backs the "Copy launch command" action.</summary>
    public string TrainingLaunchCommand
    {
        get => _trainingLaunchCommand;
        private set => SetField(ref _trainingLaunchCommand, value);
    }

    public string TrainingRunLog
    {
        get => _trainingRunLog;
        private set => SetField(ref _trainingRunLog, value);
    }

    public string TrainingRunStatus
    {
        get => _trainingRunStatus;
        private set => SetField(ref _trainingRunStatus, value);
    }

    public bool IsTrainingRunning
    {
        get => _isTrainingRunning;
        private set
        {
            if (SetField(ref _isTrainingRunning, value))
            {
                OnPropertyChanged(nameof(CanLaunchTraining));
                OnPropertyChanged(nameof(CanResumeTraining));
            }
        }
    }

    /// <summary>Whether a run can be launched (a config was generated and none is running).</summary>
    public bool CanLaunchTraining => !_isTrainingRunning && _trainingLaunchArgv.Count > 0;

    /// <summary>The structured command to spawn (empty until a config is generated).</summary>
    public IReadOnlyList<string> TrainingLaunchArgv => _trainingLaunchArgv;

    public string TrainingLaunchWorkingDirectory => _trainingLaunchWorkingDirectory;

    public const int TrainingLogMaxLines = 2000;

    /// <summary>Start a run and return its id; log appends are tagged with this
    /// id so a prior (cancelled) run's late output cannot contaminate this one.</summary>
    public int BeginTrainingRun()
    {
        _trainingRunLines.Clear();
        TrainingRunLog = string.Empty;
        TrainingRunStatus = "Running...";
        IsTrainingRunning = true;
        return ++_trainingRunId;
    }

    public void AppendTrainingRunLog(string line)
    {
        _trainingRunLines.Add(line);
        TrimAndPublishTrainingLog();
    }

    /// <summary>Append a batch of streamed lines for a specific run. Lines tagged
    /// with a stale run id (from a cancelled run) are dropped.</summary>
    public void AppendTrainingRunLogBatch(int runId, IReadOnlyList<string> lines)
    {
        if (runId != _trainingRunId || lines.Count == 0)
        {
            return;
        }

        _trainingRunLines.AddRange(lines);
        TrimAndPublishTrainingLog();
    }

    private void TrimAndPublishTrainingLog()
    {
        if (_trainingRunLines.Count > TrainingLogMaxLines)
        {
            _trainingRunLines.RemoveRange(0, _trainingRunLines.Count - TrainingLogMaxLines);
        }

        TrainingRunLog = string.Join(Environment.NewLine, _trainingRunLines);
    }

    public void CompleteTrainingRun(int exitCode)
    {
        IsTrainingRunning = false;
        TrainingRunStatus = exitCode == 0
            ? "Completed (exit 0)"
            : $"Failed (exit {exitCode})";
    }

    public void SetTrainingRunCancelled()
    {
        IsTrainingRunning = false;
        TrainingRunStatus = "Cancelled";
    }

    public void SetTrainingRunError(string message)
    {
        IsTrainingRunning = false;
        TrainingRunStatus = "Error";
        AppendTrainingRunLog($"[error] {message}");
    }

    public string TrainingRunHistorySummary
    {
        get => _trainingRunHistorySummary;
        private set => SetField(ref _trainingRunHistorySummary, value);
    }

    public string TrainingRunGateSummary
    {
        get => _trainingRunGateSummary;
        private set => SetField(ref _trainingRunGateSummary, value);
    }

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

    public void SetTrainingRunGateError(string message)
    {
        TrainingRunGateSummary = $"Regression gate could not run.{Environment.NewLine}{message}";
    }

    /// <summary>Format a training-run regression gate verdict (pass/warn/block).</summary>
    public void ApplyTrainingRunGate(GateReport report)
    {
        var mark = report.OverallStatus switch
        {
            "block" => "⛔ BLOCK",
            "warn" => "⚠ WARN",
            _ => "✅ PASS",
        };
        var result = report.Results.Count > 0 ? report.Results[0] : null;
        TrainingRunGateSummary = result is null
            ? $"Regression gate: {mark}"
            : $"Regression gate: {mark} — {result.Message}";
    }

    public void SetTrainingRunHistoryError(string message)
    {
        TrainingRunHistorySummary = $"Run history could not load.{Environment.NewLine}{message}";
    }

    /// <summary>Format the durable run registry (newest first). Reconciliation of
    /// stuck `running` records happens in the service before this is called.</summary>
    public void ApplyTrainingRunHistory(IReadOnlyList<TrainingRunRecord> records)
    {
        if (records.Count == 0)
        {
            TrainingRunHistorySummary = "No training runs recorded yet.";
            return;
        }

        var lines = new List<string> { $"Training runs ({records.Count}, newest first):", string.Empty };
        foreach (var record in records)
        {
            lines.Add($"[{record.Status}] {record.RunId} — {record.BaseModel} ({record.Target})");
            var bits = new List<string>
            {
                $"{record.Checkpoints?.Count ?? 0} checkpoint(s)",
                string.IsNullOrWhiteSpace(record.BeforeEvalPath) ? "before-eval –" : "before-eval ✓",
                string.IsNullOrWhiteSpace(record.AfterEvalPath) ? "after-eval –" : "after-eval ✓",
            };
            if (record.ExitCode is { } exit)
            {
                bits.Add($"exit {exit}");
            }
            lines.Add("   " + string.Join("; ", bits));
        }

        TrainingRunHistorySummary = string.Join(Environment.NewLine, lines);
    }

    /// <summary>Where the trainer writes checkpoints (from the last config export).</summary>
    public string TrainingOutputDirectory => _trainingOutputDirectory;

    /// <summary>The rendered config path from the last config export.</summary>
    public string TrainingConfigPath => _trainingConfigPath;

    public string TrainingCheckpointsSummary
    {
        get => _trainingCheckpointsSummary;
        private set => SetField(ref _trainingCheckpointsSummary, value);
    }

    /// <summary>The exact resume command for the latest checkpoint (empty if none).</summary>
    public string TrainingResumeCommand => _trainingResumeCommand;

    public IReadOnlyList<string> TrainingResumeArgv => _trainingResumeArgv;

    /// <summary>Resume is available when the target supports a resume flag, a
    /// checkpoint exists, and no run is active.</summary>
    public bool CanResumeTraining => !_isTrainingRunning && _trainingResumeArgv.Count > 0;

    public string TrainingComparisonSummary
    {
        get => _trainingComparisonSummary;
        private set => SetField(ref _trainingComparisonSummary, value);
    }

    /// <summary>The "before" evaluation report captured at training launch (null if
    /// no evaluation had been saved yet).</summary>
    public EvaluationReportHistoryItem? TrainingBaselineReport => _trainingBaselineReport;

    /// <summary>Capture the pre-training baseline (the newest saved evaluation
    /// report at launch time, or null when none exists).</summary>
    public void SetTrainingBaseline(EvaluationReportHistoryItem? baseline)
    {
        _trainingBaselineReport = baseline;
        TrainingComparisonSummary = baseline is null
            ? "No baseline: no evaluation report existed when this run started. "
              + "Run an evaluation before the next training run to enable before/after comparison."
            : $"Baseline captured: {baseline.DisplayName}{Environment.NewLine}"
              + "After training: load the trained adapter into your local backend, run an "
              + "evaluation against it, then click Compare vs baseline.";
    }

    /// <summary>Compare the newest post-training evaluation report against the
    /// captured baseline. <paramref name="history"/> is newest-first.</summary>
    public void CompareTrainingBaseline(IReadOnlyList<EvaluationReportHistoryItem> history)
    {
        if (_trainingBaselineReport is null)
        {
            TrainingComparisonSummary =
                "No baseline was captured for the last training run. Run an evaluation, "
                + "train, then evaluate the trained model to compare.";
            return;
        }

        var after = history.FirstOrDefault(item => !string.Equals(
            item.ReportPath,
            _trainingBaselineReport.ReportPath,
            StringComparison.OrdinalIgnoreCase
        ));
        if (after is null)
        {
            TrainingComparisonSummary =
                "No post-training evaluation found yet. Load the trained adapter into your "
                + "local backend and run an evaluation, then compare again.";
            return;
        }

        if (after.LastModified < _trainingBaselineReport.LastModified)
        {
            TrainingComparisonSummary =
                "The newest other report is older than the baseline. Run an evaluation of "
                + "the trained model first, then compare again.";
            return;
        }

        TrainingComparisonSummary =
            $"Before/after (after − before):{Environment.NewLine}"
            + Evaluation.BuildEvaluationReportComparison(after, _trainingBaselineReport);
    }

    public IReadOnlyList<string> TrainingCheckpointNames => _trainingCheckpointNames;

    public void ApplyTrainingCheckpoints(TrainingCheckpointsResult result)
    {
        _trainingCheckpointNames = result.Checkpoints.ToArray();
        if (result.Checkpoints.Count == 0)
        {
            TrainingCheckpointsSummary = "No checkpoints found yet.";
            _trainingResumeArgv = [];
            _trainingResumeCommand = string.Empty;
        }
        else
        {
            TrainingCheckpointsSummary =
                $"Checkpoints: {result.Checkpoints.Count} (latest {result.LatestCheckpoint})";
            var resumeReady = result.ResumeSupported == true
                && result.LatestCheckpoint is not null
                && result.ResumeArgv is { Count: > 0 };
            _trainingResumeArgv = resumeReady ? result.ResumeArgv!.ToArray() : [];
            _trainingResumeCommand = resumeReady ? result.ResumeCommand ?? string.Empty : string.Empty;
            if (!resumeReady && result.ResumeSupported == false)
            {
                TrainingCheckpointsSummary +=
                    " — resume is config-driven for this target; set the checkpoint in the config.";
            }
        }

        OnPropertyChanged(nameof(CanResumeTraining));
    }

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
        QualitySummary = "Quality checks will appear after examples are added.";
        ResetQualityMetrics();
        QualityHistorySummary = "Quality history appears after quality checks run.";
        ResetDebtTrend();
        QualityTriageSummary = "Synthetic quality issues appear here after quality checks run.";
        SyntheticPatternIssues.Clear();
        SelectedSyntheticPatternIssue = null;
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
        TrainingFormat = project.SchemaId;
        TrainingSummary = "Generate a training config after validation, splits, and evaluation checks.";
        TrainingConfigPreview = "Training config preview appears here.";
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
        QualitySummary = Examples.Items.Count == 0
            ? "No saved examples yet. Quality checks will run after examples are added."
            : $"{Examples.Items.Count} saved example(s). Run quality checks to inspect duplicates and empty rows.";
        // The dataset changed: any prior debt grade is now stale. Invalidate it so the
        // Debt tab can never show a verdict that no longer matches the data.
        Debt.InvalidateDebt();
    }

    public void SetQualityInProgress()
    {
        QualitySummary = "Running quality checks...";
        ResetQualityMetrics();
        QualityTriageSummary = "Refreshing synthetic quality triage...";
    }

    public void ApplyQualityReport(
        QualityReport report,
        IReadOnlyList<QualityHistoryEntry>? history = null
    )
    {
        var health = report.EmptyRowCount == 0
            && report.DuplicateExactCount == 0
            && report.DuplicateNormalizedCount == 0
            && report.LowInformationCount == 0
            && report.SyntheticPatternCount == 0
            ? "No basic quality issues found."
            : "Review the flagged rows before export.";

        var coreLines = new List<string>
        {
                $"Examples: {report.ExampleCount}",
                $"Empty rows: {report.EmptyRowCount}",
                $"Exact duplicates: {report.DuplicateExactCount}",
                $"Normalized duplicates: {report.DuplicateNormalizedCount}",
                $"Low-information rows: {report.LowInformationCount} (< {report.LowInformationTokenThreshold} tokens)",
                $"Synthetic pattern warnings: {report.SyntheticPatternCount}",
                $"Possible PII / secrets: {report.PiiFindingCount}",
                $"Status: {health}",
        };

        // The optional flagged-row sections (shared by the full-text summary and the panel's
        // detail block) so QualitySummary stays byte-identical for the dashboard card.
        var detailLines = BuildQualityDetailLines(report);

        QualitySummary = string.Join(Environment.NewLine, coreLines.Concat(detailLines));

        // Structured metric grid + PII-aware status banner for the right panel.
        BuildQualityMetrics(report);
        QualityDetail = string.Join(Environment.NewLine, detailLines).Trim();
        HasQualityDetail = QualityDetail.Length > 0;

        SetSyntheticPatternIssues(report.SyntheticPatternIssues);

        ApplyQualityHistory(history ?? []);
    }

    /// <summary>Build the structured Quality metric rows (Examples + the six issue counts) and
    /// the status banner. The banner is PII-aware: any PII/secret finding is a red problem, even
    /// though the legacy `health` line — kept for QualitySummary — does not weigh PII.</summary>
    private void BuildQualityMetrics(QualityReport report)
    {
        QualityMetrics.Clear();
        QualityMetrics.Add(QualityMetric.Info("Examples", report.ExampleCount));
        QualityMetrics.Add(QualityMetric.Issue("Empty rows", report.EmptyRowCount));
        QualityMetrics.Add(QualityMetric.Issue("Exact duplicates", report.DuplicateExactCount));
        QualityMetrics.Add(QualityMetric.Issue("Normalized duplicates", report.DuplicateNormalizedCount));
        QualityMetrics.Add(QualityMetric.Issue("Low-information rows", report.LowInformationCount));
        QualityMetrics.Add(QualityMetric.Issue("Synthetic pattern warnings", report.SyntheticPatternCount));
        QualityMetrics.Add(QualityMetric.Issue("Possible PII / secrets", report.PiiFindingCount, severe: true));
        HasQualityMetrics = true;

        var coreIssues = report.EmptyRowCount + report.DuplicateExactCount + report.DuplicateNormalizedCount
                         + report.LowInformationCount + report.SyntheticPatternCount;

        if (report.PiiFindingCount > 0)
        {
            QualityStatusLine = "Possible PII / secrets detected — review before export.";
            QualityStatusColor = "#B91C1C";
            QualityStatusBackground = "#FEF2F2";
        }
        else if (coreIssues > 0)
        {
            QualityStatusLine = "Review the flagged rows before export.";
            QualityStatusColor = "#B45309";
            QualityStatusBackground = "#FFFBEB";
        }
        else
        {
            QualityStatusLine = "No basic quality issues found.";
            QualityStatusColor = "#15803D";
            QualityStatusBackground = "#ECFDF5";
        }
    }

    private List<string> BuildQualityDetailLines(QualityReport report)
    {
        var lines = new List<string>();

        if (report.PiiFindingCount > 0)
        {
            lines.Add("");
            lines.Add($"⚠ Possible PII / secrets detected ({report.PiiFindingCount} kind(s)) — review before exporting:");
            lines.AddRange(report.PiiFindings.Take(5).Select(finding => $"- {finding.DisplayName}"));
        }

        if (report.TokenLengthOutlierCount > 0)
        {
            lines.Add("");
            lines.Add($"Token-length outliers: {report.TokenLengthOutlierCount} row(s) over ~{report.TokenLengthThreshold} tokens");
            lines.AddRange(report.TokenLengthOutliers
                .Take(3)
                .Select(outlier => $"- row {outlier.RowNumber}: ~{outlier.TokenCount} tokens"));
        }

        if (report.CategoryImbalances.Count > 0)
        {
            lines.Add("");
            lines.Add($"Category imbalance: {report.CategoryImbalances.Count} field(s) dominated by one value");
            lines.AddRange(report.CategoryImbalances.Take(3).Select(item => $"- {item.DisplayName}"));
        }

        if (report.SyntheticPatternClusters.Count > 0)
        {
            lines.Add("");
            lines.Add($"Synthetic pattern clusters: {report.SyntheticPatternClusters.Count} (near-duplicate families)");
            lines.AddRange(report.SyntheticPatternClusters.Take(3).Select(cluster => $"- {cluster.DisplayName}"));
        }

        if (report.SyntheticPatternIssues.Count > 0)
        {
            lines.Add("");
            lines.Add("Synthetic pattern samples:");
            lines.AddRange(report.SyntheticPatternIssues.Take(3).Select(FormatSyntheticPatternIssue));
        }
        else if (report.SyntheticPatternWarnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Synthetic pattern samples:");
            lines.AddRange(report.SyntheticPatternWarnings.Take(3).Select(warning => $"- {warning}"));
        }

        return lines;
    }

    public void SetQualityError(string message)
    {
        QualitySummary = $"Quality checks could not run.{Environment.NewLine}{message}";
        ResetQualityMetrics();
        QualityTriageSummary = "Synthetic quality triage could not be refreshed.";
        ReportError(message);
    }

    public bool PrepareSyntheticIssueRewrite()
    {
        if (SelectedSyntheticPatternIssue is null)
        {
            QualityTriageSummary = "Select a synthetic quality issue before preparing a rewrite.";
            return false;
        }

        var rowNumber = SelectedSyntheticPatternIssue.RowNumbers.FirstOrDefault(row => row > 0);
        if (rowNumber <= 0)
        {
            QualityTriageSummary = "Selected synthetic quality issue does not include an affected row number.";
            return false;
        }

        var example = Examples.Items.FirstOrDefault(item => item.RowNumber == rowNumber);
        if (example is null)
        {
            QualityTriageSummary = $"Affected row {rowNumber} is not loaded in the Examples list.";
            return false;
        }

        Examples.SelectedExample = example;
        WritingStudio.LoadDraft(example.Json);
        if (AiAssist.AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssist.AiAssistAction = "rewrite-output";
        }

        AiAssist.AiAssistInstruction = BuildSyntheticRewriteInstruction(SelectedSyntheticPatternIssue, rowNumber);
        QualityTriageSummary =
            $"Prepared row {rowNumber} for AI Assist rewrite. Review, run AI Assist, validate, and save only after editing.";
        return true;
    }

    public bool PrepareSyntheticBatchRewrite()
    {
        if (SyntheticPatternIssues.Count == 0)
        {
            QualityTriageSummary = "Run quality checks with synthetic warnings before preparing a batch rewrite.";
            return false;
        }

        var rowNumbers = SyntheticPatternIssues
            .SelectMany(issue => issue.RowNumbers)
            .Where(rowNumber => rowNumber > 0)
            .Distinct()
            .Order()
            .Take(12)
            .ToList();
        if (rowNumbers.Count == 0)
        {
            QualityTriageSummary = "Synthetic quality issues do not include affected row numbers.";
            return false;
        }

        var affectedRows = rowNumbers
            .Select(rowNumber => Examples.Items.FirstOrDefault(example => example.RowNumber == rowNumber))
            .Where(example => example is not null)
            .Cast<SavedExampleItem>()
            .ToList();
        if (affectedRows.Count == 0)
        {
            QualityTriageSummary = "Affected synthetic rows are not loaded in the Examples list.";
            return false;
        }

        WritingStudio.LoadDraft(BuildJsonArrayDraft(affectedRows.Select(row => row.Json)));
        if (AiAssist.AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssist.AiAssistAction = "rewrite-output";
        }

        AiAssist.AiAssistInstruction = BuildSyntheticBatchRewriteInstruction(SyntheticPatternIssues, rowNumbers);
        RewriteBatches.SetLastPrepared(new AiAssistRewriteBatch
        {
            SchemaId = ActiveSchemaId,
            Action = "rewrite-output",
            RowNumbers = rowNumbers,
            IssueCount = SyntheticPatternIssues.Count,
            IssueSummary = BuildSyntheticIssueSummary(SyntheticPatternIssues),
            SourceDraft = WritingStudio.DraftText,
            Instruction = AiAssist.AiAssistInstruction,
        });
        QualityTriageSummary =
            $"Prepared {affectedRows.Count} affected row(s) from {SyntheticPatternIssues.Count} synthetic issue(s) for batch rewrite.";
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

    public void SetTrainingConfigInProgress()
    {
        TrainingSummary = string.Join(
            Environment.NewLine,
            [
                "Generating training config...",
                $"Target: {TrainingTarget}",
                $"Base model: {TrainingBaseModel}",
            ]
        );
        TrainingConfigPreview = "Waiting for config export.";
    }

    public void ApplyTrainingConfigExportResult(TrainingConfigExportResult result)
    {
        var launcherStatus = result.TrainingLauncherImplemented ? "implemented" : "not implemented";
        var lines = new List<string>
        {
            $"Target: {result.Target}",
            $"Config: {result.OutputPath}",
            $"Training launcher: {launcherStatus}",
        };

        if (result.TokenBudget is { } budget && budget.ExampleCount > 0)
        {
            lines.Add("");
            lines.Add(
                $"Token budget ({budget.Method}): ~{budget.EstimatedTokens:N0} tokens over "
                + $"{budget.ExampleCount} example(s), ~{budget.TokensPerEpoch:N0}/epoch at seq_len "
                + $"{budget.SequenceLen}");
            lines.Add(
                $"  mean ~{budget.MeanTokensPerExample:N0}, max ~{budget.MaxTokensInExample:N0} tokens; "
                + $"{budget.ExamplesOverSequenceLen} over seq_len");
        }

        if (result.VramEstimate is { } vram)
        {
            lines.Add("");
            if (vram.ParameterCountBillions is { } paramsB)
            {
                lines.Add(
                    $"VRAM (rough, {paramsB:0.#}B params): ~{vram.TotalGbFp16:0.#} GB fp16 / "
                    + $"~{vram.TotalGbInt8:0.#} GB 8-bit / ~{vram.TotalGbInt4:0.#} GB 4-bit");
            }
            else
            {
                lines.Add("VRAM: no estimate (model size not parseable from the name).");
            }
        }

        if (result.LoraRecommendation is { } lora)
        {
            lines.Add($"LoRA suggestion: r={lora.RecommendedR}, alpha={lora.RecommendedAlpha}");
            lines.AddRange(lora.Warnings.Select(warning => $"- {warning}"));
        }

        _trainingOutputDirectory = result.TrainingOutputDirectory;
        _trainingConfigPath = result.OutputPath;
        _trainingResumeArgv = [];
        _trainingResumeCommand = string.Empty;
        TrainingCheckpointsSummary = "Refresh checkpoints after a run writes them.";
        OnPropertyChanged(nameof(CanResumeTraining));

        if (result.Launch is { } launch && !string.IsNullOrWhiteSpace(launch.Command))
        {
            TrainingLaunchCommand = launch.Command;
            _trainingLaunchArgv = launch.Argv.ToArray();
            _trainingLaunchWorkingDirectory = string.IsNullOrWhiteSpace(result.OutputPath)
                ? string.Empty
                : (System.IO.Path.GetDirectoryName(result.OutputPath) ?? string.Empty);
            lines.Add("");
            lines.Add("Launch command (review before running):");
            lines.Add($"  {launch.Command}");
            if (launch.ResumeSupported)
            {
                lines.Add($"  resume: {launch.ResumeCommand}");
            }
            if (launch.Dependencies.Count > 0)
            {
                lines.Add($"  requires: {string.Join(", ", launch.Dependencies)}");
            }
        }
        else
        {
            TrainingLaunchCommand = string.Empty;
            _trainingLaunchArgv = [];
            _trainingLaunchWorkingDirectory = string.Empty;
        }

        OnPropertyChanged(nameof(CanLaunchTraining));

        if (result.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(result.Warnings.Select(warning => $"- {warning}"));
        }

        TrainingSummary = string.Join(Environment.NewLine, lines);
        TrainingConfigPreview = result.ConfigText;
    }

    public void SetTrainingConfigError(string message)
    {
        TrainingSummary = $"Training config could not be generated.{Environment.NewLine}{message}";
        TrainingConfigPreview = "No training config was generated.";
        ReportError(message);
    }

    public void ApplyTrainingCompatibility(TrainingCompatibilityResult result)
    {
        if (result.Compatible)
        {
            TrainingSummary =
                $"Compatible: {result.Schema} / {result.Format} → {result.Target}. "
                + "No compatibility warnings — safe to generate.";
            return;
        }

        TrainingSummary = string.Join(
            Environment.NewLine,
            new[]
            {
                $"Compatibility warnings for {result.Schema} / {result.Format} → {result.Target}:",
            }.Concat(result.Warnings.Select(warning => $"- {warning}"))
        );
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

    private void ApplyQualityHistory(IReadOnlyList<QualityHistoryEntry> history)
    {
        ApplyDebtTrend(history);

        if (history.Count == 0)
        {
            QualityHistorySummary = "No quality history has been recorded yet.";
            return;
        }

        var lines = new List<string> { "Recent quality history:" };
        lines.AddRange(history.Take(5).Select(entry => $"- {entry.DisplayName}"));

        if (history.Count >= 2)
        {
            var latest = history[0];
            var previous = history[1];
            var delta = latest.IssueCount - previous.IssueCount;
            var trend = delta switch
            {
                < 0 => $"Issues improved by {Math.Abs(delta)} since previous run.",
                > 0 => $"Issues increased by {delta} since previous run.",
                _ => "Issues unchanged since previous run.",
            };
            lines.Add(trend);
        }

        QualityHistorySummary = string.Join(Environment.NewLine, lines);
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

    private void SetSyntheticPatternIssues(IReadOnlyList<SyntheticPatternIssue> issues)
    {
        var selected = SelectedSyntheticPatternIssue;
        SyntheticPatternIssues.Clear();
        foreach (var issue in issues)
        {
            SyntheticPatternIssues.Add(issue);
        }

        SelectedSyntheticPatternIssue = SyntheticPatternIssues
            .FirstOrDefault(issue => IsSameSyntheticIssue(issue, selected))
            ?? SyntheticPatternIssues.FirstOrDefault();

        if (SyntheticPatternIssues.Count == 0)
        {
            QualityTriageSummary = "No synthetic quality issues found.";
        }
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






















    private static string FormatSyntheticPatternIssue(SyntheticPatternIssue issue)
    {
        var severity = string.IsNullOrWhiteSpace(issue.Severity)
            ? "unknown"
            : issue.Severity;
        var message = string.IsNullOrWhiteSpace(issue.Message)
            ? issue.Kind
            : issue.Message;
        var suggestion = string.IsNullOrWhiteSpace(issue.Suggestion)
            ? "Review and rewrite affected rows before export."
            : issue.Suggestion;
        return $"- [{severity}] {message} Fix: {suggestion}";
    }

    private static string FormatSyntheticTriageSummary(SyntheticPatternIssue issue)
    {
        var rows = issue.RowNumbers.Count == 0
            ? "unknown"
            : string.Join(", ", issue.RowNumbers.Take(8));
        return string.Join(
            Environment.NewLine,
            [
                $"Severity: {(string.IsNullOrWhiteSpace(issue.Severity) ? "unknown" : issue.Severity)}",
                $"Kind: {(string.IsNullOrWhiteSpace(issue.Kind) ? "synthetic_pattern" : issue.Kind)}",
                $"Rows: {rows}",
                issue.Message,
                $"Repair: {issue.Suggestion}",
            ]
        );
    }

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

    private static bool IsSameSyntheticIssue(
        SyntheticPatternIssue issue,
        SyntheticPatternIssue? other
    )
    {
        if (other is null)
        {
            return false;
        }

        return string.Equals(issue.Kind, other.Kind, StringComparison.Ordinal)
            && string.Equals(issue.Message, other.Message, StringComparison.Ordinal)
            && issue.RowNumbers.SequenceEqual(other.RowNumbers);
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
