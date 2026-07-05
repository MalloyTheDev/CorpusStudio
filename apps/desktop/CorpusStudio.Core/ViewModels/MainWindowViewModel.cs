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
    private string _preferencePromptText = "Select a saved preference example to inspect the prompt.";
    private string _preferenceChosenText = "Chosen response appears here.";
    private string _preferenceRejectedText = "Rejected response appears here.";
    private string _preferenceReasonText = "Reason or preference notes appear here.";
    private string _preferenceRankingSummary =
        "Preference ranking appears after saved preference pairs are loaded.";
    private string _preferenceContrastFilter = "All";
    private string _preferenceExportFormat = "dpo";
    private string _preferenceReviewSummary =
        "Preference pair review is available for preference projects.";
    private string _evaluationBackend = "ollama";
    private string _evaluationModel = "qwen2.5-coder:7b";
    private string _evaluationBaseUrl = "http://localhost:11434";
    private string _evaluationLimit = "10";
    private string _evaluationScoreThreshold = "70";
    private string _evaluationTimeoutSeconds = "120";
    private string _evaluationSummary =
        "Run a local model against this project's saved examples.";
    private string _benchmarkModelsInput = string.Empty;
    private string _benchmarkSummary =
        "Enter one model per line, then benchmark them against this project's examples.";
    private string _evaluationReportJson = "Evaluation reports appear here after a run.";
    private string _selectedEvaluationExampleDetail =
        "Per-example evaluation results appear here after a run or report reload.";
    private string _evaluationResultsSummary =
        "Evaluation example review queue appears after a run or report reload.";
    private string _evaluationResultFilter = "All";
    private string _evaluationTagFilter = "All";
    private string _evaluationFailureReasonFilter = "All";
    private string _evaluationScoreBandFilter = "All";
    private string _evaluationFailureFilterName = "Failure View";
    private string _evaluationFailureFilterSummary =
        "Save the active status, tag, failure-reason, and score-band filters as a named view.";
    private string _evaluationManualScore = string.Empty;
    private string _evaluationManualNotes = string.Empty;
    private string _evaluationReviewSummary = "Select an evaluation result to add a manual score or note.";
    private string _evaluationModelListSummary =
        "Refresh models to load running Ollama or OpenAI-compatible models.";
    private string _evaluationComparisonSummary =
        "Select two saved evaluation reports to compare score, failure, tag, and row-level changes.";
    private string _aiAssistBackend = "ollama";
    private string _aiAssistModel = "qwen2.5-coder:7b";
    private string _aiAssistBaseUrl = "http://localhost:11434";
    private string _aiAssistAction = "review";
    private string _aiAssistTimeoutSeconds = "120";
    private string _aiAssistInstruction =
        "Review the current draft and suggest safer tags or a stronger output.";
    private string _aiAssistSummary =
        "Run AI Assist on the current draft. Suggestions require human review.";
    private string _aiAssistReviewText = "AI Assist review output appears here.";
    private string _aiAssistSourceDraftText =
        "Source draft appears here after selecting an AI Assist review.";
    private string _aiAssistSuggestedJsonlText =
        "Suggested JSONL appears here after selecting an AI Assist review.";
    private string _aiAssistDiffSummary =
        "Select a queued AI Assist review to compare source and suggestion.";
    private string _aiAssistSuggestionJsonl = string.Empty;
    // The candidate gate for the current fresh run (before a queue item is selected).
    // When a queue item is selected, its persisted gate takes precedence (see
    // ActiveAiAssistCandidateGate). Never means "approved".
    private GateReport? _aiAssistCandidateGate;
    private string _aiAssistCandidateGateStatus = "—";
    private string _aiAssistCandidateGateColor = "#64748B";  // neutral gray until a run/selection sets it
    private string _aiAssistQueueSummary = "AI Assist review queue appears after suggestions are generated.";
    private string _aiAssistQueueFilter = "All";
    private string _aiAssistQueueSearch = string.Empty;
    private string _aiAssistQueueSort = "Newest";
    private string _aiAssistQueueViewName = "Review View";
    private string _aiAssistRewriteBatchSummary =
        "Prepared rewrite batches appear here after synthetic batch triage.";
    private string _reviewedFixSummary =
        "Edited failed rows appear here so you can track which fixes were re-tested.";
    private string _aiAssistModelListSummary =
        "Refresh models to load running Ollama or OpenAI-compatible models.";
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
    private string _selectedImportQuarantineDetail =
        "Rejected import rows appear here after a mixed import.";
    private string _projectIndexSummary = "Projects list from local files. Rebuild the index to list from SQLite.";
    private bool _isBusy;
    private string _busyStatus = "Working...";
    private bool _hasError;
    private string _errorMessage = string.Empty;
    private string _labSettingsSummary = "Lab backend settings can be saved per project.";
    private DatasetProjectListItem? _selectedProject;
    private SavedExampleItem? _selectedExample;
    private PreferenceReviewItem? _selectedPreferenceReviewItem;
    private ValidationIssueNavigationItem? _selectedValidationIssue;
    private SyntheticPatternIssue? _selectedSyntheticPatternIssue;
    private ImportQuarantineItem? _selectedImportQuarantineItem;
    private EvaluationReportHistoryItem? _selectedEvaluationReportHistoryItem;
    private EvaluationReportHistoryItem? _secondaryEvaluationReportHistoryItem;
    private EvaluationExampleResult? _selectedEvaluationExampleResult;
    private AiAssistReviewQueueItem? _selectedAiAssistReviewQueueItem;
    private AiAssistQueueView? _selectedAiAssistQueueView;
    private AiAssistRewriteBatch? _selectedAiAssistRewriteBatch;
    private AiAssistRewriteBatch? _lastPreparedAiAssistRewriteBatch;
    private ReviewedFixRecord? _selectedReviewedFix;
    private ReviewedFixRecord? _lastPreparedEvaluationFix;
    private EvaluationFailureFilter? _selectedEvaluationFailureFilter;
    private readonly List<EvaluationExampleResult> _allEvaluationResults = [];
    private readonly List<PreferenceReviewItem> _allPreferenceReviewItems = [];
    private readonly List<AiAssistReviewQueueItem> _allAiAssistReviewQueue = [];
    private string _selectedExampleJson = "Saved examples appear here after a project is selected.";

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

    /// <summary>Design-time / test constructor.</summary>
    public MainWindowViewModel()
        : this(
            new DebtViewModel(), new ArenaViewModel(), new SettingsViewModel(),
            new VersionsViewModel(), new ArtifactsViewModel(), new SuitesViewModel(),
            new SplitsViewModel())
    {
    }

    public MainWindowViewModel(
        IDebtViewModel debt, IArenaViewModel arena, ISettingsViewModel settings,
        IVersionsViewModel versions, IArtifactsViewModel artifacts, ISuitesViewModel suites,
        ISplitsViewModel splits)
    {
        Debt = debt;
        Arena = arena;
        Settings = settings;
        Versions = versions;
        Artifacts = artifacts;
        Suites = suites;
        Splits = splits;
        // A split failure surfaces in the shell's shared error banner; the tab keeps no shell
        // reference, so it raises ErrorReported and the shell forwards it to ReportError.
        Splits.ErrorReported += ReportError;
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

    public ObservableCollection<SavedExampleItem> Examples { get; } = [];

    public ObservableCollection<ValidationIssueNavigationItem> ValidationIssues { get; } = [];

    public ObservableCollection<SyntheticPatternIssue> SyntheticPatternIssues { get; } = [];

    public ObservableCollection<ImportQuarantineItem> ImportQuarantineItems { get; } = [];

    public ObservableCollection<EvaluationReportHistoryItem> EvaluationReportHistory { get; } = [];

    public ObservableCollection<EvaluationExampleResult> EvaluationResults { get; } = [];

    public ObservableCollection<string> EvaluationAvailableModels { get; } = [];

    public ObservableCollection<PreferenceReviewItem> PreferenceReviewItems { get; } = [];

    public ObservableCollection<string> PreferenceContrastFilterOptions { get; } =
    [
        "All",
        "Weak",
        "Moderate",
        "Strong",
    ];

    public ObservableCollection<string> PreferenceExportFormatOptions { get; } =
    [
        "dpo",
        "kto",
        "reward",
    ];

    public ObservableCollection<string> EvaluationResultFilterOptions { get; } =
    [
        "All",
        "Failed",
        "Passed",
        "Manually Scored",
    ];

    public ObservableCollection<string> EvaluationTagFilterOptions { get; } = ["All"];

    public ObservableCollection<string> EvaluationFailureReasonFilterOptions { get; } = ["All"];

    public ObservableCollection<string> EvaluationScoreBandFilterOptions { get; } =
    [
        "All",
        "0-49",
        "50-69",
        "70-84",
        "85-100",
    ];

    public ObservableCollection<EvaluationFailureFilter> EvaluationFailureFilters { get; } = [];

    public ObservableCollection<AiAssistReviewQueueItem> AiAssistReviewQueue { get; } = [];

    public ObservableCollection<AiAssistQueueView> AiAssistQueueViews { get; } = [];

    public ObservableCollection<AiAssistRewriteBatch> AiAssistRewriteBatches { get; } = [];

    public ObservableCollection<ReviewedFixRecord> ReviewedFixes { get; } = [];

    public ObservableCollection<string> AiAssistAvailableModels { get; } = [];

    public ObservableCollection<string> AiAssistQueueFilterOptions { get; } =
    [
        "All",
        "Pending",
        "Accepted",
        "Rejected",
    ];

    public ObservableCollection<string> AiAssistQueueSortOptions { get; } =
    [
        "Newest",
        "Oldest",
        "State",
        "Model",
        "Action",
    ];

    public ObservableCollection<string> AiAssistActionPresets { get; } =
    [
        "review",
        "suggest-tags",
        "rewrite-output",
        "draft-example",
    ];

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

    public SavedExampleItem? SelectedExample
    {
        get => _selectedExample;
        set
        {
            if (SetField(ref _selectedExample, value))
            {
                SelectedExampleJson = value?.Json ?? "Select a saved example to inspect its JSON.";
            }
        }
    }

    public PreferenceReviewItem? SelectedPreferenceReviewItem
    {
        get => _selectedPreferenceReviewItem;
        set
        {
            if (SetField(ref _selectedPreferenceReviewItem, value))
            {
                ApplyPreferenceReviewItem(value);
            }
        }
    }

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

    public string SelectedExampleJson
    {
        get => _selectedExampleJson;
        private set => SetField(ref _selectedExampleJson, value);
    }

    private const string InitialDraftTemplate =
        "{\n  \"instruction\": \"Explain what a variable is.\",\n  \"input\": \"\",\n  \"output\": \"A variable stores a value.\"\n}";

    private string _draftText = InitialDraftTemplate;

    // The last programmatically loaded/saved draft. The buffer is "dirty" (unsaved user edits)
    // when DraftText diverges from this. Set via LoadDraft/MarkDraftClean so that loading a
    // known draft (template, saved example, retried row) is not reported as unsaved work.
    private string _draftBaseline = InitialDraftTemplate;

    public string DraftText
    {
        get => _draftText;
        set
        {
            if (SetField(ref _draftText, value))
            {
                OnPropertyChanged(nameof(IsDraftDirty));
                OnPropertyChanged(nameof(HasUnsavedWork));
            }
        }
    }

    /// <summary>True when the editor buffer has unsaved user edits (differs from the last
    /// loaded/saved draft).</summary>
    public bool IsDraftDirty => !string.Equals(_draftText, _draftBaseline, StringComparison.Ordinal);

    /// <summary>Unsaved work that a project switch or app close would silently discard: an
    /// edited draft, or a dirty open document in the workspace explorer.</summary>
    public bool HasUnsavedWork => IsDraftDirty || Explorer.HasDirtyDocuments;

    /// <summary>Load a known draft (template, saved example, retried row) as the clean baseline,
    /// so it is not reported as unsaved until the user edits it.</summary>
    public void LoadDraft(string text)
    {
        _draftBaseline = text ?? string.Empty;
        DraftText = text ?? string.Empty; // the setter re-raises IsDraftDirty/HasUnsavedWork
    }

    /// <summary>Mark the current draft as saved — its content becomes the clean baseline.</summary>
    public void MarkDraftClean()
    {
        _draftBaseline = _draftText;
        OnPropertyChanged(nameof(IsDraftDirty));
        OnPropertyChanged(nameof(HasUnsavedWork));
    }

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

    public string PreferencePromptText
    {
        get => _preferencePromptText;
        private set => SetField(ref _preferencePromptText, value);
    }

    public string PreferenceChosenText
    {
        get => _preferenceChosenText;
        private set => SetField(ref _preferenceChosenText, value);
    }

    public string PreferenceRejectedText
    {
        get => _preferenceRejectedText;
        private set => SetField(ref _preferenceRejectedText, value);
    }

    public string PreferenceReasonText
    {
        get => _preferenceReasonText;
        private set => SetField(ref _preferenceReasonText, value);
    }

    public string PreferenceRankingSummary
    {
        get => _preferenceRankingSummary;
        private set => SetField(ref _preferenceRankingSummary, value);
    }

    public string PreferenceContrastFilter
    {
        get => _preferenceContrastFilter;
        set
        {
            if (SetField(ref _preferenceContrastFilter, value))
            {
                RebuildPreferenceReviewItems();
            }
        }
    }

    public string PreferenceReviewSummary
    {
        get => _preferenceReviewSummary;
        private set => SetField(ref _preferenceReviewSummary, value);
    }


    // Splits tab state + logic extracted to SplitsViewModel in Phase 2. The shell wires the child's
    // ErrorReported to ReportError, pushes loaded settings, and forwards Reset() on project switch.
    // Bindings use Splits.*, code-behind ViewModel.Splits.*.

    public string EvaluationBackend
    {
        get => _evaluationBackend;
        set => SetField(ref _evaluationBackend, value);
    }

    public string EvaluationModel
    {
        get => _evaluationModel;
        set => SetField(ref _evaluationModel, value);
    }

    public string EvaluationBaseUrl
    {
        get => _evaluationBaseUrl;
        set => SetField(ref _evaluationBaseUrl, value);
    }

    public string EvaluationLimit
    {
        get => _evaluationLimit;
        set => SetField(ref _evaluationLimit, value);
    }

    public string EvaluationScoreThreshold
    {
        get => _evaluationScoreThreshold;
        set => SetField(ref _evaluationScoreThreshold, value);
    }

    public string EvaluationTimeoutSeconds
    {
        get => _evaluationTimeoutSeconds;
        set => SetField(ref _evaluationTimeoutSeconds, value);
    }

    public string EvaluationSummary
    {
        get => _evaluationSummary;
        private set => SetField(ref _evaluationSummary, value);
    }

    public string EvaluationReportJson
    {
        get => _evaluationReportJson;
        private set => SetField(ref _evaluationReportJson, value);
    }

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

    public EvaluationExampleResult? SelectedEvaluationExampleResult
    {
        get => _selectedEvaluationExampleResult;
        set
        {
            if (SetField(ref _selectedEvaluationExampleResult, value))
            {
                ApplySelectedEvaluationExample(value);
            }
        }
    }

    public string SelectedEvaluationExampleDetail
    {
        get => _selectedEvaluationExampleDetail;
        private set => SetField(ref _selectedEvaluationExampleDetail, value);
    }

    public string EvaluationResultsSummary
    {
        get => _evaluationResultsSummary;
        private set => SetField(ref _evaluationResultsSummary, value);
    }

    public string EvaluationResultFilter
    {
        get => _evaluationResultFilter;
        set
        {
            if (SetField(ref _evaluationResultFilter, value))
            {
                RebuildEvaluationResults();
            }
        }
    }

    public string EvaluationTagFilter
    {
        get => _evaluationTagFilter;
        set
        {
            if (SetField(ref _evaluationTagFilter, string.IsNullOrWhiteSpace(value) ? "All" : value))
            {
                RebuildEvaluationResults();
            }
        }
    }

    public string EvaluationFailureReasonFilter
    {
        get => _evaluationFailureReasonFilter;
        set
        {
            if (SetField(ref _evaluationFailureReasonFilter, string.IsNullOrWhiteSpace(value) ? "All" : value))
            {
                RebuildEvaluationResults();
            }
        }
    }

    public string EvaluationScoreBandFilter
    {
        get => _evaluationScoreBandFilter;
        set
        {
            if (SetField(ref _evaluationScoreBandFilter, string.IsNullOrWhiteSpace(value) ? "All" : value))
            {
                RebuildEvaluationResults();
            }
        }
    }

    public string EvaluationFailureFilterName
    {
        get => _evaluationFailureFilterName;
        set => SetField(ref _evaluationFailureFilterName, value);
    }

    public EvaluationFailureFilter? SelectedEvaluationFailureFilter
    {
        get => _selectedEvaluationFailureFilter;
        set => SetField(ref _selectedEvaluationFailureFilter, value);
    }

    public string EvaluationFailureFilterSummary
    {
        get => _evaluationFailureFilterSummary;
        private set => SetField(ref _evaluationFailureFilterSummary, value);
    }

    public string EvaluationManualScore
    {
        get => _evaluationManualScore;
        set => SetField(ref _evaluationManualScore, value);
    }

    public string EvaluationManualNotes
    {
        get => _evaluationManualNotes;
        set => SetField(ref _evaluationManualNotes, value);
    }

    public string EvaluationReviewSummary
    {
        get => _evaluationReviewSummary;
        private set => SetField(ref _evaluationReviewSummary, value);
    }

    public string EvaluationModelListSummary
    {
        get => _evaluationModelListSummary;
        private set => SetField(ref _evaluationModelListSummary, value);
    }

    public string AiAssistBackend
    {
        get => _aiAssistBackend;
        set => SetField(ref _aiAssistBackend, value);
    }

    public string AiAssistModel
    {
        get => _aiAssistModel;
        set => SetField(ref _aiAssistModel, value);
    }

    public string AiAssistBaseUrl
    {
        get => _aiAssistBaseUrl;
        set => SetField(ref _aiAssistBaseUrl, value);
    }

    public string AiAssistAction
    {
        get => _aiAssistAction;
        set => SetField(ref _aiAssistAction, value);
    }

    public string AiAssistTimeoutSeconds
    {
        get => _aiAssistTimeoutSeconds;
        set => SetField(ref _aiAssistTimeoutSeconds, value);
    }

    public string AiAssistInstruction
    {
        get => _aiAssistInstruction;
        set => SetField(ref _aiAssistInstruction, value);
    }

    public string AiAssistSummary
    {
        get => _aiAssistSummary;
        private set => SetField(ref _aiAssistSummary, value);
    }

    public string AiAssistReviewText
    {
        get => _aiAssistReviewText;
        private set => SetField(ref _aiAssistReviewText, value);
    }

    public string AiAssistSourceDraftText
    {
        get => _aiAssistSourceDraftText;
        private set => SetField(ref _aiAssistSourceDraftText, value);
    }

    public string AiAssistSuggestedJsonlText
    {
        get => _aiAssistSuggestedJsonlText;
        private set => SetField(ref _aiAssistSuggestedJsonlText, value);
    }

    public string AiAssistDiffSummary
    {
        get => _aiAssistDiffSummary;
        private set => SetField(ref _aiAssistDiffSummary, value);
    }

    /// <summary>Short candidate-gate status label for the AI Assist tab header
    /// (e.g. "PASS", "BLOCK", "not run", "n/a", "—"). Informational — never approval.</summary>
    public string AiAssistCandidateGateStatus
    {
        get => _aiAssistCandidateGateStatus;
        private set => SetField(ref _aiAssistCandidateGateStatus, value);
    }

    /// <summary>Foreground hex for the candidate-gate status label. Neutral gray for
    /// null/unknown/"not run"/"n/a" — never green (see GateReport.StatusColor).</summary>
    public string AiAssistCandidateGateColor
    {
        get => _aiAssistCandidateGateColor;
        private set => SetField(ref _aiAssistCandidateGateColor, value);
    }

    /// <summary>The gate that applies to the suggestion that would move to the draft:
    /// the selected queue item's persisted gate, else the current run's gate. Mirrors
    /// the same fallback MoveAiAssistSuggestionToDraft uses.</summary>
    private GateReport? ActiveAiAssistCandidateGate =>
        SelectedAiAssistReviewQueueItem?.CandidateGate ?? _aiAssistCandidateGate;

    /// <summary>True only when the active candidate gate BLOCKS. Drives the
    /// confirm-on-block prompt before the suggestion is moved to the draft. Pure —
    /// the gate only informs; the human still decides (never auto-rejected).</summary>
    public bool SelectedAiAssistCandidateGateBlocks =>
        string.Equals(ActiveAiAssistCandidateGate?.OverallStatus, "block", StringComparison.OrdinalIgnoreCase);

    public AiAssistReviewQueueItem? SelectedAiAssistReviewQueueItem
    {
        get => _selectedAiAssistReviewQueueItem;
        set
        {
            if (SetField(ref _selectedAiAssistReviewQueueItem, value))
            {
                if (value is null)
                {
                    ClearAiAssistComparison();
                    // No item selected: the header reverts to the current fresh-run gate
                    // (ActiveAiAssistCandidateGate falls back to it), so the block-confirm
                    // and label stay consistent with what would move to the draft.
                    ApplyCandidateGateStatus(
                        _aiAssistCandidateGate,
                        !string.IsNullOrWhiteSpace(_aiAssistSuggestionJsonl));
                }
                else
                {
                    ApplyAiAssistReviewQueueItem(value);
                }
            }
        }
    }

    public string AiAssistQueueSummary
    {
        get => _aiAssistQueueSummary;
        private set => SetField(ref _aiAssistQueueSummary, value);
    }

    public string AiAssistQueueFilter
    {
        get => _aiAssistQueueFilter;
        set
        {
            if (SetField(ref _aiAssistQueueFilter, value))
            {
                RebuildAiAssistReviewQueue();
            }
        }
    }

    public string AiAssistQueueSearch
    {
        get => _aiAssistQueueSearch;
        set
        {
            if (SetField(ref _aiAssistQueueSearch, value))
            {
                RebuildAiAssistReviewQueue();
            }
        }
    }

    public string AiAssistQueueSort
    {
        get => _aiAssistQueueSort;
        set
        {
            if (SetField(ref _aiAssistQueueSort, value))
            {
                RebuildAiAssistReviewQueue();
            }
        }
    }

    public string AiAssistQueueViewName
    {
        get => _aiAssistQueueViewName;
        set => SetField(ref _aiAssistQueueViewName, value);
    }

    public AiAssistQueueView? SelectedAiAssistQueueView
    {
        get => _selectedAiAssistQueueView;
        set => SetField(ref _selectedAiAssistQueueView, value);
    }

    public AiAssistRewriteBatch? SelectedAiAssistRewriteBatch
    {
        get => _selectedAiAssistRewriteBatch;
        set
        {
            if (SetField(ref _selectedAiAssistRewriteBatch, value) && value is not null)
            {
                AiAssistRewriteBatchSummary =
                    $"Selected rewrite batch for rows {FormatRowNumbers(value.RowNumbers)}.";
            }
        }
    }

    public string AiAssistRewriteBatchSummary
    {
        get => _aiAssistRewriteBatchSummary;
        private set => SetField(ref _aiAssistRewriteBatchSummary, value);
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

    public string AiAssistModelListSummary
    {
        get => _aiAssistModelListSummary;
        private set => SetField(ref _aiAssistModelListSummary, value);
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
            + BuildEvaluationReportComparison(after, _trainingBaselineReport);
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

    public ImportQuarantineItem? SelectedImportQuarantineItem
    {
        get => _selectedImportQuarantineItem;
        set
        {
            if (SetField(ref _selectedImportQuarantineItem, value))
            {
                SelectedImportQuarantineDetail = value?.DetailText
                    ?? "Select a rejected import row to inspect it.";
            }
        }
    }

    public EvaluationReportHistoryItem? SelectedEvaluationReportHistoryItem
    {
        get => _selectedEvaluationReportHistoryItem;
        set
        {
            if (SetField(ref _selectedEvaluationReportHistoryItem, value) && value is not null)
            {
                ApplyEvaluationReportHistoryItem(value);
            }
        }
    }

    public EvaluationReportHistoryItem? SecondaryEvaluationReportHistoryItem
    {
        get => _secondaryEvaluationReportHistoryItem;
        set => SetField(ref _secondaryEvaluationReportHistoryItem, value);
    }

    public string EvaluationComparisonSummary
    {
        get => _evaluationComparisonSummary;
        private set => SetField(ref _evaluationComparisonSummary, value);
    }

    public string SelectedImportQuarantineDetail
    {
        get => _selectedImportQuarantineDetail;
        private set => SetField(ref _selectedImportQuarantineDetail, value);
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
        LoadDraft(BuildDraftTemplate(schemaId));
    }

    public void SelectProject(DatasetProjectListItem project, string? schemaName = null)
    {
        SelectedProject = project;
        ActiveProjectTitle = project.Name;
        ActiveProjectPath = project.ProjectPath;
        ActiveSchemaId = project.SchemaId;
        ActiveSchemaDescription = $"{schemaName ?? project.SchemaId} project. Ready for examples.";
        ApplyAiAssistActionPresets(project.SchemaId);
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
        EvaluationSummary = "Run a local model against this project's saved examples.";
        EvaluationReportJson = "Evaluation reports appear here after a run.";
        EvaluationReportHistory.Clear();
        ClearEvaluationResults();
        ResetEvaluationFailureFilters();
        SelectedEvaluationReportHistoryItem = null;
        AiAssistSummary = "Run AI Assist on the current draft. Suggestions require human review.";
        AiAssistReviewText = "AI Assist review output appears here.";
        AiAssistQueueSummary = "AI Assist review queue appears after suggestions are generated.";
        AiAssistQueueViewName = "Review View";
        AiAssistQueueViews.Clear();
        SelectedAiAssistQueueView = null;
        AiAssistRewriteBatches.Clear();
        SelectedAiAssistRewriteBatch = null;
        _lastPreparedAiAssistRewriteBatch = null;
        AiAssistRewriteBatchSummary =
            "Prepared rewrite batches appear here after synthetic batch triage.";
        ReviewedFixes.Clear();
        SelectedReviewedFix = null;
        _lastPreparedEvaluationFix = null;
        ReviewedFixSummary =
            "Edited failed rows appear here so you can track which fixes were re-tested.";
        _aiAssistSuggestionJsonl = string.Empty;
        ClearAiAssistComparison();
        _allAiAssistReviewQueue.Clear();
        AiAssistReviewQueue.Clear();
        SelectedAiAssistReviewQueueItem = null;
        // Clear the candidate-gate verdict last so a prior project's gate can't linger in
        // the header or fire a spurious block-confirm in the new (gate-less) project.
        ResetCandidateGateState();
        // Likewise clear the Problems panel so a previous project's gate findings are never
        // shown against the newly selected (not-yet-gated) project.
        ResetProblems();
        TrainingFormat = project.SchemaId;
        TrainingSummary = "Generate a training config after validation, splits, and evaluation checks.";
        TrainingConfigPreview = "Training config preview appears here.";
        Examples.Clear();
        ImportQuarantineItems.Clear();
        SelectedImportQuarantineItem = null;
        _pendingRetryItem = null;
        SelectedExample = null;
        _allPreferenceReviewItems.Clear();
        PreferenceReviewItems.Clear();
        SelectedPreferenceReviewItem = null;
        PreferenceContrastFilter = "All";
        ClearPreferenceReview();
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
        LoadDraft(BuildDraftTemplate(project.SchemaId));
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
                Backend = EvaluationBackend.Trim(),
                Model = EvaluationModel.Trim(),
                BaseUrl = EvaluationBaseUrl.Trim(),
                TimeoutSeconds = ParsePositiveIntOrDefault(EvaluationTimeoutSeconds, 120),
            },
            AiAssist = new ModelBackendSettings
            {
                Backend = AiAssistBackend.Trim(),
                Model = AiAssistModel.Trim(),
                BaseUrl = AiAssistBaseUrl.Trim(),
                TimeoutSeconds = ParsePositiveIntOrDefault(AiAssistTimeoutSeconds, 120),
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

    public void SetExamples(IEnumerable<SavedExampleItem> examples)
    {
        Examples.Clear();
        foreach (var example in examples)
        {
            Examples.Add(example);
        }

        SelectedExample = Examples.FirstOrDefault();
        SelectedExampleJson = SelectedExample?.Json
            ?? "No saved examples yet. Save a valid draft from Writing Studio.";
        SetPreferenceReviewItems(Examples);
        QualitySummary = Examples.Count == 0
            ? "No saved examples yet. Quality checks will run after examples are added."
            : $"{Examples.Count} saved example(s). Run quality checks to inspect duplicates and empty rows.";
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

        var example = Examples.FirstOrDefault(item => item.RowNumber == rowNumber);
        if (example is null)
        {
            QualityTriageSummary = $"Affected row {rowNumber} is not loaded in the Examples list.";
            return false;
        }

        SelectedExample = example;
        LoadDraft(example.Json);
        if (AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssistAction = "rewrite-output";
        }

        AiAssistInstruction = BuildSyntheticRewriteInstruction(SelectedSyntheticPatternIssue, rowNumber);
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
            .Select(rowNumber => Examples.FirstOrDefault(example => example.RowNumber == rowNumber))
            .Where(example => example is not null)
            .Cast<SavedExampleItem>()
            .ToList();
        if (affectedRows.Count == 0)
        {
            QualityTriageSummary = "Affected synthetic rows are not loaded in the Examples list.";
            return false;
        }

        LoadDraft(BuildJsonArrayDraft(affectedRows.Select(row => row.Json)));
        if (AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssistAction = "rewrite-output";
        }

        AiAssistInstruction = BuildSyntheticBatchRewriteInstruction(SyntheticPatternIssues, rowNumbers);
        _lastPreparedAiAssistRewriteBatch = new AiAssistRewriteBatch
        {
            SchemaId = ActiveSchemaId,
            Action = "rewrite-output",
            RowNumbers = rowNumbers,
            IssueCount = SyntheticPatternIssues.Count,
            IssueSummary = BuildSyntheticIssueSummary(SyntheticPatternIssues),
            SourceDraft = DraftText,
            Instruction = AiAssistInstruction,
        };
        QualityTriageSummary =
            $"Prepared {affectedRows.Count} affected row(s) from {SyntheticPatternIssues.Count} synthetic issue(s) for batch rewrite.";
        return true;
    }

    public bool PrepareEvaluationFailureReview()
    {
        if (SelectedEvaluationExampleResult is null)
        {
            EvaluationReviewSummary = "Select an evaluation example before preparing failure triage.";
            return false;
        }

        if (SelectedEvaluationExampleResult.Passed)
        {
            EvaluationReviewSummary = "Selected evaluation example passed. Choose a failed example for failure triage.";
            return false;
        }

        if (ActiveSchemaId is not ("instruction" or "chat"))
        {
            EvaluationReviewSummary = "Evaluation failure triage supports instruction and chat projects.";
            return false;
        }

        LoadDraft(BuildEvaluationFailureDraft(SelectedEvaluationExampleResult, ActiveSchemaId));
        if (AiAssistActionPresets.Contains("rewrite-output"))
        {
            AiAssistAction = "rewrite-output";
        }

        AiAssistInstruction = BuildEvaluationFailureInstruction(SelectedEvaluationExampleResult);
        EvaluationReviewSummary =
            $"Prepared failed evaluation example {SelectedEvaluationExampleResult.ExampleId} for AI Assist triage.";
        return true;
    }

    public bool PrepareEvaluationFailureEdit()
    {
        if (SelectedEvaluationExampleResult is null)
        {
            EvaluationReviewSummary = "Select an evaluation example before editing a failed row.";
            return false;
        }

        if (SelectedEvaluationExampleResult.Passed)
        {
            EvaluationReviewSummary = "Selected evaluation example passed. Choose a failed example to edit.";
            return false;
        }

        if (!TryParseEvaluationRowNumber(SelectedEvaluationExampleResult.ExampleId, out var rowNumber))
        {
            EvaluationReviewSummary =
                $"Evaluation example '{SelectedEvaluationExampleResult.ExampleId}' is not linked to a saved row.";
            return false;
        }

        var example = Examples.FirstOrDefault(item => item.RowNumber == rowNumber);
        if (example is null)
        {
            EvaluationReviewSummary =
                $"Saved row {rowNumber} is not loaded in the Examples list.";
            return false;
        }

        var failure = SelectedEvaluationExampleResult;
        _lastPreparedEvaluationFix = new ReviewedFixRecord
        {
            ExampleId = failure.ExampleId,
            RowNumber = rowNumber,
            SchemaId = ActiveSchemaId,
            OriginalScore = failure.Score,
            FailureReason = FailureReason(failure),
            SourceReport = SelectedEvaluationReportHistoryItem?.DisplayName ?? "current evaluation run",
        };

        SelectedExample = example;
        LoadDraft(example.Json);
        ValidationSummary =
            $"Loaded evaluation failure row {rowNumber}. Validate before saving reviewed edits.";
        EvaluationReviewSummary =
            $"Loaded failed row {rowNumber} into Writing Studio. Edit, validate, save, then rerun evaluation.";
        return true;
    }

    public bool PreparePreferenceJudgeReview()
    {
        if (SelectedPreferenceReviewItem is null)
        {
            PreferenceReviewSummary = "Select a saved preference pair before preparing AI Assist review.";
            return false;
        }

        LoadDraft(SelectedPreferenceReviewItem.Json);
        if (AiAssistActionPresets.Contains("judge-preference-strength"))
        {
            AiAssistAction = "judge-preference-strength";
        }

        AiAssistInstruction = BuildPreferenceJudgeInstruction(
            SelectedPreferenceReviewItem.RowNumber,
            SelectedPreferenceReviewItem.Prompt,
            SelectedPreferenceReviewItem.Chosen,
            SelectedPreferenceReviewItem.Rejected
        );
        PreferenceReviewSummary =
            $"Prepared Example {SelectedPreferenceReviewItem.RowNumber} for AI Assist preference-strength review.";
        return true;
    }

    public bool PreparePreferenceBatchJudgeReview()
    {
        if (ActiveSchemaId != "preference")
        {
            PreferenceReviewSummary = "Preference batch review is available for preference projects.";
            return false;
        }

        var items = PreferenceReviewItems.ToList();
        if (items.Count == 0)
        {
            PreferenceReviewSummary = "No preference pairs match the current ranking filter.";
            return false;
        }

        LoadDraft(BuildJsonArrayDraft(items.Select(item => item.Json)));
        if (AiAssistActionPresets.Contains("judge-preference-strength"))
        {
            AiAssistAction = "judge-preference-strength";
        }

        AiAssistInstruction = BuildPreferenceBatchJudgeInstruction(items);
        PreferenceReviewSummary =
            $"Prepared {items.Count} visible preference pair(s) for AI Assist batch judging.";
        return true;
    }

    public IReadOnlyList<PreferenceReviewItem> GetVisiblePreferenceReviewItems()
    {
        return PreferenceReviewItems.ToList();
    }

    public void ApplyPreferenceRankingExport(string outputPath, int itemCount)
    {
        PreferenceReviewSummary =
            $"Exported {itemCount} visible preference ranking item(s) for DPO review: {outputPath}";
    }

    public string PreferenceExportFormat
    {
        get => _preferenceExportFormat;
        set => SetField(ref _preferenceExportFormat, value);
    }

    public void ApplyPreferenceTrainingExport(PreferenceExportResult result)
    {
        var lines = new List<string>
        {
            $"Exported {result.OutputRows} row(s) as {result.Format}: {result.OutputPath}",
        };

        if (result.DroppedDegenerate > 0)
        {
            lines.Add($"Dropped {result.DroppedDegenerate} degenerate pair(s).");
        }

        if (result.Warnings.Count > 0)
        {
            lines.Add("Warnings:");
            lines.AddRange(result.Warnings.Select(warning => $"- {warning}"));
        }

        PreferenceReviewSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetPreferenceRankingExportError(string message)
    {
        PreferenceReviewSummary = $"Preference ranking export failed.{Environment.NewLine}{message}";
    }

    public void SetEvaluationInProgress()
    {
        EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                "Running evaluation...",
                $"Backend: {EvaluationBackend}",
                $"Model: {EvaluationModel}",
            ]
        );
        EvaluationReportJson = "Waiting for local model response.";
    }

    public void SetEvaluationPreflightInProgress()
    {
        EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                "Checking evaluation backend before run...",
                $"Backend: {EvaluationBackend}",
                $"Model: {EvaluationModel}",
            ]
        );
        EvaluationReportJson = "No evaluation report has been produced yet.";
    }

    public void SetEvaluationRegressionRerunPreflightInProgress(EvaluationRunSettings settings)
    {
        ApplyEvaluationRunSettings(settings);
        EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                "Checking saved regression run settings...",
                $"Backend: {EvaluationBackend}",
                $"Model: {EvaluationModel}",
                $"Threshold: {EvaluationScoreThreshold}",
            ]
        );
        EvaluationReportJson = "No regression rerun report has been produced yet.";
    }

    public void SetEvaluationRegressionRerunInProgress(EvaluationRunSettings settings)
    {
        ApplyEvaluationRunSettings(settings);
        EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                "Rerunning saved evaluation configuration...",
                $"Backend: {EvaluationBackend}",
                $"Model: {EvaluationModel}",
                $"Threshold: {EvaluationScoreThreshold}",
            ]
        );
        EvaluationReportJson = "Waiting for local model response.";
    }

    public void ApplyEvaluationRunResult(EvaluationRunResult result)
    {
        var weakTags = result.Report.WeakTags.Count == 0
            ? "none"
            : string.Join(", ", result.Report.WeakTags);
        EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                $"Dataset: {result.Report.Dataset}",
                $"Model: {result.Report.Model}",
                $"Scoring: {DescribeEvaluationMetric(result.Report.Metric)}",
                $"Examples tested: {result.Report.ExamplesTested}",
                $"Average score: {result.Report.AverageScore:0.##}",
                $"Failed examples: {result.Report.FailedExamples}",
                $"Manual scores: {FormatManualScoreSummary(result.Report)}",
                $"Weak tags: {weakTags}",
                $"Tag summary: {FormatTagSummary(result.Report)}",
                $"Failure reasons: {FormatFailureReasonSummary(result.Report)}",
                $"Score bands: {FormatScoreBandSummary(result.Report)}",
                $"Report: {result.ReportPath}",
            ]
        );
        EvaluationReportJson = result.ReportJson;
        SetEvaluationResults(result.Report.Results);
    }

    public void SetEvaluationReportHistory(IEnumerable<EvaluationReportHistoryItem> history)
    {
        EvaluationReportHistory.Clear();
        SelectedEvaluationReportHistoryItem = null;
        SecondaryEvaluationReportHistoryItem = null;
        foreach (var item in history)
        {
            EvaluationReportHistory.Add(item);
        }

        SelectedEvaluationReportHistoryItem = EvaluationReportHistory.FirstOrDefault();
        SecondaryEvaluationReportHistoryItem = EvaluationReportHistory.Skip(1).FirstOrDefault();
        EvaluationComparisonSummary = EvaluationReportHistory.Count < 2
            ? "At least two saved evaluation reports are needed for comparison."
            : "Select a saved report and a comparison report, then click Compare Reports.";
    }

    public void ApplyEvaluationReportHistoryItem(EvaluationReportHistoryItem item)
    {
        ApplyEvaluationRunResult(new EvaluationRunResult(
            item.Report,
            item.ReportPath,
            item.ReportJson
        ));
    }

    public void ApplySavedEvaluationManualReview(EvaluationReportHistoryItem item)
    {
        EvaluationReviewSummary = "Manual evaluation review saved.";
    }

    public bool TryGetSelectedEvaluationRunSettings(
        out EvaluationRunSettings settings,
        out string errorMessage
    )
    {
        settings = new EvaluationRunSettings();
        errorMessage = string.Empty;
        if (SelectedEvaluationReportHistoryItem is null)
        {
            errorMessage = "Select a saved evaluation report to rerun.";
            return false;
        }

        if (SelectedEvaluationReportHistoryItem.Report.RunSettings is null)
        {
            errorMessage =
                "The selected report does not include saved run settings. Run a fresh evaluation before rerunning it as a regression check.";
            return false;
        }

        settings = SelectedEvaluationReportHistoryItem.Report.RunSettings;
        if (string.IsNullOrWhiteSpace(settings.SchemaId))
        {
            errorMessage = "The selected report does not include a schema id.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(settings.Backend))
        {
            errorMessage = "The selected report does not include a backend.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(settings.Model))
        {
            errorMessage = "The selected report does not include a model.";
            return false;
        }

        if (settings.TimeoutSeconds <= 0)
        {
            errorMessage = "The selected report has an invalid timeout.";
            return false;
        }

        if (!double.IsFinite(settings.ScoreThreshold)
            || settings.ScoreThreshold < 0
            || settings.ScoreThreshold > 100)
        {
            errorMessage = "The selected report has an invalid score threshold.";
            return false;
        }

        if (settings.Limit is <= 0)
        {
            errorMessage = "The selected report has an invalid sample limit.";
            return false;
        }

        return true;
    }

    public void ApplyEvaluationRunSettings(EvaluationRunSettings settings)
    {
        EvaluationBackend = settings.Backend;
        EvaluationModel = settings.Model;
        EvaluationBaseUrl = settings.BaseUrl ?? string.Empty;
        EvaluationLimit = settings.Limit?.ToString(CultureInfo.InvariantCulture) ?? string.Empty;
        EvaluationScoreThreshold = settings.ScoreThreshold.ToString("0.##", CultureInfo.InvariantCulture);
        EvaluationTimeoutSeconds = settings.TimeoutSeconds.ToString(CultureInfo.InvariantCulture);
    }

    public bool CompareSelectedEvaluationReports()
    {
        if (SelectedEvaluationReportHistoryItem is null)
        {
            EvaluationComparisonSummary = "Select a saved evaluation report first.";
            return false;
        }

        if (SecondaryEvaluationReportHistoryItem is null)
        {
            EvaluationComparisonSummary = "Select a second saved evaluation report to compare against.";
            return false;
        }

        if (string.Equals(
            SelectedEvaluationReportHistoryItem.ReportPath,
            SecondaryEvaluationReportHistoryItem.ReportPath,
            StringComparison.OrdinalIgnoreCase
        ))
        {
            EvaluationComparisonSummary = "Choose two different saved evaluation reports to compare.";
            return false;
        }

        EvaluationComparisonSummary = BuildEvaluationReportComparison(
            SelectedEvaluationReportHistoryItem,
            SecondaryEvaluationReportHistoryItem
        );
        return true;
    }

    public void SetEvaluationReviewError(string message)
    {
        EvaluationReviewSummary = $"Manual evaluation review could not be saved.{Environment.NewLine}{message}";
    }

    public void SetEvaluationError(string message)
    {
        EvaluationSummary = $"Evaluation could not run.{Environment.NewLine}{message}";
        EvaluationReportJson = "No evaluation report was produced.";
        ReportError(message);
    }

    public void SetEvaluationHealthCheckInProgress()
    {
        EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                "Checking evaluation backend...",
                $"Backend: {EvaluationBackend}",
                $"Model: {EvaluationModel}",
            ]
        );
    }

    public void ApplyEvaluationBackendHealthReport(BackendHealthReport report)
    {
        SetAvailableModels(EvaluationAvailableModels, report.AvailableModels);
        EvaluationSummary = FormatBackendHealthReport("Evaluation backend", report);
    }

    public void SetEvaluationModelListInProgress()
    {
        EvaluationModelListSummary = $"Refreshing models from {EvaluationBackend}...";
    }

    public void ApplyEvaluationModelListReport(BackendModelListReport report)
    {
        ApplyModelListReport(
            report,
            EvaluationAvailableModels,
            EvaluationModel,
            model => EvaluationModel = model,
            summary => EvaluationModelListSummary = summary,
            "Evaluation"
        );
    }

    public void SetEvaluationModelListError(string message)
    {
        EvaluationModelListSummary = $"Evaluation model refresh failed.{Environment.NewLine}{message}";
        ReportError(message);
    }

    public void SetAiAssistInProgress()
    {
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                "Running AI Assist...",
                $"Action: {AiAssistAction}",
                $"Backend: {AiAssistBackend}",
                $"Model: {AiAssistModel}",
            ]
        );
        AiAssistReviewText = "Waiting for local model response.";
        _aiAssistSuggestionJsonl = string.Empty;
        ResetCandidateGateState();  // the prior run's verdict must not linger during a new run
        AiAssistSourceDraftText = "Current draft is being sent to AI Assist.";
        AiAssistSuggestedJsonlText = "Waiting for suggested JSONL.";
        AiAssistDiffSummary = "Comparison will appear after the review is queued.";
    }

    public void ApplyAiAssistRunResult(AiAssistRunResult result)
    {
        _aiAssistSuggestionJsonl = result.SuggestedJsonl;
        _aiAssistCandidateGate = result.CandidateGate;
        var hasSuggestedContent = !string.IsNullOrWhiteSpace(result.SuggestedJsonl);
        ApplyCandidateGateStatus(result.CandidateGate, hasSuggestedContent);
        var suggestionStatus = hasSuggestedContent
            ? "available for review"
            : "none";
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                $"Action: {result.Action}",
                $"Model: {result.Model}",
                $"Review state: {result.ReviewState}",
                $"Review required: {(result.ReviewRequired ? "yes" : "no")}",
                $"Suggested JSONL: {suggestionStatus}",
                $"Candidate gate: {AiAssistCandidateGateStatus}",
                $"Warnings: {result.Warnings.Count}",
                $"Validation errors: {result.ValidationErrors.Count}",
            ]
        );

        var lines = new List<string>
        {
            "Model output:",
            result.ModelOutput,
        };

        if (!string.IsNullOrWhiteSpace(result.SuggestedJsonl))
        {
            lines.Add("");
            lines.Add("Suggested JSONL:");
            lines.Add(result.SuggestedJsonl.TrimEnd());
        }

        lines.Add("");
        lines.Add(GateReport.RenderCandidateGate(result.CandidateGate, hasSuggestedContent));

        if (result.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(result.Warnings.Select(warning => $"- {warning}"));
        }

        if (result.ValidationErrors.Count > 0)
        {
            lines.Add("");
            lines.Add("Suggested JSONL validation errors:");
            lines.AddRange(result.ValidationErrors.Select(error => $"- {error}"));
        }

        AiAssistReviewText = string.Join(Environment.NewLine, lines);
        AiAssistSourceDraftText = "Queued review will show the source draft after it is saved.";
        AiAssistSuggestedJsonlText = string.IsNullOrWhiteSpace(result.SuggestedJsonl)
            ? "No suggested JSONL for this review."
            : result.SuggestedJsonl.TrimEnd();
        AiAssistDiffSummary = string.IsNullOrWhiteSpace(result.SuggestedJsonl)
            ? "No suggested JSONL to compare."
            : "Suggested JSONL is available for side-by-side review after queue selection.";
    }

    // Sets the short candidate-gate status label + color for the tab header, honest
    // across the three states: a real gate -> its status; content but no gate ->
    // "not run"; nothing to gate -> "n/a". Never green for null/unknown.
    private void ApplyCandidateGateStatus(GateReport? gate, bool hasSuggestedContent)
    {
        if (gate is not null)
        {
            AiAssistCandidateGateStatus = (gate.OverallStatus ?? string.Empty).ToUpperInvariant();
            AiAssistCandidateGateColor = GateReport.StatusColor(gate.OverallStatus);
        }
        else
        {
            AiAssistCandidateGateStatus = hasSuggestedContent ? "not run" : "n/a";
            AiAssistCandidateGateColor = GateReport.StatusColor(null);
        }
    }

    // Revert the candidate-gate state to the neutral initial state (no current verdict).
    // Called whenever there is no meaningful current gate to show — a new run starting, a
    // run that failed, or a project switch — so a prior run's/project's verdict can never
    // linger in the header (or trigger a spurious block-confirm via ActiveAiAssistCandidateGate).
    private void ResetCandidateGateState()
    {
        _aiAssistCandidateGate = null;
        AiAssistCandidateGateStatus = "—";
        AiAssistCandidateGateColor = GateReport.StatusColor(null);
    }

    public void SetAiAssistReviewQueue(IEnumerable<AiAssistReviewQueueItem> items)
    {
        _allAiAssistReviewQueue.Clear();
        _allAiAssistReviewQueue.AddRange(items);
        RebuildAiAssistReviewQueue();
    }

    public void SetAiAssistQueueViews(IEnumerable<AiAssistQueueView> views)
    {
        var selectedName = SelectedAiAssistQueueView?.Name;
        AiAssistQueueViews.Clear();
        foreach (var view in views)
        {
            AiAssistQueueViews.Add(view);
        }

        SelectedAiAssistQueueView = AiAssistQueueViews
            .FirstOrDefault(view => string.Equals(view.Name, selectedName, StringComparison.OrdinalIgnoreCase))
            ?? AiAssistQueueViews.FirstOrDefault();
    }

    public void SetAiAssistRewriteBatches(IEnumerable<AiAssistRewriteBatch> batches)
    {
        var selectedBatchId = SelectedAiAssistRewriteBatch?.BatchId;
        AiAssistRewriteBatches.Clear();
        foreach (var batch in batches)
        {
            AiAssistRewriteBatches.Add(batch);
        }

        SelectedAiAssistRewriteBatch = AiAssistRewriteBatches
            .FirstOrDefault(batch => string.Equals(batch.BatchId, selectedBatchId, StringComparison.Ordinal))
            ?? AiAssistRewriteBatches.FirstOrDefault();

        AiAssistRewriteBatchSummary = AiAssistRewriteBatches.Count == 0
            ? "No prepared rewrite batches are saved for this project."
            : $"Saved rewrite batches: {AiAssistRewriteBatches.Count}. Select one to resume.";
    }

    public AiAssistQueueView BuildCurrentAiAssistQueueView()
    {
        return new AiAssistQueueView
        {
            Name = AiAssistQueueViewName.Trim(),
            Filter = AiAssistQueueFilter,
            Search = AiAssistQueueSearch.Trim(),
            Sort = AiAssistQueueSort,
        };
    }

    public void ApplyAiAssistQueueView(AiAssistQueueView view)
    {
        AiAssistQueueViewName = view.Name;
        AiAssistQueueFilter = IsAiAssistQueueFilterOption(view.Filter) ? view.Filter : "All";
        AiAssistQueueSearch = view.Search;
        AiAssistQueueSort = IsAiAssistQueueSortOption(view.Sort) ? view.Sort : "Newest";
    }

    public void ApplyAiAssistQueueViewSaved(AiAssistQueueView view)
    {
        AiAssistQueueSummary = $"AI Assist queue view saved: {view.Name}.";
    }

    public void ApplyAiAssistQueueViewLoaded(AiAssistQueueView view)
    {
        AiAssistQueueSummary = $"AI Assist queue view loaded: {view.Name}.";
    }

    public bool TryGetLastPreparedAiAssistRewriteBatch(
        out AiAssistRewriteBatch batch,
        out string errorMessage
    )
    {
        if (_lastPreparedAiAssistRewriteBatch is null)
        {
            batch = new AiAssistRewriteBatch();
            errorMessage = "Prepare a synthetic batch rewrite before saving it.";
            return false;
        }

        batch = _lastPreparedAiAssistRewriteBatch;
        errorMessage = string.Empty;
        return true;
    }

    public void ApplyAiAssistRewriteBatchSaved(AiAssistRewriteBatch batch)
    {
        AiAssistRewriteBatchSummary =
            $"Saved rewrite batch for rows {FormatRowNumbers(batch.RowNumbers)}.";
    }

    public bool ResumeAiAssistRewriteBatch()
    {
        if (SelectedAiAssistRewriteBatch is null)
        {
            AiAssistRewriteBatchSummary = "Select a saved rewrite batch before resuming.";
            return false;
        }

        LoadDraft(SelectedAiAssistRewriteBatch.SourceDraft);
        AiAssistAction = AiAssistActionPresets.Contains(SelectedAiAssistRewriteBatch.Action)
            ? SelectedAiAssistRewriteBatch.Action
            : "rewrite-output";
        AiAssistInstruction = SelectedAiAssistRewriteBatch.Instruction;
        AiAssistRewriteBatchSummary =
            $"Resumed rewrite batch for rows {FormatRowNumbers(SelectedAiAssistRewriteBatch.RowNumbers)}.";
        return true;
    }

    public void SetAiAssistRewriteBatchError(string message)
    {
        AiAssistRewriteBatchSummary = $"AI Assist rewrite batch could not be updated.{Environment.NewLine}{message}";
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

        var example = Examples.FirstOrDefault(item => item.RowNumber == SelectedReviewedFix.RowNumber);
        if (example is null)
        {
            ReviewedFixSummary =
                $"Saved row {SelectedReviewedFix.RowNumber} for {SelectedReviewedFix.ExampleId} is not loaded in the Examples list.";
            return false;
        }

        SelectedExample = example;
        LoadDraft(example.Json);
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

    public void SetEvaluationFailureFilters(IEnumerable<EvaluationFailureFilter> filters)
    {
        var selectedName = SelectedEvaluationFailureFilter?.Name;
        EvaluationFailureFilters.Clear();
        foreach (var filter in filters)
        {
            EvaluationFailureFilters.Add(filter);
        }

        SelectedEvaluationFailureFilter = EvaluationFailureFilters
            .FirstOrDefault(filter => string.Equals(filter.Name, selectedName, StringComparison.OrdinalIgnoreCase))
            ?? EvaluationFailureFilters.FirstOrDefault();

        EvaluationFailureFilterSummary = EvaluationFailureFilters.Count == 0
            ? "No saved failure filters. Set filters, name the view, then Save Filter."
            : $"Saved failure filters: {EvaluationFailureFilters.Count}. Select one and Apply Filter.";
    }

    public EvaluationFailureFilter BuildCurrentEvaluationFailureFilter()
    {
        return new EvaluationFailureFilter
        {
            Name = EvaluationFailureFilterName.Trim(),
            Status = EvaluationResultFilter,
            Tag = EvaluationTagFilter,
            FailureReason = EvaluationFailureReasonFilter,
            ScoreBand = EvaluationScoreBandFilter,
        };
    }

    public void ApplyEvaluationFailureFilter(EvaluationFailureFilter filter)
    {
        EvaluationFailureFilterName = filter.Name;
        SetField(
            ref _evaluationResultFilter,
            EvaluationResultFilterOptions.Contains(filter.Status) ? filter.Status : "All",
            nameof(EvaluationResultFilter)
        );
        SetField(
            ref _evaluationTagFilter,
            EvaluationTagFilterOptions.Contains(filter.Tag) ? filter.Tag : "All",
            nameof(EvaluationTagFilter)
        );
        SetField(
            ref _evaluationFailureReasonFilter,
            EvaluationFailureReasonFilterOptions.Contains(filter.FailureReason) ? filter.FailureReason : "All",
            nameof(EvaluationFailureReasonFilter)
        );
        SetField(
            ref _evaluationScoreBandFilter,
            EvaluationScoreBandFilterOptions.Contains(filter.ScoreBand) ? filter.ScoreBand : "All",
            nameof(EvaluationScoreBandFilter)
        );
        RebuildEvaluationResults();
        EvaluationFailureFilterSummary = $"Applied failure filter: {filter.Name}.";
    }

    public void ApplyEvaluationFailureFilterSaved(EvaluationFailureFilter filter)
    {
        EvaluationFailureFilterSummary = $"Saved failure filter: {filter.Name}.";
    }

    public void SetEvaluationFailureFilterError(string message)
    {
        EvaluationFailureFilterSummary =
            $"Failure filter could not be updated.{Environment.NewLine}{message}";
    }

    private void RebuildEvaluationFilterOptions()
    {
        var tags = _allEvaluationResults
            .SelectMany(result => result.Tags)
            .Where(tag => !string.IsNullOrWhiteSpace(tag))
            .Select(tag => tag.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(tag => tag, StringComparer.OrdinalIgnoreCase)
            .ToList();
        SyncFilterOptions(EvaluationTagFilterOptions, tags);

        var reasons = _allEvaluationResults
            .Where(result => !result.Passed)
            .Select(FailureReason)
            .Where(reason => !string.IsNullOrWhiteSpace(reason))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(reason => reason, StringComparer.OrdinalIgnoreCase)
            .ToList();
        SyncFilterOptions(EvaluationFailureReasonFilterOptions, reasons);

        if (!EvaluationTagFilterOptions.Contains(EvaluationTagFilter))
        {
            SetField(ref _evaluationTagFilter, "All", nameof(EvaluationTagFilter));
        }
        if (!EvaluationFailureReasonFilterOptions.Contains(EvaluationFailureReasonFilter))
        {
            SetField(ref _evaluationFailureReasonFilter, "All", nameof(EvaluationFailureReasonFilter));
        }
    }

    private static void SyncFilterOptions(ObservableCollection<string> target, IReadOnlyList<string> values)
    {
        var desired = new List<string> { "All" };
        desired.AddRange(values);
        if (target.SequenceEqual(desired, StringComparer.Ordinal))
        {
            return;
        }

        target.Clear();
        foreach (var value in desired)
        {
            target.Add(value);
        }
    }

    private void ResetEvaluationFailureFilters()
    {
        EvaluationFailureFilters.Clear();
        SelectedEvaluationFailureFilter = null;
        EvaluationFailureFilterName = "Failure View";
        SetField(ref _evaluationTagFilter, "All", nameof(EvaluationTagFilter));
        SetField(ref _evaluationFailureReasonFilter, "All", nameof(EvaluationFailureReasonFilter));
        SetField(ref _evaluationScoreBandFilter, "All", nameof(EvaluationScoreBandFilter));
        SyncFilterOptions(EvaluationTagFilterOptions, []);
        SyncFilterOptions(EvaluationFailureReasonFilterOptions, []);
        EvaluationFailureFilterSummary =
            "Save the active status, tag, failure-reason, and score-band filters as a named view.";
    }

    public void ApplyAiAssistReviewQueueItem(AiAssistReviewQueueItem item)
    {
        _aiAssistSuggestionJsonl = item.SuggestedJsonl;
        var hasSuggestedContent = !string.IsNullOrWhiteSpace(item.SuggestedJsonl);
        ApplyCandidateGateStatus(item.CandidateGate, hasSuggestedContent);
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                $"Action: {item.Action}",
                $"Model: {item.Model}",
                $"Review state: {item.ReviewState}",
                $"Suggested JSONL: {(hasSuggestedContent ? "available for review" : "none")}",
                $"Candidate gate: {AiAssistCandidateGateStatus}",
                $"Warnings: {item.Warnings.Count}",
                $"Validation errors: {item.ValidationErrors.Count}",
            ]
        );
        AiAssistReviewText = item.DetailText;
        AiAssistSourceDraftText = string.IsNullOrWhiteSpace(item.SourceDraft)
            ? "No source draft was recorded for this review."
            : item.SourceDraft.TrimEnd();
        AiAssistSuggestedJsonlText = string.IsNullOrWhiteSpace(item.SuggestedJsonl)
            ? "No suggested JSONL for this review."
            : item.SuggestedJsonl.TrimEnd();
        AiAssistDiffSummary = BuildAiAssistDiffSummary(item);
    }

    public void ApplyAiAssistReviewState(AiAssistReviewQueueItem item)
    {
        ApplyAiAssistReviewQueueItem(item);
        AiAssistQueueSummary = $"AI Assist review marked {item.ReviewState}.";
    }

    public void ApplyAiAssistBulkReviewState(int updatedCount, string reviewState, int undoStepsAvailable)
    {
        AiAssistQueueSummary =
            $"AI Assist bulk triage marked {updatedCount} review(s) {reviewState}. Undo steps available: {undoStepsAvailable}.";
    }

    public void ApplyAiAssistBulkUndoReviewState(int updatedCount, int undoStepsAvailable)
    {
        AiAssistQueueSummary =
            $"AI Assist bulk triage undo restored {updatedCount} review(s). Undo steps remaining: {undoStepsAvailable}.";
    }

    public IReadOnlyList<string> GetVisibleAiAssistReviewIds()
    {
        return AiAssistReviewQueue.Select(item => item.ReviewId).ToList();
    }

    public IReadOnlyDictionary<string, string> GetVisibleAiAssistReviewStates()
    {
        return AiAssistReviewQueue.ToDictionary(
            item => item.ReviewId,
            item => item.ReviewState,
            StringComparer.Ordinal
        );
    }

    public void SetAiAssistError(string message)
    {
        AiAssistSummary = $"AI Assist could not run.{Environment.NewLine}{message}";
        ReportError(message);
        AiAssistReviewText = "No AI Assist suggestion was produced.";
        _aiAssistSuggestionJsonl = string.Empty;
        ResetCandidateGateState();  // a failed run must not keep the previous run's verdict
        ClearAiAssistComparison(
            "No source draft was compared.",
            "No suggested JSONL was produced.",
            "No comparison is available."
        );
    }

    public void SetAiAssistQueueError(string message)
    {
        AiAssistQueueSummary = $"AI Assist review queue could not be updated.{Environment.NewLine}{message}";
    }

    public void SetAiAssistHealthCheckInProgress()
    {
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                "Checking AI Assist backend...",
                $"Backend: {AiAssistBackend}",
                $"Model: {AiAssistModel}",
            ]
        );
    }

    public void ApplyAiAssistBackendHealthReport(BackendHealthReport report)
    {
        SetAvailableModels(AiAssistAvailableModels, report.AvailableModels);
        AiAssistSummary = FormatBackendHealthReport("AI Assist backend", report);
    }

    public void SetAiAssistModelListInProgress()
    {
        AiAssistModelListSummary = $"Refreshing models from {AiAssistBackend}...";
    }

    public void ApplyAiAssistModelListReport(BackendModelListReport report)
    {
        ApplyModelListReport(
            report,
            AiAssistAvailableModels,
            AiAssistModel,
            model => AiAssistModel = model,
            summary => AiAssistModelListSummary = summary,
            "AI Assist"
        );
    }

    public void SetAiAssistModelListError(string message)
    {
        AiAssistModelListSummary = $"AI Assist model refresh failed.{Environment.NewLine}{message}";
        ReportError(message);
    }

    public bool MoveAiAssistSuggestionToDraft()
    {
        var suggestionJsonl = SelectedAiAssistReviewQueueItem?.SuggestedJsonl ?? _aiAssistSuggestionJsonl;
        if (string.IsNullOrWhiteSpace(suggestionJsonl))
        {
            AiAssistSummary = "No AI Assist JSONL suggestion is available to move into the draft.";
            return false;
        }

        LoadDraft(suggestionJsonl.TrimEnd());
        AiAssistSummary = "AI Assist suggestion moved to Writing Studio. Validate and edit before saving.";
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

    public void SetImportQuarantineItems(IEnumerable<ImportQuarantineItem> items)
    {
        ImportQuarantineItems.Clear();
        foreach (var item in items)
        {
            ImportQuarantineItems.Add(item);
        }

        SelectedImportQuarantineItem = ImportQuarantineItems.FirstOrDefault();
        SelectedImportQuarantineDetail = SelectedImportQuarantineItem?.DetailText
            ?? "No rejected import rows are in quarantine for this project.";
    }

    private ImportQuarantineItem? _pendingRetryItem;

    public void RetrySelectedImportQuarantineItem()
    {
        if (SelectedImportQuarantineItem is not null)
        {
            LoadDraft(SelectedImportQuarantineItem.Raw);
            // Remember which quarantine row is being repaired so a successful save can
            // remove it (otherwise the record orphans in quarantine forever).
            _pendingRetryItem = SelectedImportQuarantineItem;
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

    private void SetEvaluationResults(IReadOnlyList<EvaluationExampleResult> results)
    {
        _allEvaluationResults.Clear();
        _allEvaluationResults.AddRange(results);
        RebuildEvaluationFilterOptions();
        RebuildEvaluationResults();
    }

    private void RebuildEvaluationResults()
    {
        var selectedExampleId = SelectedEvaluationExampleResult?.ExampleId;
        EvaluationResults.Clear();
        SelectedEvaluationExampleResult = null;

        foreach (var result in _allEvaluationResults.Where(MatchesEvaluationResultFilter))
        {
            EvaluationResults.Add(result);
        }

        EvaluationResultsSummary = BuildEvaluationResultsSummary();
        SelectedEvaluationExampleResult = EvaluationResults
            .FirstOrDefault(result => result.ExampleId == selectedExampleId)
            ?? EvaluationResults.FirstOrDefault();

        if (EvaluationResults.Count == 0)
        {
            ClearEvaluationExampleSelection();
        }
    }

    private bool MatchesEvaluationResultFilter(EvaluationExampleResult result)
    {
        var statusMatch = EvaluationResultFilter switch
        {
            "Failed" => !result.Passed,
            "Passed" => result.Passed,
            "Manually Scored" => result.ManualScore is not null
                || !string.IsNullOrWhiteSpace(result.ManualNotes),
            _ => true,
        };
        if (!statusMatch)
        {
            return false;
        }

        if (!IsAllFilter(EvaluationTagFilter)
            && !result.Tags.Any(tag =>
                string.Equals(tag?.Trim(), EvaluationTagFilter, StringComparison.OrdinalIgnoreCase)))
        {
            return false;
        }

        if (!IsAllFilter(EvaluationFailureReasonFilter)
            && !string.Equals(FailureReason(result), EvaluationFailureReasonFilter, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (!IsAllFilter(EvaluationScoreBandFilter)
            && !string.Equals(ScoreBand(result.Score), EvaluationScoreBandFilter, StringComparison.Ordinal))
        {
            return false;
        }

        return true;
    }

    private static bool IsAllFilter(string value)
    {
        return string.IsNullOrWhiteSpace(value)
            || string.Equals(value, "All", StringComparison.OrdinalIgnoreCase);
    }

    private string BuildEvaluationResultsSummary()
    {
        if (_allEvaluationResults.Count == 0)
        {
            return "No evaluation examples are loaded.";
        }

        var failed = _allEvaluationResults.Count(result => !result.Passed);
        var passed = _allEvaluationResults.Count(result => result.Passed);
        var manuallyScored = _allEvaluationResults.Count(result =>
            result.ManualScore is not null || !string.IsNullOrWhiteSpace(result.ManualNotes)
        );

        var drilldown = new List<string>();
        if (!IsAllFilter(EvaluationTagFilter)) drilldown.Add($"tag={EvaluationTagFilter}");
        if (!IsAllFilter(EvaluationFailureReasonFilter)) drilldown.Add($"reason={EvaluationFailureReasonFilter}");
        if (!IsAllFilter(EvaluationScoreBandFilter)) drilldown.Add($"band={EvaluationScoreBandFilter}");
        var drilldownText = drilldown.Count == 0
            ? string.Empty
            : $" Drilldown: {string.Join(", ", drilldown)}.";

        return $"Results: {failed} failed, {passed} passed, {manuallyScored} manually reviewed. Filter: {EvaluationResultFilter}, showing {EvaluationResults.Count} of {_allEvaluationResults.Count}.{drilldownText}";
    }

    private void ApplySelectedEvaluationExample(EvaluationExampleResult? result)
    {
        if (result is null)
        {
            ClearEvaluationExampleSelection();
            return;
        }

        SelectedEvaluationExampleDetail = result.DetailText;
        EvaluationManualScore = result.ManualScore?.ToString("0.##") ?? string.Empty;
        EvaluationManualNotes = result.ManualNotes ?? string.Empty;
        EvaluationReviewSummary = "Edit the manual score or note, then save review.";
    }

    private void ClearEvaluationResults()
    {
        _allEvaluationResults.Clear();
        EvaluationResults.Clear();
        SelectedEvaluationExampleResult = null;
        EvaluationResultsSummary = "Evaluation example review queue appears after a run or report reload.";
        ClearEvaluationExampleSelection();
    }

    private void SetPreferenceReviewItems(IEnumerable<SavedExampleItem> examples)
    {
        _allPreferenceReviewItems.Clear();
        if (ActiveSchemaId == "preference")
        {
            foreach (var example in examples)
            {
                if (TryCreatePreferenceReviewItem(example, out var item))
                {
                    _allPreferenceReviewItems.Add(item);
                }
            }
        }

        RebuildPreferenceReviewItems();
    }

    private void RebuildPreferenceReviewItems()
    {
        var selectedRowNumber = SelectedPreferenceReviewItem?.RowNumber;
        PreferenceReviewItems.Clear();
        SelectedPreferenceReviewItem = null;

        foreach (var item in _allPreferenceReviewItems
            .Where(MatchesPreferenceContrastFilter)
            .OrderBy(item => PreferenceContrastRank(item.Contrast))
            .ThenByDescending(item => item.TokenOverlap)
            .ThenBy(item => item.RowNumber))
        {
            PreferenceReviewItems.Add(item);
        }

        PreferenceRankingSummary = BuildPreferenceRankingSummary();
        SelectedPreferenceReviewItem = PreferenceReviewItems
            .FirstOrDefault(item => item.RowNumber == selectedRowNumber)
            ?? PreferenceReviewItems.FirstOrDefault();

        if (PreferenceReviewItems.Count == 0)
        {
            ClearPreferenceReview();
        }
    }

    private bool MatchesPreferenceContrastFilter(PreferenceReviewItem item)
    {
        return PreferenceContrastFilter switch
        {
            "Weak" => item.Contrast == "weak",
            "Moderate" => item.Contrast == "moderate",
            "Strong" => item.Contrast == "strong",
            _ => true,
        };
    }

    private string BuildPreferenceRankingSummary()
    {
        if (ActiveSchemaId != "preference")
        {
            return "Preference ranking is available for preference projects.";
        }

        if (_allPreferenceReviewItems.Count == 0)
        {
            return "No valid preference pairs are loaded.";
        }

        var weak = _allPreferenceReviewItems.Count(item => item.Contrast == "weak");
        var moderate = _allPreferenceReviewItems.Count(item => item.Contrast == "moderate");
        var strong = _allPreferenceReviewItems.Count(item => item.Contrast == "strong");
        return $"Ranking: {weak} weak, {moderate} moderate, {strong} strong. Filter: {PreferenceContrastFilter}, showing {PreferenceReviewItems.Count} of {_allPreferenceReviewItems.Count}.";
    }

    private void ApplyPreferenceReviewItem(PreferenceReviewItem? item)
    {
        if (item is null)
        {
            ClearPreferenceReview();
            return;
        }

        if (ActiveSchemaId != "preference")
        {
            PreferencePromptText = "Preference review is intended for preference projects.";
            PreferenceChosenText = "Create or select a preference project to inspect chosen responses.";
            PreferenceRejectedText = "Create or select a preference project to inspect rejected responses.";
            PreferenceReasonText = "No preference reason is selected.";
            PreferenceReviewSummary = "Current project is not a preference dataset.";
            return;
        }

        PreferencePromptText = item.Prompt;
        PreferenceChosenText = item.Chosen;
        PreferenceRejectedText = item.Rejected;
        PreferenceReasonText = string.IsNullOrWhiteSpace(item.Reason)
            ? "No reason field is present for this pair."
            : item.Reason;
        PreferenceReviewSummary = $"Example {item.RowNumber}. {item.ContrastSummary}";
    }

    private void ClearPreferenceReview()
    {
        PreferencePromptText = "Select a saved preference example to inspect the prompt.";
        PreferenceChosenText = "Chosen response appears here.";
        PreferenceRejectedText = "Rejected response appears here.";
        PreferenceReasonText = "Reason or preference notes appear here.";
        PreferenceReviewSummary = ActiveSchemaId == "preference"
            ? "Save or select a preference pair to review chosen/rejected contrast."
            : "Preference pair review is available for preference projects.";
    }

    private static bool TryCreatePreferenceReviewItem(
        SavedExampleItem example,
        out PreferenceReviewItem item
    )
    {
        item = new PreferenceReviewItem();
        if (!TryReadPreferencePair(
            example.Json,
            out var prompt,
            out var chosen,
            out var rejected,
            out var reason,
            out _,
            out _
        ))
        {
            return false;
        }

        var metrics = BuildPreferenceContrastMetrics(chosen, rejected);
        item = new PreferenceReviewItem
        {
            RowNumber = example.RowNumber,
            Prompt = prompt,
            Chosen = chosen,
            Rejected = rejected,
            Reason = reason,
            Json = example.Json,
            Contrast = metrics.Contrast,
            TokenOverlap = metrics.TokenOverlap,
            CharacterDelta = metrics.CharacterDelta,
        };
        return true;
    }

    private static bool TryReadPreferencePair(
        string json,
        out string prompt,
        out string chosen,
        out string rejected,
        out string reason,
        out string contrastSummary,
        out string errorMessage
    )
    {
        prompt = string.Empty;
        chosen = string.Empty;
        rejected = string.Empty;
        reason = string.Empty;
        contrastSummary = string.Empty;
        errorMessage = string.Empty;

        try
        {
            using var document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                errorMessage = "Preference row must be a JSON object.";
                return false;
            }

            var root = document.RootElement;
            prompt = ReadStringProperty(root, "prompt");
            chosen = ReadStringProperty(root, "chosen");
            rejected = ReadStringProperty(root, "rejected");
            reason = ReadStringProperty(root, "reason");
        }
        catch (JsonException)
        {
            errorMessage = "Preference row contains invalid JSON.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(prompt)
            || string.IsNullOrWhiteSpace(chosen)
            || string.IsNullOrWhiteSpace(rejected))
        {
            errorMessage = "Preference row must include non-empty prompt, chosen, and rejected fields.";
            return false;
        }

        contrastSummary = BuildPreferenceContrastSummary(chosen, rejected);
        return true;
    }

    private static string ReadStringProperty(JsonElement root, string propertyName)
    {
        return root.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;
    }

    private static string BuildPreferenceContrastSummary(string chosen, string rejected)
    {
        var metrics = BuildPreferenceContrastMetrics(chosen, rejected);
        var deltaText = metrics.CharacterDelta >= 0
            ? $"+{metrics.CharacterDelta}"
            : metrics.CharacterDelta.ToString();
        return $"Contrast: {metrics.Contrast}. Token overlap: {metrics.TokenOverlap:P0}. Chosen/rejected character delta: {deltaText}.";
    }

    private static (string Contrast, double TokenOverlap, int CharacterDelta) BuildPreferenceContrastMetrics(
        string chosen,
        string rejected
    )
    {
        var chosenTokens = TokenizeForPreference(chosen).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var rejectedTokens = TokenizeForPreference(rejected).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var overlap = 0.0;
        if (chosenTokens.Count > 0 && rejectedTokens.Count > 0)
        {
            var shared = chosenTokens.Intersect(rejectedTokens, StringComparer.OrdinalIgnoreCase).Count();
            overlap = shared / (double)Math.Max(chosenTokens.Count, rejectedTokens.Count);
        }

        var contrast = overlap switch
        {
            >= 0.9 => "weak",
            >= 0.65 => "moderate",
            _ => "strong",
        };
        return (contrast, overlap, chosen.Length - rejected.Length);
    }

    private static int PreferenceContrastRank(string contrast)
    {
        return contrast switch
        {
            "weak" => 0,
            "moderate" => 1,
            "strong" => 2,
            _ => 3,
        };
    }

    private static IEnumerable<string> TokenizeForPreference(string value)
    {
        return value
            .ToLowerInvariant()
            .Split(
                [' ', '\r', '\n', '\t', '.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '[', ']'],
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries
            )
            .Where(token => token.Length > 1);
    }

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

    private void ClearEvaluationExampleSelection()
    {
        SelectedEvaluationExampleDetail =
            "Per-example evaluation results appear here after a run or report reload.";
        EvaluationManualScore = string.Empty;
        EvaluationManualNotes = string.Empty;
        EvaluationReviewSummary = "Select an evaluation result to add a manual score or note.";
    }

    private void ClearAiAssistComparison(
        string sourceMessage = "Source draft appears here after selecting an AI Assist review.",
        string suggestionMessage = "Suggested JSONL appears here after selecting an AI Assist review.",
        string summaryMessage = "Select a queued AI Assist review to compare source and suggestion."
    )
    {
        AiAssistSourceDraftText = sourceMessage;
        AiAssistSuggestedJsonlText = suggestionMessage;
        AiAssistDiffSummary = summaryMessage;
    }

    private void RebuildAiAssistReviewQueue()
    {
        var selectedReviewId = SelectedAiAssistReviewQueueItem?.ReviewId;
        AiAssistReviewQueue.Clear();
        SelectedAiAssistReviewQueueItem = null;

        foreach (var item in SortAiAssistReviewQueue(
            _allAiAssistReviewQueue
                .Where(MatchesAiAssistQueueFilter)
                .Where(MatchesAiAssistQueueSearch)
        ))
        {
            AiAssistReviewQueue.Add(item);
        }

        AiAssistQueueSummary = BuildAiAssistQueueSummary();
        SelectedAiAssistReviewQueueItem = AiAssistReviewQueue
            .FirstOrDefault(item => item.ReviewId == selectedReviewId)
            ?? AiAssistReviewQueue.FirstOrDefault();

        if (SelectedAiAssistReviewQueueItem is null)
        {
            _aiAssistSuggestionJsonl = string.Empty;
            AiAssistReviewText = _allAiAssistReviewQueue.Count == 0
                ? "AI Assist review output appears here."
                : "No AI Assist reviews match the current queue controls.";
            ClearAiAssistComparison(
                "No source draft is selected.",
                "No suggested JSONL is selected.",
                "No comparison is available for the current queue controls."
            );
        }
    }

    private bool MatchesAiAssistQueueFilter(AiAssistReviewQueueItem item)
    {
        return AiAssistQueueFilter switch
        {
            "Pending" => item.ReviewState == "review_required",
            "Accepted" => item.ReviewState == "accepted",
            "Rejected" => item.ReviewState == "rejected",
            _ => true,
        };
    }

    private bool MatchesAiAssistQueueSearch(AiAssistReviewQueueItem item)
    {
        var search = AiAssistQueueSearch.Trim();
        if (string.IsNullOrWhiteSpace(search))
        {
            return true;
        }

        return ContainsSearch(item.ReviewId, search)
            || ContainsSearch(item.ReviewState, search)
            || ContainsSearch(item.Action, search)
            || ContainsSearch(item.Model, search)
            || ContainsSearch(item.PromptTemplateId, search)
            || ContainsSearch(item.SourceDraft, search)
            || ContainsSearch(item.ModelOutput, search)
            || ContainsSearch(item.SuggestedJsonl, search)
            || item.Warnings.Any(warning => ContainsSearch(warning, search))
            || item.ValidationErrors.Any(error => ContainsSearch(error, search));
    }

    private IEnumerable<AiAssistReviewQueueItem> SortAiAssistReviewQueue(
        IEnumerable<AiAssistReviewQueueItem> items
    )
    {
        return AiAssistQueueSort switch
        {
            "Oldest" => items
                .OrderBy(item => item.CreatedAt)
                .ThenBy(item => item.Model, StringComparer.OrdinalIgnoreCase),
            "State" => items
                .OrderBy(item => item.ReviewState, StringComparer.OrdinalIgnoreCase)
                .ThenByDescending(item => item.CreatedAt),
            "Model" => items
                .OrderBy(item => item.Model, StringComparer.OrdinalIgnoreCase)
                .ThenByDescending(item => item.CreatedAt),
            "Action" => items
                .OrderBy(item => item.Action, StringComparer.OrdinalIgnoreCase)
                .ThenByDescending(item => item.CreatedAt),
            _ => items
                .OrderByDescending(item => item.CreatedAt)
                .ThenBy(item => item.Model, StringComparer.OrdinalIgnoreCase),
        };
    }

    private string BuildAiAssistQueueSummary()
    {
        if (_allAiAssistReviewQueue.Count == 0)
        {
            return "No AI Assist reviews are queued for this project.";
        }

        var pending = _allAiAssistReviewQueue.Count(item => item.ReviewState == "review_required");
        var accepted = _allAiAssistReviewQueue.Count(item => item.ReviewState == "accepted");
        var rejected = _allAiAssistReviewQueue.Count(item => item.ReviewState == "rejected");
        var search = string.IsNullOrWhiteSpace(AiAssistQueueSearch)
            ? "none"
            : AiAssistQueueSearch.Trim();
        return $"Queue: {pending} pending, {accepted} accepted, {rejected} rejected. Filter: {AiAssistQueueFilter}, search: {search}, sort: {AiAssistQueueSort}, showing {AiAssistReviewQueue.Count} of {_allAiAssistReviewQueue.Count}.";
    }

    private bool IsAiAssistQueueFilterOption(string value)
    {
        return AiAssistQueueFilterOptions.Any(option =>
            string.Equals(option, value, StringComparison.OrdinalIgnoreCase)
        );
    }

    private bool IsAiAssistQueueSortOption(string value)
    {
        return AiAssistQueueSortOptions.Any(option =>
            string.Equals(option, value, StringComparison.OrdinalIgnoreCase)
        );
    }

    private static bool ContainsSearch(string value, string search)
    {
        return value.Contains(search, StringComparison.OrdinalIgnoreCase);
    }

    private void ApplyAiAssistActionPresets(string schemaId)
    {
        var presets = schemaId switch
        {
            "instruction" or "chat" or "code" => new[]
            {
                "review",
                "suggest-tags",
                "rewrite-output",
                "draft-example",
            },
            "preference" => new[]
            {
                "review",
                "judge-preference-strength",
                "rewrite-output",
                "suggest-tags",
            },
            "raw_text" => new[]
            {
                "review",
                "suggest-tags",
            },
            _ => new[]
            {
                "review",
                "suggest-tags",
                "draft-example",
            },
        };

        AiAssistActionPresets.Clear();
        foreach (var preset in presets)
        {
            AiAssistActionPresets.Add(preset);
        }

        if (!AiAssistActionPresets.Contains(AiAssistAction))
        {
            AiAssistAction = AiAssistActionPresets.FirstOrDefault() ?? "review";
        }
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

    private static string FormatManualScoreSummary(EvaluationReport report)
    {
        if (report.ManuallyScoredExamples == 0 || report.AverageManualScore is null)
        {
            return "none";
        }

        return $"{report.ManuallyScoredExamples} example(s), average {report.AverageManualScore:0.##}";
    }

    /// <summary>Describe the evaluation metric honestly so the score is never read as a
    /// quality judgment when it is only lexical overlap.</summary>
    private static string DescribeEvaluationMetric(string? metric) => (metric ?? "keyword_overlap") switch
    {
        "llm_judge" => "LLM judge (0-100 quality + rationale)",
        "keyword_overlap" => "keyword overlap (recall) — a lexical proxy, NOT a quality judgment; "
            + "confirm with manual scores",
        _ => metric!,
    };

    private static string FormatTagSummary(EvaluationReport report)
    {
        var summaries = report.TagSummary.Count == 0
            ? BuildTagSummary(report.Results)
            : report.TagSummary;
        if (summaries.Count == 0)
        {
            return "none";
        }

        return string.Join(
            "; ",
            summaries
                .Take(5)
                .Select(summary =>
                    $"{summary.Tag}: {summary.Examples} ex, {summary.FailedExamples} failed, avg {summary.AverageScore:0.##}")
        );
    }

    private static string FormatFailureReasonSummary(EvaluationReport report)
    {
        var summaries = report.FailureReasonSummary.Count == 0
            ? BuildFailureReasonSummary(report.Results)
            : report.FailureReasonSummary;
        if (summaries.Count == 0)
        {
            return "none";
        }

        return string.Join(
            "; ",
            summaries
                .Take(5)
                .Select(summary =>
                    $"{FormatFailureReason(summary.Reason)}: {summary.FailedExamples}")
        );
    }

    private static string FormatScoreBandSummary(EvaluationReport report)
    {
        var summaries = report.ScoreBandSummary.Count == 0
            ? BuildScoreBandSummary(report.Results)
            : report.ScoreBandSummary;
        if (summaries.Count == 0)
        {
            return "none";
        }

        return string.Join(
            "; ",
            summaries
                .Select(summary =>
                    $"{summary.Band}: {summary.Examples} ex, {summary.FailedExamples} failed, avg {summary.AverageScore:0.##}")
        );
    }

    private static List<EvaluationTagSummary> BuildTagSummary(
        IReadOnlyList<EvaluationExampleResult> results
    )
    {
        return results
            .SelectMany(result => NormalizeTags(result.Tags).Select(tag => new { Tag = tag, Result = result }))
            .GroupBy(item => item.Tag, StringComparer.OrdinalIgnoreCase)
            .Select(group =>
            {
                var groupResults = group.Select(item => item.Result).ToList();
                return new EvaluationTagSummary
                {
                    Tag = group.Key,
                    Examples = groupResults.Count,
                    FailedExamples = groupResults.Count(result => !result.Passed),
                    AverageScore = AverageScore(groupResults),
                };
            })
            .OrderByDescending(summary => summary.FailedExamples)
            .ThenBy(summary => summary.Tag, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<EvaluationFailureReasonSummary> BuildFailureReasonSummary(
        IReadOnlyList<EvaluationExampleResult> results
    )
    {
        return results
            .Where(result => !result.Passed)
            .GroupBy(result => FailureReason(result), StringComparer.OrdinalIgnoreCase)
            .Select(group => new EvaluationFailureReasonSummary
            {
                Reason = group.Key,
                FailedExamples = group.Count(),
            })
            .OrderByDescending(summary => summary.FailedExamples)
            .ThenBy(summary => summary.Reason, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static List<EvaluationScoreBandSummary> BuildScoreBandSummary(
        IReadOnlyList<EvaluationExampleResult> results
    )
    {
        return results
            .GroupBy(result => ScoreBand(result.Score), StringComparer.Ordinal)
            .Select(group =>
            {
                var groupResults = group.ToList();
                return new EvaluationScoreBandSummary
                {
                    Band = group.Key,
                    Examples = groupResults.Count,
                    FailedExamples = groupResults.Count(result => !result.Passed),
                    AverageScore = AverageScore(groupResults),
                };
            })
            .OrderBy(summary => ScoreBandSortKey(summary.Band))
            .ToList();
    }

    private static IReadOnlyList<string> NormalizeTags(IReadOnlyList<string> tags)
    {
        var normalized = tags
            .Where(tag => !string.IsNullOrWhiteSpace(tag))
            .Select(tag => tag.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(tag => tag, StringComparer.OrdinalIgnoreCase)
            .ToList();

        return normalized.Count == 0 ? ["untagged"] : normalized;
    }

    private static string FailureReason(EvaluationExampleResult result)
    {
        return string.IsNullOrWhiteSpace(result.Notes)
            ? "score_below_threshold"
            : result.Notes.Trim();
    }

    private static string FormatFailureReason(string reason)
    {
        return string.IsNullOrWhiteSpace(reason)
            ? "unknown"
            : reason.Replace('_', ' ');
    }

    private static string ScoreBand(double score)
    {
        return score switch
        {
            < 50 => "0-49",
            < 70 => "50-69",
            < 85 => "70-84",
            _ => "85-100",
        };
    }

    private static int ScoreBandSortKey(string band)
    {
        return band switch
        {
            "0-49" => 0,
            "50-69" => 1,
            "70-84" => 2,
            "85-100" => 3,
            _ => 99,
        };
    }

    private static double AverageScore(IReadOnlyList<EvaluationExampleResult> results)
    {
        return results.Count == 0 ? 0.0 : Math.Round(results.Average(result => result.Score), 2);
    }

    private static string BuildEvaluationReportComparison(
        EvaluationReportHistoryItem selected,
        EvaluationReportHistoryItem comparison
    )
    {
        var selectedReport = selected.Report;
        var comparisonReport = comparison.Report;
        var lines = new List<string>
        {
            "Selected report:",
            selected.DisplayName,
            "Compared with:",
            comparison.DisplayName,
            "",
            $"Dataset: {selectedReport.Dataset} vs {comparisonReport.Dataset}",
            $"Model: {selectedReport.Model} vs {comparisonReport.Model}",
            $"Examples tested: {selectedReport.ExamplesTested} ({FormatSignedInt(selectedReport.ExamplesTested - comparisonReport.ExamplesTested)})",
            $"Average score: {selectedReport.AverageScore:0.##} ({FormatSignedDouble(selectedReport.AverageScore - comparisonReport.AverageScore)})",
            $"Failed examples: {selectedReport.FailedExamples} ({FormatSignedInt(selectedReport.FailedExamples - comparisonReport.FailedExamples)}; {FormatFailureTrend(selectedReport.FailedExamples - comparisonReport.FailedExamples)})",
        };

        if (selectedReport.AverageManualScore is not null || comparisonReport.AverageManualScore is not null)
        {
            lines.Add(
                $"Manual average: {FormatNullableScore(selectedReport.AverageManualScore)} vs {FormatNullableScore(comparisonReport.AverageManualScore)}"
            );
            if (selectedReport.AverageManualScore is not null && comparisonReport.AverageManualScore is not null)
            {
                lines[^1] += $" ({FormatSignedDouble(selectedReport.AverageManualScore.Value - comparisonReport.AverageManualScore.Value)})";
            }
        }

        AddWeakTagComparison(lines, selectedReport, comparisonReport);
        AddExampleResultComparison(lines, selectedReport, comparisonReport);

        return string.Join(Environment.NewLine, lines);
    }

    private static void AddWeakTagComparison(
        List<string> lines,
        EvaluationReport selectedReport,
        EvaluationReport comparisonReport
    )
    {
        var selectedTags = selectedReport.WeakTags
            .Where(tag => !string.IsNullOrWhiteSpace(tag))
            .Select(tag => tag.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        var comparisonTags = comparisonReport.WeakTags
            .Where(tag => !string.IsNullOrWhiteSpace(tag))
            .Select(tag => tag.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        var added = selectedTags
            .Except(comparisonTags, StringComparer.OrdinalIgnoreCase)
            .OrderBy(tag => tag, StringComparer.OrdinalIgnoreCase)
            .ToList();
        var cleared = comparisonTags
            .Except(selectedTags, StringComparer.OrdinalIgnoreCase)
            .OrderBy(tag => tag, StringComparer.OrdinalIgnoreCase)
            .ToList();

        lines.Add(
            $"Weak tags added: {(added.Count == 0 ? "none" : string.Join(", ", added))}"
        );
        lines.Add(
            $"Weak tags cleared: {(cleared.Count == 0 ? "none" : string.Join(", ", cleared))}"
        );
    }

    private static void AddExampleResultComparison(
        List<string> lines,
        EvaluationReport selectedReport,
        EvaluationReport comparisonReport
    )
    {
        var selectedById = BuildEvaluationResultMap(selectedReport.Results);
        var comparisonById = BuildEvaluationResultMap(comparisonReport.Results);
        var commonIds = selectedById.Keys
            .Intersect(comparisonById.Keys, StringComparer.Ordinal)
            .OrderBy(id => id, StringComparer.Ordinal)
            .ToList();

        lines.Add("");
        if (commonIds.Count == 0)
        {
            lines.Add("Common examples: none.");
            lines.Add($"Only in selected report: {selectedById.Count}");
            lines.Add($"Only in comparison report: {comparisonById.Count}");
            return;
        }

        var improvedToPass = 0;
        var regressedToFail = 0;
        var totalDelta = 0.0;
        var scoreChanges = new List<(string ExampleId, double Before, double After, double Delta)>();

        foreach (var exampleId in commonIds)
        {
            var selectedResult = selectedById[exampleId];
            var comparisonResult = comparisonById[exampleId];
            if (!comparisonResult.Passed && selectedResult.Passed)
            {
                improvedToPass++;
            }

            if (comparisonResult.Passed && !selectedResult.Passed)
            {
                regressedToFail++;
            }

            var delta = selectedResult.Score - comparisonResult.Score;
            totalDelta += delta;
            scoreChanges.Add((exampleId, comparisonResult.Score, selectedResult.Score, delta));
        }

        lines.Add(
            $"Common examples: {commonIds.Count}; now passing: {improvedToPass}; regressed: {regressedToFail}; average row score delta: {FormatSignedDouble(totalDelta / commonIds.Count)}"
        );
        lines.Add($"Only in selected report: {selectedById.Count - commonIds.Count}");
        lines.Add($"Only in comparison report: {comparisonById.Count - commonIds.Count}");

        foreach (var change in scoreChanges
            .OrderByDescending(change => Math.Abs(change.Delta))
            .ThenBy(change => change.ExampleId, StringComparer.Ordinal)
            .Take(5))
        {
            lines.Add(
                $"- {change.ExampleId}: {change.Before:0.##} -> {change.After:0.##} ({FormatSignedDouble(change.Delta)})"
            );
        }
    }

    private static Dictionary<string, EvaluationExampleResult> BuildEvaluationResultMap(
        IEnumerable<EvaluationExampleResult> results
    )
    {
        var map = new Dictionary<string, EvaluationExampleResult>(StringComparer.Ordinal);
        foreach (var result in results)
        {
            if (!string.IsNullOrWhiteSpace(result.ExampleId) && !map.ContainsKey(result.ExampleId))
            {
                map.Add(result.ExampleId, result);
            }
        }

        return map;
    }

    private static string FormatSignedDouble(double value)
    {
        return value >= 0 ? $"+{value:0.##}" : $"{value:0.##}";
    }

    private static string FormatSignedInt(int value)
    {
        return value >= 0 ? $"+{value}" : value.ToString(CultureInfo.InvariantCulture);
    }

    private static string FormatNullableScore(double? value)
    {
        return value is null ? "none" : value.Value.ToString("0.##", CultureInfo.InvariantCulture);
    }

    private static string FormatFailureTrend(int failedDelta)
    {
        return failedDelta switch
        {
            < 0 => "improved",
            > 0 => "more failures",
            _ => "unchanged",
        };
    }

    private static string BuildAiAssistDiffSummary(AiAssistReviewQueueItem item)
    {
        if (string.IsNullOrWhiteSpace(item.SuggestedJsonl))
        {
            return "No suggested JSONL to compare.";
        }

        if (string.IsNullOrWhiteSpace(item.SourceDraft))
        {
            return "Suggested JSONL is available, but no source draft was recorded.";
        }

        var source = item.SourceDraft.TrimEnd();
        var suggestion = item.SuggestedJsonl.TrimEnd();
        if (string.Equals(source, suggestion, StringComparison.Ordinal))
        {
            return "Suggestion matches the source draft exactly.";
        }

        var delta = suggestion.Length - source.Length;
        var deltaText = delta >= 0 ? $"+{delta}" : delta.ToString();
        return $"Source lines: {CountLines(source)}; suggested lines: {CountLines(suggestion)}; character delta: {deltaText}.";
    }

    private static int CountLines(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return 0;
        }

        return text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n').Length;
    }

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

    private static string FormatRowNumbers(IEnumerable<int> rowNumbers)
    {
        var rows = rowNumbers
            .Where(rowNumber => rowNumber > 0)
            .Distinct()
            .Order()
            .ToList();
        return rows.Count == 0 ? "none" : string.Join(", ", rows);
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
            EvaluationBackend = backend;
            EvaluationModel = model;
            EvaluationBaseUrl = baseUrl;
            EvaluationTimeoutSeconds = timeoutSeconds.ToString();
            return;
        }

        AiAssistBackend = backend;
        AiAssistModel = model;
        AiAssistBaseUrl = baseUrl;
        AiAssistTimeoutSeconds = timeoutSeconds.ToString();
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
