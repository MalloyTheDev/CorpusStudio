using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;

using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels;

public sealed class MainWindowViewModel : INotifyPropertyChanged
{
    private string _activeProjectTitle = "New Dataset Project";
    private string? _activeProjectPath;
    private string _activeSchemaId = "instruction";
    private string _activeSchemaDescription =
        "Choose a schema, write examples, validate rows, and export model-ready JSONL.";
    private string _validationSummary = "Create a project to start validation.";
    private string _qualitySummary = "Create or select a project to run quality checks.";
    private string _gateSummary = "Run gates to check whether this dataset may move forward.";
    private string _arenaPromptsInput = string.Empty;
    private string _arenaModelsInput = string.Empty;
    private string _arenaJudgeModelInput = string.Empty;
    private string _arenaSummary =
        "Enter prompts (one per line) and models (comma or newline separated), then Run Arena.";
    private string _providerPolicySummary =
        "Provider generation policy: refresh to see which providers may create trainable rows.";
    private string _qualityHistorySummary = "Quality history appears after quality checks run.";
    private string _qualityTriageSummary = "Synthetic quality issues appear here after quality checks run.";
    private string _splitSummary = "Create or select a project to generate train, validation, and test splits.";
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
    private string _artifactSummary = "Register a model artifact from a completed run, then keep or reject it.";
    private string _artifactDetail = "Select an artifact, then View card or Keep (Keep is promote-gated).";
    private ArtifactDisplayItem? _selectedModelArtifact;
    private string _datasetVersionSummary =
        "Refresh to see dataset versions, or capture the current dataset as a version.";
    private string _datasetVersionDetail =
        "Select a version and View card to see its lineage (runs, artifacts, evals) and integrity.";
    private string _datasetVersionLabel = string.Empty;
    private DatasetVersionDisplayItem? _selectedDatasetVersion;
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
    private string _splitTrainPercent = "90";
    private string _splitValidationPercent = "5";
    private string _splitSeed = "42";
    private string _selectedImportQuarantineDetail =
        "Rejected import rows appear here after a mixed import.";
    private string _settingsSummary = "Settings load when the app starts.";
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
        private set => SetField(ref _activeSchemaId, value);
    }

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

    private string _draftText =
        "{\n  \"instruction\": \"Explain what a variable is.\",\n  \"input\": \"\",\n  \"output\": \"A variable stores a value.\"\n}";

    public string DraftText
    {
        get => _draftText;
        set => SetField(ref _draftText, value);
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

    public string GateSummary
    {
        get => _gateSummary;
        private set => SetField(ref _gateSummary, value);
    }

    public string ProviderPolicySummary
    {
        get => _providerPolicySummary;
        private set => SetField(ref _providerPolicySummary, value);
    }

    public string ArenaPromptsInput
    {
        get => _arenaPromptsInput;
        set => SetField(ref _arenaPromptsInput, value);
    }

    public string ArenaModelsInput
    {
        get => _arenaModelsInput;
        set => SetField(ref _arenaModelsInput, value);
    }

    public string ArenaJudgeModelInput
    {
        get => _arenaJudgeModelInput;
        set => SetField(ref _arenaJudgeModelInput, value);
    }

    public string ArenaSummary
    {
        get => _arenaSummary;
        private set => SetField(ref _arenaSummary, value);
    }

    /// <summary>Split a comma/newline-separated model list into a trimmed,
    /// de-duplicated (case-insensitive, order-preserving) list.</summary>
    public static IReadOnlyList<string> ParseModelList(string text)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var models = new List<string>();
        foreach (var token in (text ?? string.Empty).Split(
                     [',', '\n', '\r'],
                     StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (seen.Add(token))
            {
                models.Add(token);
            }
        }

        return models;
    }

    public void SetArenaInProgress()
    {
        ArenaSummary = "Running arena...";
    }

    public void SetArenaError(string message)
    {
        ArenaSummary = $"Arena could not run.{Environment.NewLine}{message}";
    }

    /// <summary>Format an arena report: per-model win/score summary, then each
    /// prompt's side-by-side responses with the judge's winner marked.</summary>
    public void ApplyArenaReport(ArenaReport report)
    {
        var judged = !string.IsNullOrWhiteSpace(report.JudgeModel);
        var judgmentByPrompt = report.Judgments
            .GroupBy(judgment => judgment.PromptId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Last(), StringComparer.Ordinal);

        var header = $"Arena: {report.Models.Count} model(s) × {report.PromptCount} prompt(s)";
        if (judged)
        {
            header += $"   judge: {report.JudgeModel}";
        }

        var lines = new List<string> { header, string.Empty, "Models:" };
        foreach (var summary in report.ModelSummaries)
        {
            var parts = new List<string> { $"{summary.WinCount} win(s)" };
            if (summary.AverageJudgeScore is { } score)
            {
                parts.Add($"avg {score:0.#}");
            }
            if (summary.EmptyResponseCount > 0)
            {
                parts.Add($"{summary.EmptyResponseCount} empty");
            }
            lines.Add($"  {summary.Model} — {string.Join(", ", parts)}");
        }

        foreach (var prompt in report.Prompts)
        {
            lines.Add(string.Empty);
            lines.Add($"── {prompt.Id}: {SingleLine(prompt.Prompt)}");
            judgmentByPrompt.TryGetValue(prompt.Id, out var judgment);
            foreach (var response in report.Responses.Where(r => r.PromptId == prompt.Id))
            {
                var win = judgment is not null && judgment.Winner == response.Model ? "🏆 " : "   ";
                lines.Add($"  {win}{response.Model}:");
                lines.Add(IndentBlock(string.IsNullOrWhiteSpace(response.Text) ? "(empty response)" : response.Text));
            }

            if (judged && judgment is not null)
            {
                lines.Add(judgment.Parsed
                    ? $"  judge: {judgment.Winner} — {SingleLine(judgment.Rationale)}"
                    : "  judge: (could not parse judge output)");
            }
        }

        ArenaSummary = string.Join(Environment.NewLine, lines);
    }

    private static string SingleLine(string text)
    {
        return string.Join(" ", (text ?? string.Empty).Split(
            ['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
    }

    private static string IndentBlock(string text)
    {
        var body = (text ?? string.Empty).Replace("\r\n", "\n").Split('\n');
        return string.Join(Environment.NewLine, body.Select(line => "       " + line));
    }

    public void SetProviderPolicyError(string message)
    {
        ProviderPolicySummary = $"Provider policy action failed.{Environment.NewLine}{message}";
    }

    /// <summary>Format the provider generation policy (who may create trainable rows).</summary>
    public void ApplyProviderPolicies(IReadOnlyList<ProviderPolicyItem> policies)
    {
        var lines = new List<string>
        {
            "Provider generation policy (who may create trainable rows):",
            string.Empty,
        };

        foreach (var policy in policies)
        {
            var name = string.IsNullOrWhiteSpace(policy.DisplayName) ? policy.ProviderId : policy.DisplayName;
            if (policy.GenerationAllowed)
            {
                lines.Add($"✅ {name} ({policy.ProviderKind}) — generation ALLOWED (approved)");
            }
            else if (policy.UserApprovedGeneration)
            {
                // Approved but still blocked (frontier provider): make that explicit.
                lines.Add($"⛔ {name} ({policy.ProviderKind}) — approval ignored; evaluator-only");
            }
            else
            {
                lines.Add($"⛔ {name} ({policy.ProviderKind}) — generation blocked (evaluator-only or unapproved)");
            }
        }

        lines.Add(string.Empty);
        lines.Add("Approve a local model below to let AI Assist use it for generation (still review-required).");
        ProviderPolicySummary = string.Join(Environment.NewLine, lines);
    }

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
    }

    public string QualityHistorySummary
    {
        get => _qualityHistorySummary;
        private set => SetField(ref _qualityHistorySummary, value);
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


    public string SplitSummary
    {
        get => _splitSummary;
        private set => SetField(ref _splitSummary, value);
    }

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

    public ObservableCollection<ArtifactDisplayItem> ModelArtifacts { get; } = [];

    public ArtifactDisplayItem? SelectedModelArtifact
    {
        get => _selectedModelArtifact;
        set => SetField(ref _selectedModelArtifact, value);
    }

    public string ArtifactSummary
    {
        get => _artifactSummary;
        private set => SetField(ref _artifactSummary, value);
    }

    public string ArtifactDetail
    {
        get => _artifactDetail;
        private set => SetField(ref _artifactDetail, value);
    }

    public void SetArtifactError(string message)
    {
        ArtifactSummary = $"Artifact action failed.{Environment.NewLine}{message}";
    }

    /// <summary>Set the detail pane (weight card markdown or a gate verdict).</summary>
    public void SetArtifactDetail(string text)
    {
        ArtifactDetail = text;
    }

    /// <summary>Format a promote-gate verdict for the detail pane. Returns whether
    /// the keep is allowed (block => refused).</summary>
    public bool ApplyPromoteGate(GateReport report)
    {
        // Drive the decision from the canonical OverallStatus (worst of results),
        // and fail closed on anything that is not an explicit pass/warn.
        string ReasonFor(string status) =>
            report.Results.FirstOrDefault(r => r.Status == status)?.Message ?? string.Empty;

        switch (report.OverallStatus)
        {
            case "block":
                var blockReason = ReasonFor("block");
                ArtifactDetail = "⛔ Keep blocked by the promote gate:" + Environment.NewLine
                    + (string.IsNullOrEmpty(blockReason) ? "the artifact did not pass promotion." : blockReason);
                return false;
            case "warn":
                var warnReason = ReasonFor("warn");
                ArtifactDetail = "⚠ Kept, but the promote gate warned:" + Environment.NewLine
                    + (string.IsNullOrEmpty(warnReason) ? "review the weight card." : warnReason)
                    + Environment.NewLine + "View the weight card before relying on it.";
                return true;
            case "pass":
                ArtifactDetail = "✅ Kept — the promote gate passed (integrity ok, no regression).";
                return true;
            default:
                ArtifactDetail = $"⛔ Keep blocked: unrecognized gate status '{report.OverallStatus}'.";
                return false;
        }
    }

    /// <summary>Refresh the artifact list + a one-line summary (kept / flagged counts).</summary>
    public void ApplyArtifacts(IReadOnlyList<ArtifactDisplayItem> items)
    {
        var selectedId = SelectedModelArtifact?.Record.ArtifactId;
        ModelArtifacts.Clear();
        foreach (var item in items)
        {
            ModelArtifacts.Add(item);
        }
        SelectedModelArtifact = ModelArtifacts.FirstOrDefault(i => i.Record.ArtifactId == selectedId);

        if (items.Count == 0)
        {
            ArtifactSummary = "No artifacts registered yet. Register one from a completed run.";
            return;
        }
        var kept = items.Count(i => i.Record.Status == "kept");
        var flagged = items.Count(i => i.Integrity != "ok");
        ArtifactSummary = $"{items.Count} artifact(s): {kept} kept, {flagged} with integrity issues (missing/modified).";
    }

    // --- Dataset version history (v1.0) -------------------------------------

    public ObservableCollection<DatasetVersionDisplayItem> DatasetVersions { get; } = [];

    public DatasetVersionDisplayItem? SelectedDatasetVersion
    {
        get => _selectedDatasetVersion;
        set => SetField(ref _selectedDatasetVersion, value);
    }

    public string DatasetVersionSummary
    {
        get => _datasetVersionSummary;
        private set => SetField(ref _datasetVersionSummary, value);
    }

    public string DatasetVersionDetail
    {
        get => _datasetVersionDetail;
        private set => SetField(ref _datasetVersionDetail, value);
    }

    /// <summary>Optional label typed before capturing a version (two-way bound).</summary>
    public string DatasetVersionLabel
    {
        get => _datasetVersionLabel;
        set => SetField(ref _datasetVersionLabel, value);
    }

    public void SetDatasetVersionError(string message)
    {
        DatasetVersionSummary = $"Dataset version action failed.{Environment.NewLine}{message}";
    }

    /// <summary>Set the detail pane (a rendered version card or a capture confirmation).</summary>
    public void SetDatasetVersionDetail(string text)
    {
        DatasetVersionDetail = text;
    }

    /// <summary>Honest capture confirmation. A record with no content fingerprint
    /// (examples.jsonl was missing/unreadable) can never be verified against the
    /// dataset — the engine annotates it 'unreadable' forever — so it must NOT read
    /// as a green "captured" success (the ✅ vocabulary the 'matches' badge uses).</summary>
    public static string FormatCaptureConfirmation(DatasetVersionRecord record)
    {
        if (string.IsNullOrEmpty(record.ContentFingerprint))
        {
            return $"⛔ Recorded version {record.VersionId}, but examples.jsonl was missing or "
                + "unreadable — no fingerprint was captured, so this version's integrity can never be verified.";
        }
        return $"✅ Captured version {record.VersionId} ({record.RowCount} rows).";
    }

    /// <summary>Honest confirmation text for an in-place restore. It overwrites the
    /// current dataset, so it names both row counts, the undo safety net, and the
    /// canonical caveat. Pure/testable.</summary>
    public static string BuildRestoreConfirmation(DatasetVersionDisplayItem version, int currentRowCount)
    {
        return $"Overwrite the current dataset ({currentRowCount} row(s)) with version "
            + $"{version.Record.VersionId} ({version.Record.RowCount} row(s))?"
            + Environment.NewLine + Environment.NewLine
            + "Your current dataset is captured as a version first (a readable dataset becomes a "
            + "restorable undo point); if it cannot be captured for undo, the restore is refused."
            + Environment.NewLine + Environment.NewLine
            + "Rows are reconstructed in canonical form (key order may change).";
    }

    /// <summary>Label for the undo version captured just before a restore.</summary>
    public static string BuildRestoreUndoLabel(DatasetVersionDisplayItem version)
    {
        return $"before restore of {version.Record.VersionId}";
    }

    /// <summary>Report a completed in-place restore honestly in the detail pane.</summary>
    public void ApplyRestoreResult(RestoreResult result)
    {
        var verifiedNote = result.Verified
            ? "verified — fingerprint matches, semantically identical to the recorded version"
            : (result.VerifySkipped ? "unverified" : "written");
        SetDatasetVersionDetail(
            $"✅ Restored version {result.VersionId}: {result.RowsWritten} row(s) [{verifiedNote}]. "
            + "Your previous dataset was captured as an undo version (restore it to revert). "
            + "Rows are in canonical form (key order may be normalized).");
    }

    /// <summary>Refresh the version list (newest first) + a one-line integrity summary.
    /// Selection is preserved by version_id across refreshes.</summary>
    public void ApplyDatasetVersions(IReadOnlyList<DatasetVersionDisplayItem> items)
    {
        var selectedId = SelectedDatasetVersion?.Record.VersionId;
        DatasetVersions.Clear();
        foreach (var item in items)
        {
            DatasetVersions.Add(item);
        }
        SelectedDatasetVersion = DatasetVersions.FirstOrDefault(i => i.Record.VersionId == selectedId);

        if (items.Count == 0)
        {
            DatasetVersionSummary = "No versions captured yet. Capture the current dataset to start a history.";
            return;
        }
        var matches = items.Count(i => i.Integrity == "matches");
        var drifted = items.Count(i => i.Integrity == "drifted");
        var unreadable = items.Count(i => i.Integrity == "unreadable");
        DatasetVersionSummary =
            $"{items.Count} version(s): {matches} matching the current dataset, {drifted} drifted, {unreadable} unverifiable.";
    }

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

    public string SplitTrainPercent
    {
        get => _splitTrainPercent;
        set => SetField(ref _splitTrainPercent, value);
    }

    public string SplitValidationPercent
    {
        get => _splitValidationPercent;
        set => SetField(ref _splitValidationPercent, value);
    }

    public string SplitSeed
    {
        get => _splitSeed;
        set => SetField(ref _splitSeed, value);
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

    public string SettingsSummary
    {
        get => _settingsSummary;
        private set => SetField(ref _settingsSummary, value);
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

    public void ApplyNewProjectTemplate(string exampleText)
    {
        if (string.IsNullOrWhiteSpace(exampleText))
        {
            return;
        }

        DraftText = exampleText;
        ValidationSummary = "Loaded the schema's example row. Edit the values, then Save Example.";
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

    public void SetSettings(DesktopSettings settings)
    {
        SettingsSummary = string.Join(
            Environment.NewLine,
            [
                $"Repository: {settings.RepositoryRoot}",
                $"Engine: {settings.EngineDirectory}",
                $"Python: {settings.PythonExecutable}",
                $"Projects: {settings.ProjectDirectory}",
                $"Exports: {settings.ExportDirectory}",
            ]
        );
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
        DraftText = BuildDraftTemplate(schemaId);
    }

    public void SelectProject(DatasetProjectListItem project, string? schemaName = null)
    {
        SelectedProject = project;
        ActiveProjectTitle = project.Name;
        ActiveProjectPath = project.ProjectPath;
        ActiveSchemaId = project.SchemaId;
        ActiveSchemaDescription = $"{schemaName ?? project.SchemaId} project. Ready for examples.";
        ApplyAiAssistActionPresets(project.SchemaId);
        ApplySplitSettings(project.Project.SplitSettings ?? SplitSettings.Default);
        ApplyLabSettings(project.Project.LabSettings ?? LabBackendSettings.Default);
        ValidationSummary = "No validation has run yet.";
        ClearValidationIssues();
        QualitySummary = "Quality checks will appear after examples are added.";
        QualityHistorySummary = "Quality history appears after quality checks run.";
        QualityTriageSummary = "Synthetic quality issues appear here after quality checks run.";
        SyntheticPatternIssues.Clear();
        SelectedSyntheticPatternIssue = null;
        SplitSummary = "Generate splits after examples are saved.";
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
        TrainingFormat = project.SchemaId;
        TrainingSummary = "Generate a training config after validation, splits, and evaluation checks.";
        TrainingConfigPreview = "Training config preview appears here.";
        Examples.Clear();
        ImportQuarantineItems.Clear();
        SelectedImportQuarantineItem = null;
        SelectedExample = null;
        _allPreferenceReviewItems.Clear();
        PreferenceReviewItems.Clear();
        SelectedPreferenceReviewItem = null;
        PreferenceContrastFilter = "All";
        ClearPreferenceReview();
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
    }

    public void SetQualityInProgress()
    {
        QualitySummary = "Running quality checks...";
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

        var lines = new List<string>
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

        QualitySummary = string.Join(Environment.NewLine, lines);
        SetSyntheticPatternIssues(report.SyntheticPatternIssues);

        ApplyQualityHistory(history ?? []);
    }

    public void SetQualityError(string message)
    {
        QualitySummary = $"Quality checks could not run.{Environment.NewLine}{message}";
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
        DraftText = example.Json;
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

        DraftText = BuildJsonArrayDraft(affectedRows.Select(row => row.Json));
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

        DraftText = BuildEvaluationFailureDraft(SelectedEvaluationExampleResult, ActiveSchemaId);
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
        DraftText = example.Json;
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

        DraftText = SelectedPreferenceReviewItem.Json;
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

        DraftText = BuildJsonArrayDraft(items.Select(item => item.Json));
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

    public void SetSplitInProgress(double trainRatio, double validationRatio, int seed)
    {
        var testRatio = 1 - trainRatio - validationRatio;
        SplitSummary = string.Join(
            Environment.NewLine,
            [
                "Generating train, validation, and test splits...",
                $"Train: {FormatPercent(trainRatio)}",
                $"Validation: {FormatPercent(validationRatio)}",
                $"Test: {FormatPercent(testRatio)}",
                $"Seed: {seed}",
            ]
        );
    }

    public void ApplySplitSettings(SplitSettings settings)
    {
        SplitTrainPercent = settings.TrainPercentText;
        SplitValidationPercent = settings.ValidationPercentText;
        SplitSeed = settings.Seed.ToString();
    }

    public void ApplySplitReport(SplitReport report)
    {
        var lines = new List<string>
        {
            $"Train: {report.Train}",
            $"Validation: {report.Validation}",
            $"Test: {report.Test}",
            $"Ratios: train {FormatPercent(report.TrainRatio)}, validation {FormatPercent(report.ValidationRatio)}, test {FormatPercent(report.TestRatio)}",
            $"Seed: {report.Seed}",
            $"Rows shared across splits: {report.RowsSharedAcrossSplits}"
                + (report.RowsSharedAcrossSplits > 0 ? " (train/test leakage)" : ""),
            $"Output: {report.OutputDirectory}",
        };

        if (report.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(report.Warnings.Select(warning => $"- {warning}"));
        }

        SplitSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetSplitError(string message)
    {
        SplitSummary = $"Splits could not be generated.{Environment.NewLine}{message}";
        ReportError(message);
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
        AiAssistSourceDraftText = "Current draft is being sent to AI Assist.";
        AiAssistSuggestedJsonlText = "Waiting for suggested JSONL.";
        AiAssistDiffSummary = "Comparison will appear after the review is queued.";
    }

    public void ApplyAiAssistRunResult(AiAssistRunResult result)
    {
        _aiAssistSuggestionJsonl = result.SuggestedJsonl;
        var suggestionStatus = string.IsNullOrWhiteSpace(result.SuggestedJsonl)
            ? "none"
            : "available for review";
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                $"Action: {result.Action}",
                $"Model: {result.Model}",
                $"Review state: {result.ReviewState}",
                $"Review required: {(result.ReviewRequired ? "yes" : "no")}",
                $"Suggested JSONL: {suggestionStatus}",
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

        DraftText = SelectedAiAssistRewriteBatch.SourceDraft;
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
        DraftText = example.Json;
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
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                $"Action: {item.Action}",
                $"Model: {item.Model}",
                $"Review state: {item.ReviewState}",
                $"Suggested JSONL: {(string.IsNullOrWhiteSpace(item.SuggestedJsonl) ? "none" : "available for review")}",
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

        DraftText = suggestionJsonl.TrimEnd();
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

    public void RetrySelectedImportQuarantineItem()
    {
        if (SelectedImportQuarantineItem is not null)
        {
            DraftText = SelectedImportQuarantineItem.Raw;
        }
    }

    private void ApplyQualityHistory(IReadOnlyList<QualityHistoryEntry> history)
    {
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

    private static string FormatPercent(double ratio)
    {
        return $"{ratio * 100:0.##}%";
    }

    private static string FormatManualScoreSummary(EvaluationReport report)
    {
        if (report.ManuallyScoredExamples == 0 || report.AverageManualScore is null)
        {
            return "none";
        }

        return $"{report.ManuallyScoredExamples} example(s), average {report.AverageManualScore:0.##}";
    }

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
