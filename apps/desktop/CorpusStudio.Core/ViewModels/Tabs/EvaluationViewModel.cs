using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Windows.Input;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Evaluation tab core view-model (backend-cluster slice 3, PR 3b). Behaviour moved
/// verbatim from the shell (<c>MainWindowViewModel</c>) — the run + report panes, the per-example result
/// review queue (filters + saved failure views + manual score/notes), and the report-history + comparison.
/// Reads the connection child for the run's backend/model; a run failure surfaces via
/// <see cref="ErrorReported"/> (the shell forwards it to its error banner). Honesty invariants intact:
/// a verdict reflects structure/thresholds not quality, and per-example results stay truthful.</summary>
public sealed class EvaluationViewModel : ViewModelBase, IEvaluationViewModel
{
    private readonly IEvaluationConnectionViewModel _connection;

    /// <summary>Raised when an evaluation run fails; the shell forwards it to its shared error banner.</summary>
    public event Action<string>? ErrorReported;

    public EvaluationViewModel(IEvaluationConnectionViewModel connection)
    {
        _connection = connection;
        CompareReportsCommand = new RelayCommand(() => CompareSelectedEvaluationReports());
        ApplyFailureFilterCommand = new RelayCommand(ApplySelectedFailureFilter);
        ShowAllResultsCommand = new RelayCommand(() => EvaluationResultFilter = "All");
        ShowPassedResultsCommand = new RelayCommand(() => EvaluationResultFilter = "Passed");
        ShowFailedResultsCommand = new RelayCommand(() => EvaluationResultFilter = "Failed");
    }

    /// <summary>Compare the two selected saved reports. Bound as a command so both heads share the action.</summary>
    public ICommand CompareReportsCommand { get; }

    /// <summary>Apply the selected saved failure view (guarding on a selection). Command target.</summary>
    public ICommand ApplyFailureFilterCommand { get; }

    /// <summary>Segmented-control commands (Evaluation screen): set the results status filter to All /
    /// Passed / Failed. Thin wrappers over <see cref="EvaluationResultFilter"/> so the toggle pills bind a
    /// command (the parameterless <see cref="RelayCommand"/> ignores CommandParameter).</summary>
    public ICommand ShowAllResultsCommand { get; }

    public ICommand ShowPassedResultsCommand { get; }

    public ICommand ShowFailedResultsCommand { get; }

    /// <summary>Apply the selected saved failure view; reports if none is selected. Command target
    /// (the logic moved here from the desktop code-behind so both heads can bind the command).</summary>
    public void ApplySelectedFailureFilter()
    {
        if (SelectedEvaluationFailureFilter is null)
        {
            SetEvaluationFailureFilterError("Select a saved failure filter before applying it.");
            return;
        }

        ApplyEvaluationFailureFilter(SelectedEvaluationFailureFilter);
    }

    private string _evaluationLimit = "10";

    private string _evaluationScoreThreshold = "70";

    private string _evaluationSummary =
        "Run a local model against this project's saved examples.";

    private string _evaluationReportJson = "Evaluation reports appear here after a run.";

    // Live per-example progress during a run (#191): the engine streams '[k/N] evaluated' to stderr.
    private bool _isEvaluationInProgress;
    private double _evaluationProgressPercent;
    private string _evaluationProgressText = string.Empty;

    private string _selectedEvaluationExampleDetail =
        "Per-example evaluation results appear here after a run or report reload.";

    private string _evaluationResultsSummary =
        "Evaluation example review queue appears after a run or report reload.";

    private string _evaluationResultFilter = "All";

    // Discrete KPI display strings (Evaluation screen stat-card row): computed from the report on each
    // run so the cards bind ready-to-render text and stay hidden (HasEvaluationReport) pre-run.
    private bool _hasEvaluationReport;
    private string _averageScoreDisplay = "—";
    private string _passRateDisplay = "—";
    private string _passRateDetail = string.Empty;
    private string _evaluatedDisplay = "—";
    private string _metricDisplay = "—";

    // Pass/fail totals over the full (unfiltered) result set — the segmented-control pill counts.
    private int _evaluationPassCount;
    private int _evaluationFailCount;

    private string _evaluationTagFilter = "All";

    private string _evaluationFailureReasonFilter = "All";

    private string _evaluationScoreBandFilter = "All";

    private string _evaluationFailureFilterName = "Failure View";

    private string _evaluationFailureFilterSummary =
        "Save the active status, tag, failure-reason, and score-band filters as a named view.";

    private string _evaluationManualScore = string.Empty;

    private string _evaluationManualNotes = string.Empty;

    private string _evaluationReviewSummary = "Select an evaluation result to add a manual score or note.";

    private string _evaluationComparisonSummary =
        "Select two saved evaluation reports to compare score, failure, tag, and row-level changes.";

    private EvaluationReportHistoryItem? _selectedEvaluationReportHistoryItem;

    private EvaluationReportHistoryItem? _secondaryEvaluationReportHistoryItem;

    private EvaluationExampleResult? _selectedEvaluationExampleResult;

    private EvaluationFailureFilter? _selectedEvaluationFailureFilter;

    private readonly List<EvaluationExampleResult> _allEvaluationResults = [];

    public ObservableCollection<EvaluationReportHistoryItem> EvaluationReportHistory { get; } = [];

    public ObservableCollection<EvaluationExampleResult> EvaluationResults { get; } = [];

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

    public string EvaluationSummary
    {
        get => _evaluationSummary;
        // public set: the shell's backend health-check operations write it.
        set => SetField(ref _evaluationSummary, value);
    }

    public string EvaluationReportJson
    {
        get => _evaluationReportJson;
        private set => SetField(ref _evaluationReportJson, value);
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
                RaiseResultFilterActiveStates();
                RebuildEvaluationResults();
            }
        }
    }

    /// <summary>True once a run/report has produced examples — gates the KPI stat-card row (hidden pre-run
    /// so no faked numbers show).</summary>
    public bool HasEvaluationReport
    {
        get => _hasEvaluationReport;
        private set => SetField(ref _hasEvaluationReport, value);
    }

    /// <summary>Average automatic score, formatted for the "Average score" KPI card.</summary>
    public string AverageScoreDisplay
    {
        get => _averageScoreDisplay;
        private set => SetField(ref _averageScoreDisplay, value);
    }

    /// <summary>Pass rate as a whole-percent string ("74%") for the "Pass rate" KPI card.</summary>
    public string PassRateDisplay
    {
        get => _passRateDisplay;
        private set => SetField(ref _passRateDisplay, value);
    }

    /// <summary>Pass-rate sub-line ("14 / 19 ≥ 70") naming the passing count, total, and run threshold.</summary>
    public string PassRateDetail
    {
        get => _passRateDetail;
        private set => SetField(ref _passRateDetail, value);
    }

    /// <summary>Examples-tested count for the "Evaluated" KPI card.</summary>
    public string EvaluatedDisplay
    {
        get => _evaluatedDisplay;
        private set => SetField(ref _evaluatedDisplay, value);
    }

    /// <summary>Human-readable scorer name ("keyword overlap" / "LLM judge") for the "Metric" KPI card.</summary>
    public string MetricDisplay
    {
        get => _metricDisplay;
        private set => SetField(ref _metricDisplay, value);
    }

    /// <summary>Passing-example count over the full result set — the segmented "Pass N" pill.</summary>
    public int EvaluationPassCount
    {
        get => _evaluationPassCount;
        private set => SetField(ref _evaluationPassCount, value);
    }

    /// <summary>Failing-example count over the full result set — the segmented "Fail N" pill.</summary>
    public int EvaluationFailCount
    {
        get => _evaluationFailCount;
        private set => SetField(ref _evaluationFailCount, value);
    }

    /// <summary>Segmented-control active states — true when the results filter is the matching status,
    /// so the active pill highlights (Classes.active). Re-raised whenever the filter changes.</summary>
    public bool IsAllResultsFilterActive =>
        string.Equals(EvaluationResultFilter, "All", StringComparison.Ordinal);

    public bool IsPassedResultsFilterActive =>
        string.Equals(EvaluationResultFilter, "Passed", StringComparison.Ordinal);

    public bool IsFailedResultsFilterActive =>
        string.Equals(EvaluationResultFilter, "Failed", StringComparison.Ordinal);

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
        // public set: the shell's eval-failure review/edit bridges (which drive AI Assist / Writing
        // Studio) report their status here.
        set => SetField(ref _evaluationReviewSummary, value);
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

    /// <summary>True while a run streams progress — drives the progress bar's visibility.</summary>
    public bool IsEvaluationInProgress
    {
        get => _isEvaluationInProgress;
        private set => SetField(ref _isEvaluationInProgress, value);
    }

    /// <summary>0–100 completion for the progress bar (indeterminate until the first update).</summary>
    public double EvaluationProgressPercent
    {
        get => _evaluationProgressPercent;
        private set => SetField(ref _evaluationProgressPercent, value);
    }

    /// <summary>"Evaluating k/N…" caption shown beside the progress bar.</summary>
    public string EvaluationProgressText
    {
        get => _evaluationProgressText;
        private set => SetField(ref _evaluationProgressText, value);
    }

    /// <summary>Start showing the progress bar (before any per-example update arrives).</summary>
    public void BeginEvaluationProgress()
    {
        IsEvaluationInProgress = true;
        EvaluationProgressPercent = 0;
        EvaluationProgressText = "Starting evaluation…";
    }

    /// <summary>Apply one per-example progress update ("k of N evaluated"). Ignores a non-positive total,
    /// and — since progress lines are marshaled asynchronously — ignores any update that arrives after the
    /// run cleared (so a late line can't re-show the bar with stale data). <see cref="BeginEvaluationProgress"/>
    /// opens the window.</summary>
    public void SetEvaluationProgress(int completed, int total)
    {
        if (total <= 0 || !IsEvaluationInProgress)
        {
            return;
        }

        EvaluationProgressPercent = System.Math.Clamp(completed * 100.0 / total, 0, 100);
        EvaluationProgressText = $"Evaluating {completed}/{total}…";
    }

    /// <summary>Hide the progress bar (the run finished or failed).</summary>
    public void ClearEvaluationProgress()
    {
        IsEvaluationInProgress = false;
        EvaluationProgressText = string.Empty;
    }

    public void SetEvaluationInProgress()
    {
        EvaluationSummary = string.Join(
            Environment.NewLine,
            [
                "Running evaluation...",
                $"Backend: {_connection.EvaluationBackend}",
                $"Model: {_connection.EvaluationModel}",
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
                $"Backend: {_connection.EvaluationBackend}",
                $"Model: {_connection.EvaluationModel}",
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
                $"Backend: {_connection.EvaluationBackend}",
                $"Model: {_connection.EvaluationModel}",
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
                $"Backend: {_connection.EvaluationBackend}",
                $"Model: {_connection.EvaluationModel}",
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
        UpdateEvaluationKpis(result.Report);
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
        _connection.EvaluationBackend = settings.Backend;
        _connection.EvaluationModel = settings.Model;
        _connection.EvaluationBaseUrl = settings.BaseUrl ?? string.Empty;
        EvaluationLimit = settings.Limit?.ToString(CultureInfo.InvariantCulture) ?? string.Empty;
        EvaluationScoreThreshold = settings.ScoreThreshold.ToString("0.##", CultureInfo.InvariantCulture);
        _connection.EvaluationTimeoutSeconds = settings.TimeoutSeconds.ToString(CultureInfo.InvariantCulture);
    }

    public void SetEvaluationReviewError(string message)
    {
        EvaluationReviewSummary = $"Manual evaluation review could not be saved.{Environment.NewLine}{message}";
    }

    public void SetEvaluationError(string message)
    {
        EvaluationSummary = $"Evaluation could not run.{Environment.NewLine}{message}";
        EvaluationReportJson = "No evaluation report was produced.";
        ErrorReported?.Invoke(message);
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
        RaiseResultFilterActiveStates();
        RebuildEvaluationResults();
        EvaluationFailureFilterSummary = $"Applied failure filter: {filter.Name}.";
    }

    private void RaiseResultFilterActiveStates()
    {
        OnPropertyChanged(nameof(IsAllResultsFilterActive));
        OnPropertyChanged(nameof(IsPassedResultsFilterActive));
        OnPropertyChanged(nameof(IsFailedResultsFilterActive));
    }

    /// <summary>Compute the KPI stat-card strings from a finished report. Uses the report's own run
    /// threshold when saved (else the current threshold field) so the pass-rate sub-line names the
    /// threshold the numbers were actually produced at. A zero-example report clears the cards.</summary>
    private void UpdateEvaluationKpis(EvaluationReport report)
    {
        var tested = report.ExamplesTested;
        if (tested <= 0)
        {
            ClearEvaluationKpis();
            return;
        }

        var passed = Math.Max(0, tested - report.FailedExamples);
        var threshold = report.RunSettings?.ScoreThreshold ?? ParseThresholdOrDefault();
        HasEvaluationReport = true;
        AverageScoreDisplay = report.AverageScore.ToString("0.#", CultureInfo.InvariantCulture);
        PassRateDisplay = $"{Math.Round(passed * 100.0 / tested).ToString("0", CultureInfo.InvariantCulture)}%";
        PassRateDetail =
            $"{passed} / {tested} ≥ {threshold.ToString("0.#", CultureInfo.InvariantCulture)}";
        EvaluatedDisplay = tested.ToString(CultureInfo.InvariantCulture);
        MetricDisplay = FormatMetricLabel(report.Metric);
    }

    private void ClearEvaluationKpis()
    {
        HasEvaluationReport = false;
        AverageScoreDisplay = "—";
        PassRateDisplay = "—";
        PassRateDetail = string.Empty;
        EvaluatedDisplay = "—";
        MetricDisplay = "—";
    }

    private double ParseThresholdOrDefault()
    {
        return double.TryParse(
            EvaluationScoreThreshold,
            NumberStyles.Any,
            CultureInfo.InvariantCulture,
            out var threshold)
            ? threshold
            : 70.0;
    }

    /// <summary>Compact scorer label for the Metric KPI card — never a bare metric key.</summary>
    private static string FormatMetricLabel(string? metric) => (metric ?? "keyword_overlap") switch
    {
        "llm_judge" => "LLM judge",
        "keyword_overlap" => "keyword overlap",
        _ => metric!.Replace('_', ' '),
    };

    public void ApplyEvaluationFailureFilterSaved(EvaluationFailureFilter filter)
    {
        EvaluationFailureFilterSummary = $"Saved failure filter: {filter.Name}.";
    }

    public void SetEvaluationFailureFilterError(string message)
    {
        EvaluationFailureFilterSummary =
            $"Failure filter could not be updated.{Environment.NewLine}{message}";
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

    private void SetEvaluationResults(IReadOnlyList<EvaluationExampleResult> results)
    {
        _allEvaluationResults.Clear();
        _allEvaluationResults.AddRange(results);
        EvaluationPassCount = _allEvaluationResults.Count(result => result.Passed);
        EvaluationFailCount = _allEvaluationResults.Count(result => !result.Passed);
        RebuildEvaluationFilterOptions();
        RebuildEvaluationResults();
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
        EvaluationPassCount = 0;
        EvaluationFailCount = 0;
        SelectedEvaluationExampleResult = null;
        EvaluationResultsSummary = "Evaluation example review queue appears after a run or report reload.";
        ClearEvaluationExampleSelection();
        ClearEvaluationKpis();
    }

    private void ClearEvaluationExampleSelection()
    {
        SelectedEvaluationExampleDetail =
            "Per-example evaluation results appear here after a run or report reload.";
        EvaluationManualScore = string.Empty;
        EvaluationManualNotes = string.Empty;
        EvaluationReviewSummary = "Select an evaluation result to add a manual score or note.";
    }

    // public (and non-static) so the shell's Training-baseline bridge can reuse it via IEvaluationViewModel.
    public string BuildEvaluationReportComparison(
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

    /// <summary>Reset all run/report/result state on a project switch so nothing leaks across projects.</summary>
    public void Reset()
    {
        EvaluationSummary = "Run a local model against this project's saved examples.";
        EvaluationReportJson = "Evaluation reports appear here after a run.";
        EvaluationReportHistory.Clear();
        ClearEvaluationResults();
        ResetEvaluationFailureFilters();
        SelectedEvaluationReportHistoryItem = null;
    }

    // ---- helpers missed by the first extraction pass (eval-core; moved verbatim) ----

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

    private static bool IsAllFilter(string value)
    {
        return string.IsNullOrWhiteSpace(value)
            || string.Equals(value, "All", StringComparison.OrdinalIgnoreCase);
    }

    private static string FormatManualScoreSummary(EvaluationReport report)
    {
        if (report.ManuallyScoredExamples == 0 || report.AverageManualScore is null)
        {
            return "none";
        }

        return $"{report.ManuallyScoredExamples} example(s), average {report.AverageManualScore:0.##}";
    }

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

    // ---- filter/summary helpers (second missed batch; moved verbatim) ----

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

    /// <summary>Public wrapper over the internal failure classifier, for the shell's
    /// eval-failure-edit bridge (which records the reason on a ReviewedFixRecord).</summary>
    public string ClassifyFailureReason(EvaluationExampleResult result) => FailureReason(result);

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

    // ---- leaf pure helpers (tag/score-band; moved verbatim, static) ----

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
}
