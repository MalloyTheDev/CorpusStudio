using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Evaluation Suites tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>); honesty invariants unchanged — the per-metric roll-up is never
/// folded into one cross-metric score, an error state never reads as a green pass, and the report
/// framing states the verdict is structure/thresholds, not model quality.</summary>
public sealed class SuitesViewModel : ViewModelBase, ISuitesViewModel
{
    private const string DefaultStatus =
        "Registered suites appear here. Create one with New, then run it.";
    private const string DefaultReportSummary =
        "Select a suite and Run. Running makes live model calls.";

    private SuiteSummary? _selectedSuite;
    private bool _isSuitesBusy;
    private bool _hasActiveProject;
    private bool _hasSuiteReport;
    private string _suitesStatus = DefaultStatus;
    private string _suiteReportSummary = DefaultReportSummary;
    private string _suiteOverallStatus = string.Empty;
    private string _suiteOverallColor = SuiteReport.ColorForStatus(null);
    private string _suiteHistorySummary = "Run history appears here after a suite runs.";
    private string _newSuiteName = string.Empty;

    /// <summary>Registered evaluation suites (from `suite-list`).</summary>
    public ObservableCollection<SuiteSummary> Suites { get; } = [];

    /// <summary>The last run's per-metric roll-up (never a folded cross-metric score).</summary>
    public ObservableCollection<SuiteMetricRollup> SuiteMetricRows { get; } = [];

    /// <summary>The last run's per-case results.</summary>
    public ObservableCollection<SuiteCaseResult> SuiteCaseRows { get; } = [];

    /// <summary>The selected suite's run history (newest first) for the trend (#190).</summary>
    public ObservableCollection<SuiteHistoryEntry> SuiteHistory { get; } = [];

    public string SuiteHistorySummary
    {
        get => _suiteHistorySummary;
        private set => SetField(ref _suiteHistorySummary, value);
    }

    /// <summary>Honest framing shown under the report — the verdict is not a quality judgment.</summary>
    public string SuiteHonestyNote =>
        "Verdicts case structure + score thresholds, not model quality. Keyword-overlap is a lexical proxy.";

    /// <summary>Raised when a suite is selected (with its name) so the shell can load its run history.
    /// Keeps the async engine call off this VM, mirroring the error-up event pattern.</summary>
    public event System.Action<string>? SuiteSelected;

    public SuiteSummary? SelectedSuite
    {
        get => _selectedSuite;
        set
        {
            if (SetField(ref _selectedSuite, value))
            {
                OnPropertyChanged(nameof(CanRunSuite));
                if (value is not null)
                {
                    SuiteSelected?.Invoke(value.Name);
                }
            }
        }
    }

    public bool IsSuitesBusy
    {
        get => _isSuitesBusy;
        set
        {
            if (SetField(ref _isSuitesBusy, value))
            {
                OnPropertyChanged(nameof(CanRunSuite));
            }
        }
    }

    /// <summary>The name typed into the "New suite" box (bound two-way). The shell's create-suite
    /// command reads it and clears it after a successful scaffold.</summary>
    public string NewSuiteName
    {
        get => _newSuiteName;
        set => SetField(ref _newSuiteName, value);
    }

    /// <summary>Mirror of the shell's HasActiveProject, pushed on project switch; feeds
    /// <see cref="CanRunSuite"/>.</summary>
    public bool HasActiveProject
    {
        get => _hasActiveProject;
        set
        {
            if (SetField(ref _hasActiveProject, value))
            {
                OnPropertyChanged(nameof(CanRunSuite));
            }
        }
    }

    /// <summary>Run is enabled for a valid selected suite when a project is open and not busy.</summary>
    public bool CanRunSuite =>
        HasActiveProject && !IsSuitesBusy && SelectedSuite is { Valid: true };

    public bool HasSuiteReport
    {
        get => _hasSuiteReport;
        private set => SetField(ref _hasSuiteReport, value);
    }

    public string SuitesStatus
    {
        get => _suitesStatus;
        private set => SetField(ref _suitesStatus, value);
    }

    public string SuiteReportSummary
    {
        get => _suiteReportSummary;
        private set => SetField(ref _suiteReportSummary, value);
    }

    public string SuiteOverallStatus
    {
        get => _suiteOverallStatus;
        private set => SetField(ref _suiteOverallStatus, value);
    }

    public string SuiteOverallColor
    {
        get => _suiteOverallColor;
        private set => SetField(ref _suiteOverallColor, value);
    }

    /// <summary>Populate the suites list (from `suite-list`), preserving selection.</summary>
    public void ApplySuites(IReadOnlyList<SuiteSummary> summaries)
    {
        var selectedName = SelectedSuite?.Name;
        Suites.Clear();
        foreach (var summary in summaries)
        {
            Suites.Add(summary);
        }
        SelectedSuite = Suites.FirstOrDefault(s => s.Name == selectedName);

        if (summaries.Count == 0)
        {
            SuitesStatus = "No suites defined. Create one with New, then edit it in Files.";
            return;
        }
        var invalid = summaries.Count(s => !s.Valid);
        SuitesStatus = invalid == 0
            ? $"{summaries.Count} suite(s)."
            : $"{summaries.Count} suite(s), {invalid} invalid.";
    }

    /// <summary>Show a run's SuiteReport — per-metric roll-up + per-case rows + overall verdict.</summary>
    public void ApplySuiteReport(SuiteReport report)
    {
        SuiteMetricRows.Clear();
        foreach (var rollup in report.PerMetric)
        {
            SuiteMetricRows.Add(rollup);
        }
        SuiteCaseRows.Clear();
        foreach (var result in report.Cases)
        {
            SuiteCaseRows.Add(result);
        }
        SuiteOverallStatus = string.IsNullOrWhiteSpace(report.OverallStatus)
            ? "UNKNOWN"
            : report.OverallStatus.ToUpperInvariant();
        SuiteOverallColor = SuiteReport.ColorForStatus(report.OverallStatus);
        SuiteReportSummary = report.Summary;
        HasSuiteReport = true;
    }

    /// <summary>Collapse the report to a neutral error state (never a green pass).</summary>
    public void SetSuitesError(string message)
    {
        SuiteMetricRows.Clear();
        SuiteCaseRows.Clear();
        SuiteOverallStatus = string.Empty;
        SuiteOverallColor = SuiteReport.ColorForStatus(null);
        SuiteReportSummary = message;
        HasSuiteReport = false;
    }

    /// <summary>Load the selected suite's run history (newest first for the trend list).</summary>
    public void SetSuiteHistory(IEnumerable<SuiteHistoryEntry> history)
    {
        SuiteHistory.Clear();
        foreach (var entry in history.Reverse())
        {
            SuiteHistory.Add(entry);
        }
        SuiteHistorySummary = SuiteHistory.Count == 0
            ? "No run history yet — run this suite to start a trend."
            : $"{SuiteHistory.Count} run(s), newest first (a count over time, not a quality score).";
    }

    /// <summary>Reset all suite state on a project switch so it can't leak across projects. The
    /// shell re-pushes <see cref="HasActiveProject"/> after this, so the Run gate is left disabled
    /// until the new project's flag is synced.</summary>
    public void Reset()
    {
        Suites.Clear();
        SelectedSuite = null;
        SuiteMetricRows.Clear();
        SuiteCaseRows.Clear();
        SuiteHistory.Clear();
        SuiteHistorySummary = "Run history appears here after a suite runs.";
        SuiteOverallStatus = string.Empty;
        SuiteOverallColor = SuiteReport.ColorForStatus(null);
        HasSuiteReport = false;
        SuitesStatus = DefaultStatus;
        SuiteReportSummary = DefaultReportSummary;
    }
}
