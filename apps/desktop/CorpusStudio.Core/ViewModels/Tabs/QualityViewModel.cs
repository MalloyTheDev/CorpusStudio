using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Quality panel view-model (Avalonia Phase 2, slice 5). Owns the quality report display —
/// the summary text, the structured metric grid + PII-aware status banner, the flagged-row detail, the
/// quality-history summary + the debt-trend mini-chart, and the synthetic-pattern issue list/triage
/// summary. It is not a Studio tab but the dashboard's Quality panel + the Debt trend.
///
/// <para>The shell keeps the synthetic-triage AI-Assist handoffs (<c>PrepareSyntheticIssueRewrite</c> /
/// <c>PrepareSyntheticBatchRewrite</c>), reaching in for <see cref="SelectedSyntheticPatternIssue"/> /
/// <see cref="SyntheticPatternIssues"/> and setting <see cref="QualityTriageSummary"/>. A failed quality
/// run surfaces via <see cref="ErrorReported"/> (the shell forwards it to its error banner). Honesty
/// invariants intact: the status banner is PII-aware (any secret/PII is a red problem), the debt trend is
/// a lexical issue-rate proxy (presence-based PII/secrets are graded live in the Debt tab, not trended).</summary>
public sealed class QualityViewModel : ViewModelBase, IQualityViewModel
{
    /// <summary>Raised when a quality run fails; the shell forwards it to its shared error banner.</summary>
    public event Action<string>? ErrorReported;

    private string _qualitySummary = "Create or select a project to run quality checks.";

    private bool _hasQualityMetrics;

    private string _qualityStatusLine = string.Empty;

    private string _qualityStatusColor = "#64748B";

    private string _qualityStatusBackground = "#F1F5F9";

    private string _qualityDetail = string.Empty;

    private bool _hasQualityDetail;

    private string _qualityHistorySummary = "Quality history appears after quality checks run.";

    private bool _hasDebtTrend;

    private string _debtTrendDirection = string.Empty;

    private string _debtTrendDirectionColor = "#64748B";

    private string _debtTrendSummary = "Run quality checks to build a debt trend.";

    private string _qualityTriageSummary = "Synthetic quality issues appear here after quality checks run.";

    private SyntheticPatternIssue? _selectedSyntheticPatternIssue;

    public ObservableCollection<SyntheticPatternIssue> SyntheticPatternIssues { get; } = [];

    /// <summary>Whether any synthetic-pattern issues were flagged. Drives the Quality screen's
    /// synthetic-triage section: a green "no issues" card when false, the issue list when true.</summary>
    public bool HasSyntheticPatternIssues => SyntheticPatternIssues.Count > 0;

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

    public string QualitySummary
    {
        get => _qualitySummary;
        // public set: the shell's SetExamples orchestrator sets the example-count summary here.
        set => SetField(ref _qualitySummary, value);
    }

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

    public string QualityHistorySummary
    {
        get => _qualityHistorySummary;
        private set => SetField(ref _qualityHistorySummary, value);
    }

    public ObservableCollection<DebtTrendPoint> DebtTrend { get; } = [];

    public bool HasDebtTrendPoints => DebtTrend.Count > 0;

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
        // public set: the shell's synthetic-triage AI-Assist handoffs report their status here.
        set => SetField(ref _qualityTriageSummary, value);
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
        ErrorReported?.Invoke(message);
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

        OnPropertyChanged(nameof(HasSyntheticPatternIssues));
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
    /// <summary>Reset the quality/debt-trend/synthetic panel state on a project switch.</summary>
    public void Reset()
    {
        QualitySummary = "Quality checks will appear after examples are added.";
        ResetQualityMetrics();
        ResetDebtTrend();
        QualityTriageSummary = "Synthetic quality issues appear here after quality checks run.";
        SyntheticPatternIssues.Clear();
        SelectedSyntheticPatternIssue = null;
        OnPropertyChanged(nameof(HasSyntheticPatternIssues));
        QualityHistorySummary = "Quality history appears after quality checks run.";
    }

    // ---- helpers missed by the first pass (debt-trend build + synthetic summary) ----

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
}
