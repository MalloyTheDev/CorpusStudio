using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Quality panel view-model (Avalonia Phase 2, slice 5). Owns the quality report display —
/// summary text, the structured metric grid + PII-aware status banner, the flagged-row detail, the
/// quality-history summary + the debt-trend mini-chart, and the synthetic-pattern issue list/triage
/// summary. It is the dashboard's Quality panel + the Debt trend, not a Studio tab.
///
/// <para>The shell keeps the synthetic-triage AI-Assist handoffs, reaching in for
/// <see cref="SelectedSyntheticPatternIssue"/> / <see cref="SyntheticPatternIssues"/> and setting
/// <see cref="QualityTriageSummary"/>. A failed quality run surfaces via <see cref="ErrorReported"/>
/// (the shell forwards it to its error banner). Behind an interface so the shell/tests/DI depend on the
/// contract.</para></summary>
public interface IQualityViewModel : INotifyPropertyChanged
{
    event Action<string>? ErrorReported;

    ObservableCollection<SyntheticPatternIssue> SyntheticPatternIssues { get; }

    /// <summary>Whether any synthetic-pattern issues were flagged (drives the triage empty/list state).</summary>
    bool HasSyntheticPatternIssues { get; }
    SyntheticPatternIssue? SelectedSyntheticPatternIssue { get; set; }
    ObservableCollection<QualityMetric> QualityMetrics { get; }
    ObservableCollection<DebtTrendPoint> DebtTrend { get; }

    string QualitySummary { get; set; }
    bool HasQualityMetrics { get; }
    string QualityStatusLine { get; }
    string QualityStatusColor { get; }
    string QualityStatusBackground { get; }
    string QualityDetail { get; }
    bool HasQualityDetail { get; }
    string QualityHistorySummary { get; }
    bool HasDebtTrendPoints { get; }
    bool HasDebtTrend { get; }
    string DebtTrendDirection { get; }
    string DebtTrendDirectionColor { get; }
    string DebtTrendSummary { get; }

    /// <summary>The synthetic-triage status line. Public set: the shell's rewrite handoffs report here.</summary>
    string QualityTriageSummary { get; set; }

    void SetQualityInProgress();
    void ApplyQualityReport(QualityReport report, IReadOnlyList<QualityHistoryEntry>? history = null);
    void SetQualityError(string message);

    /// <summary>Reset the quality/debt-trend/synthetic panel state on a project switch.</summary>
    void Reset();
}
