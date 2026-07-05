using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Evaluation tab core view-model (backend-cluster slice 3, PR 3b). Owns the run + report
/// panes, the per-example result review queue (filters + saved failure views + manual score/notes), and
/// the report-history + comparison.
///
/// <para>The shell keeps the cross-tab bridges that reach in: the backend health operations set
/// <see cref="EvaluationSummary"/> (via <see cref="SetSummary"/>); the eval-failure review/edit actions
/// read <see cref="SelectedEvaluationExampleResult"/> and drive AI Assist / Writing Studio / reviewed
/// fixes; Training reads <see cref="SelectedEvaluationReportHistoryItem"/> for its baseline/comparison.
/// A run failure surfaces via <see cref="ErrorReported"/> (the shell forwards it to its error banner).
/// Behind an interface so the shell/tests/DI depend on the contract.</para></summary>
public interface IEvaluationViewModel : INotifyPropertyChanged
{
    event Action<string>? ErrorReported;

    ObservableCollection<EvaluationReportHistoryItem> EvaluationReportHistory { get; }
    ObservableCollection<EvaluationExampleResult> EvaluationResults { get; }
    ObservableCollection<string> EvaluationResultFilterOptions { get; }
    ObservableCollection<string> EvaluationTagFilterOptions { get; }
    ObservableCollection<string> EvaluationFailureReasonFilterOptions { get; }
    ObservableCollection<string> EvaluationScoreBandFilterOptions { get; }
    ObservableCollection<EvaluationFailureFilter> EvaluationFailureFilters { get; }

    string EvaluationLimit { get; set; }
    string EvaluationScoreThreshold { get; set; }
    string EvaluationSummary { get; set; }
    string EvaluationReportJson { get; }
    EvaluationExampleResult? SelectedEvaluationExampleResult { get; set; }
    string SelectedEvaluationExampleDetail { get; }
    string EvaluationResultsSummary { get; }
    string EvaluationResultFilter { get; set; }
    string EvaluationTagFilter { get; set; }
    string EvaluationFailureReasonFilter { get; set; }
    string EvaluationScoreBandFilter { get; set; }
    string EvaluationFailureFilterName { get; set; }
    EvaluationFailureFilter? SelectedEvaluationFailureFilter { get; set; }
    string EvaluationFailureFilterSummary { get; }
    string EvaluationManualScore { get; set; }
    string EvaluationManualNotes { get; set; }
    string EvaluationReviewSummary { get; set; }
    EvaluationReportHistoryItem? SelectedEvaluationReportHistoryItem { get; set; }
    EvaluationReportHistoryItem? SecondaryEvaluationReportHistoryItem { get; set; }
    string EvaluationComparisonSummary { get; }

    void SetEvaluationInProgress();
    void SetEvaluationPreflightInProgress();
    void SetEvaluationRegressionRerunPreflightInProgress(EvaluationRunSettings settings);
    void SetEvaluationRegressionRerunInProgress(EvaluationRunSettings settings);
    void ApplyEvaluationRunResult(EvaluationRunResult result);
    void SetEvaluationReportHistory(IEnumerable<EvaluationReportHistoryItem> history);
    void ApplyEvaluationReportHistoryItem(EvaluationReportHistoryItem item);
    void ApplySavedEvaluationManualReview(EvaluationReportHistoryItem item);
    bool TryGetSelectedEvaluationRunSettings(out EvaluationRunSettings settings, out string errorMessage);
    void ApplyEvaluationRunSettings(EvaluationRunSettings settings);
    void SetEvaluationReviewError(string message);
    void SetEvaluationError(string message);
    void SetEvaluationFailureFilters(IEnumerable<EvaluationFailureFilter> filters);
    EvaluationFailureFilter BuildCurrentEvaluationFailureFilter();
    void ApplyEvaluationFailureFilter(EvaluationFailureFilter filter);
    void ApplyEvaluationFailureFilterSaved(EvaluationFailureFilter filter);
    void SetEvaluationFailureFilterError(string message);

    /// <summary>Compare the two selected saved evaluation reports; writes <see cref="EvaluationComparisonSummary"/>.</summary>
    bool CompareSelectedEvaluationReports();

    /// <summary>Build a before/after comparison of two reports. Public so the shell's Training-baseline
    /// bridge (<c>CompareTrainingBaseline</c>) can reuse it until Training decomposes.</summary>
    string BuildEvaluationReportComparison(EvaluationReportHistoryItem selected, EvaluationReportHistoryItem comparison);

    /// <summary>Classify a failed evaluation result's reason. Public so the shell's eval-failure-edit
    /// bridge can record it on a reviewed-fix record until that flow decomposes.</summary>
    string ClassifyFailureReason(EvaluationExampleResult result);

    /// <summary>Reset all run/report/result state on a project switch.</summary>
    void Reset();
}
