using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The AI-Assist tab core view-model (backend-cluster slice 2, PR 3/3). Owns the run + result
/// panes, the honest candidate gate (informational — never approval), the action/instruction inputs,
/// and the review queue (filter/search/sort + saved views + bulk triage).
///
/// <para>The shell keeps the cross-tab bridges that reach in: the synthetic/eval-failure/preference
/// writers set <see cref="AiAssistAction"/>/<see cref="AiAssistInstruction"/>; the backend health
/// operations set <see cref="AiAssistSummary"/>; ResumeAiAssistRewriteBatch + MoveAiAssistSuggestionToDraft
/// read the selection / <see cref="AiAssistSuggestionJsonl"/>. A run failure surfaces via
/// <see cref="ErrorReported"/> (the shell forwards it to its error banner). Behind an interface so the
/// shell/tests/DI depend on the contract.</para></summary>
public interface IAiAssistViewModel : INotifyPropertyChanged
{
    event Action<string>? ErrorReported;

    ObservableCollection<AiAssistReviewQueueItem> AiAssistReviewQueue { get; }
    ObservableCollection<AiAssistQueueView> AiAssistQueueViews { get; }
    ObservableCollection<string> AiAssistQueueFilterOptions { get; }
    ObservableCollection<string> AiAssistQueueSortOptions { get; }
    ObservableCollection<string> AiAssistActionPresets { get; }

    string AiAssistAction { get; set; }
    string AiAssistInstruction { get; set; }
    string AiAssistSummary { get; set; }
    string AiAssistReviewText { get; }
    string AiAssistSourceDraftText { get; }
    string AiAssistSuggestedJsonlText { get; }
    string AiAssistDiffSummary { get; }
    string AiAssistCandidateGateStatus { get; }
    string AiAssistCandidateGateColor { get; }
    bool SelectedAiAssistCandidateGateBlocks { get; }
    AiAssistReviewQueueItem? SelectedAiAssistReviewQueueItem { get; set; }
    string AiAssistQueueSummary { get; }
    string AiAssistQueueFilter { get; set; }
    string AiAssistQueueSearch { get; set; }
    string AiAssistQueueSort { get; set; }
    string AiAssistQueueViewName { get; set; }
    AiAssistQueueView? SelectedAiAssistQueueView { get; set; }
    string AiAssistSuggestionJsonl { get; }

    void SetAiAssistInProgress();
    void ApplyAiAssistRunResult(AiAssistRunResult result);
    void SetAiAssistReviewQueue(IEnumerable<AiAssistReviewQueueItem> items);
    void SetAiAssistQueueViews(IEnumerable<AiAssistQueueView> views);
    AiAssistQueueView BuildCurrentAiAssistQueueView();
    void ApplyAiAssistQueueView(AiAssistQueueView view);
    void ApplyAiAssistQueueViewSaved(AiAssistQueueView view);
    void ApplyAiAssistQueueViewLoaded(AiAssistQueueView view);
    void ApplyAiAssistReviewQueueItem(AiAssistReviewQueueItem item);
    void ApplyAiAssistReviewState(AiAssistReviewQueueItem item);
    void ApplyAiAssistBulkReviewState(int updatedCount, string reviewState, int undoStepsAvailable);
    void ApplyAiAssistBulkUndoReviewState(int updatedCount, int undoStepsAvailable);
    IReadOnlyList<string> GetVisibleAiAssistReviewIds();
    IReadOnlyDictionary<string, string> GetVisibleAiAssistReviewStates();
    void SetAiAssistError(string message);
    void SetAiAssistQueueError(string message);
    void ApplyAiAssistActionPresets(string schemaId);
    void Reset();
}
