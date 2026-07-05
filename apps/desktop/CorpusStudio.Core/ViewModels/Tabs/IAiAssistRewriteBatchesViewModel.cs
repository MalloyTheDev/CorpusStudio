using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The AI-Assist saved rewrite-batches sub-view-model (backend-cluster slice 2, PR 1 of 3 —
/// the most independent leaf of the AI-Assist tab). Owns the saved rewrite-batch list, its selection,
/// the summary pane, and the "last prepared" batch handed over from synthetic batch triage.
///
/// <para>The shell keeps the cross-tab actions that write other tabs' state: synthetic batch triage
/// prepares a batch (<see cref="SetLastPrepared"/>), and <c>ResumeAiAssistRewriteBatch</c> loads a
/// batch's source draft into Writing Studio + sets the run-core action, then calls
/// <see cref="SetRewriteBatchSummary"/>. Behind an interface so the shell/tests/DI depend on the
/// contract.</para></summary>
public interface IAiAssistRewriteBatchesViewModel : INotifyPropertyChanged
{
    ObservableCollection<AiAssistRewriteBatch> AiAssistRewriteBatches { get; }
    AiAssistRewriteBatch? SelectedAiAssistRewriteBatch { get; set; }
    string AiAssistRewriteBatchSummary { get; }

    void SetAiAssistRewriteBatches(IEnumerable<AiAssistRewriteBatch> batches);
    bool TryGetLastPreparedAiAssistRewriteBatch(out AiAssistRewriteBatch batch, out string errorMessage);

    /// <summary>Record the batch just prepared by synthetic batch triage (shell bridge) so a
    /// subsequent save can persist it.</summary>
    void SetLastPrepared(AiAssistRewriteBatch batch);

    void ApplyAiAssistRewriteBatchSaved(AiAssistRewriteBatch batch);
    void SetAiAssistRewriteBatchError(string message);

    /// <summary>Set the summary pane. Used by the shell's ResumeAiAssistRewriteBatch bridge.</summary>
    void SetRewriteBatchSummary(string message);

    /// <summary>Clear the batch list/selection/summary on a project switch.</summary>
    void Reset();
}
