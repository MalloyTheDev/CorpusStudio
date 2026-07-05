using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Preference Review tab's own view-model (Phase-2 decomposition). Owns the self-contained
/// display + export core: the contrast-ranked pair list (built from the project's saved examples), the
/// contrast filter, the selected pair's prompt/chosen/rejected/reason panes, the ranking summary, and
/// the DPO/training export result formatting.
///
/// <para>The tab is gated on the project schema, so the shell pushes <see cref="ActiveSchemaId"/> down
/// and forwards <see cref="SetItems"/> (on example changes) and <see cref="Reset"/> (on project switch).
/// The AI-Assist handoff actions (<c>PreparePreference*JudgeReview</c>) stay on the shell for now —
/// they write the not-yet-extracted AI Assist tab's state — and reach in via
/// <see cref="SelectedPreferenceReviewItem"/> / <see cref="GetVisiblePreferenceReviewItems"/> /
/// <see cref="SetReviewSummary"/>. Behind an interface so the shell/tests/DI depend on the
/// contract.</para></summary>
public interface IPreferenceReviewViewModel : INotifyPropertyChanged
{
    ObservableCollection<PreferenceReviewItem> PreferenceReviewItems { get; }
    ObservableCollection<string> PreferenceContrastFilterOptions { get; }
    ObservableCollection<string> PreferenceExportFormatOptions { get; }

    PreferenceReviewItem? SelectedPreferenceReviewItem { get; set; }
    string PreferencePromptText { get; }
    string PreferenceChosenText { get; }
    string PreferenceRejectedText { get; }
    string PreferenceReasonText { get; }
    string PreferenceRankingSummary { get; }
    string PreferenceReviewSummary { get; }
    string PreferenceContrastFilter { get; set; }
    string PreferenceExportFormat { get; set; }

    /// <summary>Mirror of the shell's ActiveSchemaId, pushed on project switch. The tab only builds /
    /// shows pairs for a "preference" project.</summary>
    string ActiveSchemaId { get; set; }

    /// <summary>Rebuild the pair list from the project's saved examples (preference schema only).</summary>
    void SetItems(IEnumerable<SavedExampleItem> examples);

    IReadOnlyList<PreferenceReviewItem> GetVisiblePreferenceReviewItems();
    void ApplyPreferenceTrainingExport(PreferenceExportResult result);
    void ApplyPreferenceRankingExport(string outputPath, int itemCount);
    void SetPreferenceRankingExportError(string message);

    /// <summary>Set the review summary pane. Used by the shell's AI-Assist handoff actions (the
    /// preference↔AI-Assist bridge) until AI Assist is decomposed.</summary>
    void SetReviewSummary(string message);

    /// <summary>Reset all pair/selection/filter/pane state on a project switch.</summary>
    void Reset();
}
