using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows.Input;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Saved Examples tab's own view-model (Phase-2 decomposition). Owns the dataset's saved
/// rows (<see cref="Items"/>), the selected row, its JSON inspection pane, and the structured-screen
/// state (search + All/Clean/Flagged filter → <see cref="FilteredItems"/>).
///
/// <para>Examples is the dataset spine: many other features read the list and navigate to a row (set
/// <see cref="SelectedExample"/>) — quality triage, synthetic patterns, reviewed-fix navigation — so
/// those cross-tab actions stay on the shell and reach in via <see cref="Items"/> /
/// <see cref="SelectedExample"/>. The shell's <c>SetExamples</c> orchestrator calls
/// <see cref="SetItems"/> and then fans the dataset change out to Preference/Quality/Debt. Behind an
/// interface so the shell/tests/DI depend on the contract.</para></summary>
public interface IExamplesViewModel : INotifyPropertyChanged
{
    ObservableCollection<SavedExampleItem> Items { get; }

    /// <summary>The search+filter projection of <see cref="Items"/> the list binds to.</summary>
    ObservableCollection<SavedExampleItem> FilteredItems { get; }

    SavedExampleItem? SelectedExample { get; set; }
    string SelectedExampleJson { get; }

    /// <summary>True when a row is selected — toggles the detail card vs the honest empty/prompt state.</summary>
    bool HasSelectedExample { get; }

    /// <summary>Free-text search over the visible rows (matches title/preview/instruction/output/tags/source).</summary>
    string SearchText { get; set; }

    /// <summary>Sets the active pill; bound by the All/Clean/Flagged pills with a CommandParameter.</summary>
    ICommand SetFilterCommand { get; }

    string ActiveFilter { get; }
    bool IsFilterAll { get; }
    bool IsFilterClean { get; }
    bool IsFilterFlagged { get; }

    string SearchWatermark { get; }
    string AllPillLabel { get; }
    string CleanPillLabel { get; }
    string FlaggedPillLabel { get; }

    /// <summary>Whether any rows survive the current search+filter (list empty state).</summary>
    bool HasVisibleExamples { get; }

    /// <summary>Whether the dataset has any rows at all, regardless of search/filter.</summary>
    bool HasAnyExamples { get; }

    /// <summary>Replace the saved-examples list (dataset changed), selecting the first visible row.</summary>
    void SetItems(IEnumerable<SavedExampleItem> examples);

    /// <summary>Clear the list + selection on a project switch.</summary>
    void Reset();
}
