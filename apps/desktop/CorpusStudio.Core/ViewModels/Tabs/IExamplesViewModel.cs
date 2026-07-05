using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Saved Examples tab's own view-model (Phase-2 decomposition). Owns the dataset's saved
/// rows (<see cref="Items"/>), the selected row, and its JSON inspection pane.
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
    SavedExampleItem? SelectedExample { get; set; }
    string SelectedExampleJson { get; }

    /// <summary>Replace the saved-examples list (dataset changed), selecting the first row.</summary>
    void SetItems(IEnumerable<SavedExampleItem> examples);

    /// <summary>Clear the list + selection on a project switch.</summary>
    void Reset();
}
