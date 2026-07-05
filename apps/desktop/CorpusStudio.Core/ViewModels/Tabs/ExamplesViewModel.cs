using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Saved Examples tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>) — the saved dataset rows with a JSON inspection pane; selecting a row
/// shows its JSON, and an empty dataset reads honestly ("save a valid draft from Writing Studio").</summary>
public sealed class ExamplesViewModel : ViewModelBase, IExamplesViewModel
{
    private SavedExampleItem? _selectedExample;
    private string _selectedExampleJson = "Saved examples appear here after a project is selected.";

    public ObservableCollection<SavedExampleItem> Items { get; } = [];

    public SavedExampleItem? SelectedExample
    {
        get => _selectedExample;
        set
        {
            if (SetField(ref _selectedExample, value))
            {
                SelectedExampleJson = value?.Json ?? "Select a saved example to inspect its JSON.";
            }
        }
    }

    public string SelectedExampleJson
    {
        get => _selectedExampleJson;
        private set => SetField(ref _selectedExampleJson, value);
    }

    /// <summary>Replace the saved-examples list (dataset changed), selecting the first row. The shell's
    /// SetExamples orchestrator calls this, then fans the dataset change out to Preference/Quality/Debt.</summary>
    public void SetItems(IEnumerable<SavedExampleItem> examples)
    {
        Items.Clear();
        foreach (var example in examples)
        {
            Items.Add(example);
        }

        SelectedExample = Items.FirstOrDefault();
        SelectedExampleJson = SelectedExample?.Json
            ?? "No saved examples yet. Save a valid draft from Writing Studio.";
    }

    /// <summary>Clear the list + selection on a project switch so nothing leaks across projects.</summary>
    public void Reset()
    {
        Items.Clear();
        SelectedExample = null;
    }
}
