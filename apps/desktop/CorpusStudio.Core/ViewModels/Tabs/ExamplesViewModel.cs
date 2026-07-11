using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Input;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Saved Examples tab view-model. The dataset rows with a JSON inspection pane;
/// selecting a row shows it, and an empty dataset reads honestly ("save a valid draft from Writing
/// Studio").
///
/// <para>Slice adds the structured-screen state: a <see cref="SearchText"/> box and an All/Clean/Flagged
/// <see cref="ActiveFilter"/> that project <see cref="Items"/> (the full dataset spine, unchanged) into
/// <see cref="FilteredItems"/> — what the list binds to. The Clean/Flagged split + counts come from each
/// row's REAL <see cref="SavedExampleItem.Status"/> (derived from the row JSON), never invented.</para></summary>
public sealed class ExamplesViewModel : ViewModelBase, IExamplesViewModel
{
    public const string FilterAll = "All";
    public const string FilterClean = "Clean";
    public const string FilterFlagged = "Flagged";

    private SavedExampleItem? _selectedExample;
    private string _selectedExampleJson = "Saved examples appear here after a project is selected.";
    private string _searchText = string.Empty;
    private string _activeFilter = FilterAll;

    public ExamplesViewModel()
    {
        SetFilterCommand = new RelayCommand<string>(SetFilter);
    }

    public ObservableCollection<SavedExampleItem> Items { get; } = [];

    /// <summary>The search+filter projection of <see cref="Items"/> that the list binds to. Keeping
    /// <see cref="Items"/> as the full spine means the cross-tab reads/navigation are unaffected.</summary>
    public ObservableCollection<SavedExampleItem> FilteredItems { get; } = [];

    /// <summary>Sets the active All/Clean/Flagged pill. Bound by the three pills with a CommandParameter.</summary>
    public ICommand SetFilterCommand { get; }

    public SavedExampleItem? SelectedExample
    {
        get => _selectedExample;
        set
        {
            if (SetField(ref _selectedExample, value))
            {
                SelectedExampleJson = value?.Json ?? "Select a saved example to inspect its JSON.";
                OnPropertyChanged(nameof(HasSelectedExample));
            }
        }
    }

    public string SelectedExampleJson
    {
        get => _selectedExampleJson;
        private set => SetField(ref _selectedExampleJson, value);
    }

    /// <summary>True when a row is selected — toggles the detail pane between the structured card and
    /// the honest empty/prompt state (which surfaces <see cref="SelectedExampleJson"/>).</summary>
    public bool HasSelectedExample => _selectedExample is not null;

    public string SearchText
    {
        get => _searchText;
        set
        {
            if (SetField(ref _searchText, value))
            {
                ApplyView();
            }
        }
    }

    public string ActiveFilter
    {
        get => _activeFilter;
        private set
        {
            if (SetField(ref _activeFilter, value))
            {
                OnPropertyChanged(nameof(IsFilterAll));
                OnPropertyChanged(nameof(IsFilterClean));
                OnPropertyChanged(nameof(IsFilterFlagged));
                ApplyView();
            }
        }
    }

    public bool IsFilterAll => _activeFilter == FilterAll;
    public bool IsFilterClean => _activeFilter == FilterClean;
    public bool IsFilterFlagged => _activeFilter == FilterFlagged;

    // Honest counts derived from each row's real status.
    public int AllCount => Items.Count;
    public int CleanCount => Items.Count(item => item.IsClean);
    public int FlaggedCount => Items.Count(item => item.IsFlagged);

    public string SearchWatermark => $"Search {AllCount} examples…";
    public string AllPillLabel => $"All {AllCount}";
    public string CleanPillLabel => $"Clean {CleanCount}";
    public string FlaggedPillLabel => $"Flagged {FlaggedCount}";

    /// <summary>Whether any rows survive the current search+filter — drives the list's empty state.</summary>
    public bool HasVisibleExamples => FilteredItems.Count > 0;

    /// <summary>Whether the dataset has any rows at all (regardless of search/filter).</summary>
    public bool HasAnyExamples => Items.Count > 0;

    /// <summary>Replace the saved-examples list (dataset changed), selecting the first visible row. The
    /// shell's SetExamples orchestrator calls this, then fans the dataset change out to
    /// Preference/Quality/Debt.</summary>
    public void SetItems(IEnumerable<SavedExampleItem> examples)
    {
        Items.Clear();
        foreach (var example in examples)
        {
            Items.Add(example);
        }

        ApplyView();
        NotifyCounts();

        SelectedExample = FilteredItems.FirstOrDefault();
        SelectedExampleJson = SelectedExample?.Json
            ?? "No saved examples yet. Save a valid draft from Writing Studio.";
    }

    /// <summary>Clear the list + selection on a project switch so nothing leaks across projects.</summary>
    public void Reset()
    {
        Items.Clear();
        SelectedExample = null;
        ApplyView();
        NotifyCounts();
    }

    private void SetFilter(string? filter) => ActiveFilter = filter switch
    {
        FilterClean => FilterClean,
        FilterFlagged => FilterFlagged,
        _ => FilterAll,
    };

    private void ApplyView()
    {
        FilteredItems.Clear();
        foreach (var item in Items.Where(PassesFilter).Where(PassesSearch))
        {
            FilteredItems.Add(item);
        }

        OnPropertyChanged(nameof(HasVisibleExamples));
    }

    private bool PassesFilter(SavedExampleItem item) => _activeFilter switch
    {
        FilterClean => item.IsClean,
        FilterFlagged => item.IsFlagged,
        _ => true,
    };

    private bool PassesSearch(SavedExampleItem item)
    {
        var query = _searchText?.Trim();
        if (string.IsNullOrEmpty(query))
        {
            return true;
        }

        return Haystack(item).Contains(query, System.StringComparison.OrdinalIgnoreCase);
    }

    private static string Haystack(SavedExampleItem item)
        => string.Join('\n', item.Title, item.Preview, item.Instruction, item.Output, item.Tags, item.Source);

    private void NotifyCounts()
    {
        OnPropertyChanged(nameof(AllCount));
        OnPropertyChanged(nameof(CleanCount));
        OnPropertyChanged(nameof(FlaggedCount));
        OnPropertyChanged(nameof(SearchWatermark));
        OnPropertyChanged(nameof(AllPillLabel));
        OnPropertyChanged(nameof(CleanPillLabel));
        OnPropertyChanged(nameof(FlaggedPillLabel));
        OnPropertyChanged(nameof(HasAnyExamples));
    }
}
