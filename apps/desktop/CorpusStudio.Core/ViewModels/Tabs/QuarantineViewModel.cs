using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Import Quarantine tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>) — the rejected import rows are held out of the dataset for inspection
/// and repair; the detail pane names the row and its validation errors.</summary>
public sealed class QuarantineViewModel : ViewModelBase, IQuarantineViewModel
{
    private const string DefaultDetail = "Rejected import rows appear here after a mixed import.";

    private string _selectedImportQuarantineDetail = DefaultDetail;
    private ImportQuarantineItem? _selectedImportQuarantineItem;

    public ObservableCollection<ImportQuarantineItem> ImportQuarantineItems { get; } = [];

    public ImportQuarantineItem? SelectedImportQuarantineItem
    {
        get => _selectedImportQuarantineItem;
        set
        {
            if (SetField(ref _selectedImportQuarantineItem, value))
            {
                SelectedImportQuarantineDetail = value?.DetailText
                    ?? "Select a rejected import row to inspect it.";
            }
        }
    }

    public string SelectedImportQuarantineDetail
    {
        get => _selectedImportQuarantineDetail;
        private set => SetField(ref _selectedImportQuarantineDetail, value);
    }

    public void SetItems(IEnumerable<ImportQuarantineItem> items)
    {
        ImportQuarantineItems.Clear();
        foreach (var item in items)
        {
            ImportQuarantineItems.Add(item);
        }

        SelectedImportQuarantineItem = ImportQuarantineItems.FirstOrDefault();
        SelectedImportQuarantineDetail = SelectedImportQuarantineItem?.DetailText
            ?? "No rejected import rows are in quarantine for this project.";
    }

    /// <summary>Clear the quarantine list + selection on a project switch so nothing leaks across
    /// projects. Setting the selection null resets the detail pane to the neutral prompt.</summary>
    public void Reset()
    {
        ImportQuarantineItems.Clear();
        SelectedImportQuarantineItem = null;
    }
}
