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

    public int RejectedCount => ImportQuarantineItems.Count;

    public bool HasQuarantine => ImportQuarantineItems.Count > 0;

    public string RejectedBadge => $"{RejectedCount} rejected";

    public string QuarantineSummary =>
        RejectedCount == 1
            ? "1 row held out of the dataset for repair"
            : $"{RejectedCount} rows held out of the dataset for repair";

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
        RaiseQuarantineCountsChanged();
    }

    /// <summary>Clear the quarantine list + selection on a project switch so nothing leaks across
    /// projects. Setting the selection null resets the detail pane to the neutral prompt.</summary>
    public void Reset()
    {
        ImportQuarantineItems.Clear();
        SelectedImportQuarantineItem = null;
        RaiseQuarantineCountsChanged();
    }

    /// <summary>The count-derived display members (badge, summary, has-any) don't observe the
    /// ObservableCollection directly, so re-raise them whenever the list is replaced or cleared.</summary>
    private void RaiseQuarantineCountsChanged()
    {
        OnPropertyChanged(nameof(RejectedCount));
        OnPropertyChanged(nameof(HasQuarantine));
        OnPropertyChanged(nameof(RejectedBadge));
        OnPropertyChanged(nameof(QuarantineSummary));
    }
}
