using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Dataset Versions tab's own view-model (Phase-2 decomposition). Self-contained: the
/// version list + selection, the pinned diff base, and the summary/detail panes. The shell forwards
/// only the per-project lifecycle (<see cref="Reset"/> on project switch). The engine still owns
/// capture/restore/diff; this holds the tab's display state. Behind an interface so the
/// shell/tests/DI depend on the contract.</summary>
public interface IVersionsViewModel : INotifyPropertyChanged
{
    ObservableCollection<DatasetVersionDisplayItem> DatasetVersions { get; }
    DatasetVersionDisplayItem? SelectedDatasetVersion { get; set; }
    string DatasetDiffBaseId { get; }
    string DatasetDiffBaseLabel { get; }
    string DatasetVersionSummary { get; }
    string DatasetVersionDetail { get; }
    string DatasetVersionLabel { get; set; }

    void SetDatasetDiffBase(DatasetVersionDisplayItem version);
    void SetDatasetVersionError(string message);
    void SetDatasetVersionDetail(string text);
    void ApplyRestoreResult(RestoreResult result);
    void ApplyDatasetVersions(IReadOnlyList<DatasetVersionDisplayItem> items);

    /// <summary>Clear all version state on a project switch (list, selection, diff base, panes).</summary>
    void Reset();
}
