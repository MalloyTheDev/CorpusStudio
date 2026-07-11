using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Import Quarantine tab's own view-model (Phase-2 decomposition). Owns the self-contained
/// display core: the rejected-import-row list (loaded per project), the selection, and the detail pane.
/// The shell forwards <see cref="SetItems"/> (after an import / on project load) and <see cref="Reset"/>
/// (on project switch). The "retry" action (repair a row in Writing Studio) stays on the shell — it
/// writes the not-yet-extracted Writing Studio draft and tracks the pending row — and reaches in via
/// <see cref="SelectedImportQuarantineItem"/>. Behind an interface so the shell/tests/DI depend on the
/// contract.</summary>
public interface IQuarantineViewModel : INotifyPropertyChanged
{
    ObservableCollection<ImportQuarantineItem> ImportQuarantineItems { get; }
    ImportQuarantineItem? SelectedImportQuarantineItem { get; set; }
    string SelectedImportQuarantineDetail { get; }

    /// <summary>Number of rejected rows currently held in quarantine.</summary>
    int RejectedCount { get; }

    /// <summary>True when any rows are quarantined — drives the section header / list vs. empty-state.</summary>
    bool HasQuarantine { get; }

    /// <summary>Warn-badge label for the section header, e.g. "2 rejected".</summary>
    string RejectedBadge { get; }

    /// <summary>Right-aligned summary of what's held out of the dataset for repair. Only reflects the
    /// real quarantine count (the accepted-row count isn't tracked on this tab), so nothing is faked.</summary>
    string QuarantineSummary { get; }

    /// <summary>Replace the quarantine list (newest import result), selecting the first row.</summary>
    void SetItems(IEnumerable<ImportQuarantineItem> items);

    /// <summary>Clear the quarantine list + selection on a project switch.</summary>
    void Reset();
}
