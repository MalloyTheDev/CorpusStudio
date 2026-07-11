using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Model Artifacts tab's own view-model (Phase-2 decomposition). Self-contained: the
/// artifact list + selection and the summary/detail panes. The shell forwards only the per-project
/// lifecycle (<see cref="Reset"/> on project switch). The engine still owns register/promote/gate;
/// this holds the tab's display state and the promote-gate verdict formatter. Behind an interface so
/// the shell/tests/DI depend on the contract.</summary>
public interface IArtifactsViewModel : INotifyPropertyChanged
{
    ObservableCollection<ArtifactDisplayItem> ModelArtifacts { get; }
    ArtifactDisplayItem? SelectedModelArtifact { get; set; }
    string ArtifactSummary { get; }
    string ArtifactDetail { get; }

    /// <summary>Whether any artifact rows are loaded (card list vs. empty state).</summary>
    bool HasArtifacts { get; }

    /// <summary>Whether the detail pane holds a rendered weight card / promote-gate verdict
    /// (rather than the idle hint) — drives the collapsible detail surface.</summary>
    bool HasArtifactDetail { get; }

    void SetArtifactError(string message);
    void SetArtifactDetail(string text);

    /// <summary>Format a promote-gate verdict for the detail pane. Returns whether the keep is
    /// allowed (block =&gt; refused).</summary>
    bool ApplyPromoteGate(GateReport report);

    void ApplyArtifacts(IReadOnlyList<ArtifactDisplayItem> items);

    /// <summary>Clear all artifact state on a project switch (list, selection, panes).</summary>
    void Reset();
}
