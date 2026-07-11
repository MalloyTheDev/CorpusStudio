using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Model Artifacts tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>); honesty invariants unchanged — the promote gate fails closed on
/// anything that is not an explicit pass/warn, and the integrity summary counts kept vs. flagged
/// (missing/modified).</summary>
public sealed class ArtifactsViewModel : ViewModelBase, IArtifactsViewModel
{
    private const string DefaultSummary =
        "Register a model artifact from a completed run, then keep or reject it.";
    private const string DefaultDetail =
        "Select an artifact, then View card or Keep (Keep is promote-gated).";

    private string _artifactSummary = DefaultSummary;
    private string _artifactDetail = DefaultDetail;
    private ArtifactDisplayItem? _selectedModelArtifact;

    public ObservableCollection<ArtifactDisplayItem> ModelArtifacts { get; } = [];

    public ArtifactDisplayItem? SelectedModelArtifact
    {
        get => _selectedModelArtifact;
        set => SetField(ref _selectedModelArtifact, value);
    }

    public string ArtifactSummary
    {
        get => _artifactSummary;
        private set => SetField(ref _artifactSummary, value);
    }

    public string ArtifactDetail
    {
        get => _artifactDetail;
        private set
        {
            if (SetField(ref _artifactDetail, value))
            {
                OnPropertyChanged(nameof(HasArtifactDetail));
            }
        }
    }

    /// <summary>Whether any artifact rows are loaded — drives the card list vs. the empty state.</summary>
    public bool HasArtifacts => ModelArtifacts.Count > 0;

    /// <summary>Whether the detail pane holds real content (a rendered weight card or a promote-gate
    /// verdict) rather than the idle hint — drives the collapsible detail surface on the card screen.</summary>
    public bool HasArtifactDetail =>
        !string.IsNullOrWhiteSpace(_artifactDetail) && _artifactDetail != DefaultDetail;

    public void SetArtifactError(string message)
    {
        ArtifactSummary = $"Artifact action failed.{Environment.NewLine}{message}";
    }

    /// <summary>Set the detail pane (weight card markdown or a gate verdict).</summary>
    public void SetArtifactDetail(string text)
    {
        ArtifactDetail = text;
    }

    /// <summary>Format a promote-gate verdict for the detail pane. Returns whether
    /// the keep is allowed (block => refused).</summary>
    public bool ApplyPromoteGate(GateReport report)
    {
        // Drive the decision from the canonical OverallStatus (worst of results),
        // and fail closed on anything that is not an explicit pass/warn.
        string ReasonFor(string status) =>
            report.Results.FirstOrDefault(r => r.Status == status)?.Message ?? string.Empty;

        switch (report.OverallStatus)
        {
            case "block":
                var blockReason = ReasonFor("block");
                ArtifactDetail = "⛔ Keep blocked by the promote gate:" + Environment.NewLine
                    + (string.IsNullOrEmpty(blockReason) ? "the artifact did not pass promotion." : blockReason);
                return false;
            case "warn":
                var warnReason = ReasonFor("warn");
                ArtifactDetail = "⚠ Kept, but the promote gate warned:" + Environment.NewLine
                    + (string.IsNullOrEmpty(warnReason) ? "review the weight card." : warnReason)
                    + Environment.NewLine + "View the weight card before relying on it.";
                return true;
            case "pass":
                ArtifactDetail = "✅ Kept — the promote gate passed (integrity ok, no regression).";
                return true;
            default:
                ArtifactDetail = $"⛔ Keep blocked: unrecognized gate status '{report.OverallStatus}'.";
                return false;
        }
    }

    /// <summary>Refresh the artifact list + a one-line summary (kept / flagged counts).</summary>
    public void ApplyArtifacts(IReadOnlyList<ArtifactDisplayItem> items)
    {
        var selectedId = SelectedModelArtifact?.Record.ArtifactId;
        ModelArtifacts.Clear();
        foreach (var item in items)
        {
            ModelArtifacts.Add(item);
        }
        SelectedModelArtifact = ModelArtifacts.FirstOrDefault(i => i.Record.ArtifactId == selectedId);
        OnPropertyChanged(nameof(HasArtifacts));

        if (items.Count == 0)
        {
            ArtifactSummary = "No artifacts registered yet. Register one from a completed run.";
            return;
        }
        var kept = items.Count(i => i.Record.Status == "kept");
        var flagged = items.Count(i => i.Integrity != "ok");
        ArtifactSummary = $"{items.Count} artifact(s): {kept} kept, {flagged} with integrity issues (missing/modified).";
    }

    /// <summary>Clear all artifact state on a project switch so nothing leaks across projects.
    /// The artifact registry is per-project on disk; without this the previous project's list,
    /// selection, and panes would linger — and a Keep/Reject would act on a stale artifact id
    /// against the newly selected project.</summary>
    public void Reset()
    {
        ModelArtifacts.Clear();
        SelectedModelArtifact = null;
        ArtifactSummary = DefaultSummary;
        ArtifactDetail = DefaultDetail;
        OnPropertyChanged(nameof(HasArtifacts));
    }
}
