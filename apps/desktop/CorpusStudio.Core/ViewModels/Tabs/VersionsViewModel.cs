using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Dataset Versions tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>); honesty invariants unchanged — a fingerprint-less capture never
/// reads as a green success, and the integrity summary counts matching/drifted/unverifiable.</summary>
public sealed class VersionsViewModel : ViewModelBase, IVersionsViewModel
{
    private const string DefaultSummary =
        "Refresh to see dataset versions, or capture the current dataset as a version.";
    private const string DefaultDetail =
        "Select a version and View card to see its lineage (runs, artifacts, evals) and integrity.";
    private const string NoDiffBase = "No diff base pinned.";

    private string _datasetVersionSummary = DefaultSummary;
    private string _datasetVersionDetail = DefaultDetail;
    private string _datasetVersionLabel = string.Empty;
    private DatasetVersionDisplayItem? _selectedDatasetVersion;
    private string _datasetDiffBaseId = string.Empty;
    private string _datasetDiffBaseLabel = NoDiffBase;

    public ObservableCollection<DatasetVersionDisplayItem> DatasetVersions { get; } = [];

    public DatasetVersionDisplayItem? SelectedDatasetVersion
    {
        get => _selectedDatasetVersion;
        set => SetField(ref _selectedDatasetVersion, value);
    }

    /// <summary>The version pinned as the diff base (empty until "Set diff base").</summary>
    public string DatasetDiffBaseId
    {
        get => _datasetDiffBaseId;
        private set => SetField(ref _datasetDiffBaseId, value);
    }

    /// <summary>Persistent, always-visible indicator of the pinned diff base, so the
    /// user never loses track of which version the next diff uses as its base.</summary>
    public string DatasetDiffBaseLabel
    {
        get => _datasetDiffBaseLabel;
        private set => SetField(ref _datasetDiffBaseLabel, value);
    }

    public string DatasetVersionSummary
    {
        get => _datasetVersionSummary;
        private set => SetField(ref _datasetVersionSummary, value);
    }

    public string DatasetVersionDetail
    {
        get => _datasetVersionDetail;
        private set => SetField(ref _datasetVersionDetail, value);
    }

    /// <summary>Optional label typed before capturing a version (two-way bound).</summary>
    public string DatasetVersionLabel
    {
        get => _datasetVersionLabel;
        set => SetField(ref _datasetVersionLabel, value);
    }

    /// <summary>Pin a version as the base for the next diff, and prompt the next step.</summary>
    public void SetDatasetDiffBase(DatasetVersionDisplayItem version)
    {
        DatasetDiffBaseId = version.Record.VersionId;
        DatasetDiffBaseLabel = $"Diff base: {version.Record.VersionId}";
        SetDatasetVersionDetail(
            $"Diff base set to {version.Record.VersionId}. "
            + "Select another version and click 'Diff base → selected'.");
    }

    public void SetDatasetVersionError(string message)
    {
        DatasetVersionSummary = $"Dataset version action failed.{Environment.NewLine}{message}";
    }

    /// <summary>Set the detail pane (a rendered version card or a capture confirmation).</summary>
    public void SetDatasetVersionDetail(string text)
    {
        DatasetVersionDetail = text;
    }

    /// <summary>Honest capture confirmation. A record with no content fingerprint
    /// (examples.jsonl was missing/unreadable) can never be verified against the
    /// dataset — the engine annotates it 'unreadable' forever — so it must NOT read
    /// as a green "captured" success (the ✅ vocabulary the 'matches' badge uses).</summary>
    public static string FormatCaptureConfirmation(DatasetVersionRecord record)
    {
        if (string.IsNullOrEmpty(record.ContentFingerprint))
        {
            return $"⛔ Recorded version {record.VersionId}, but examples.jsonl was missing or "
                + "unreadable — no fingerprint was captured, so this version's integrity can never be verified.";
        }
        return $"✅ Captured version {record.VersionId} ({record.RowCount} rows).";
    }

    /// <summary>Honest confirmation text for an in-place restore. It overwrites the
    /// current dataset, so it names both row counts, the undo safety net, and the
    /// canonical caveat. Pure/testable.</summary>
    public static string BuildRestoreConfirmation(DatasetVersionDisplayItem version, int currentRowCount)
    {
        return $"Overwrite the current dataset ({currentRowCount} row(s)) with version "
            + $"{version.Record.VersionId} ({version.Record.RowCount} row(s))?"
            + Environment.NewLine + Environment.NewLine
            + "Your current dataset is captured as a version first (a readable dataset becomes a "
            + "restorable undo point); if it cannot be captured for undo, the restore is refused."
            + Environment.NewLine + Environment.NewLine
            + "Rows are reconstructed in canonical form (key order may change).";
    }

    /// <summary>Label for the undo version captured just before a restore.</summary>
    public static string BuildRestoreUndoLabel(DatasetVersionDisplayItem version)
    {
        return $"before restore of {version.Record.VersionId}";
    }

    /// <summary>Report a completed in-place restore honestly in the detail pane.</summary>
    public void ApplyRestoreResult(RestoreResult result)
    {
        var verifiedNote = result.Verified
            ? "verified — fingerprint matches, semantically identical to the recorded version"
            : (result.VerifySkipped ? "unverified" : "written");
        SetDatasetVersionDetail(
            $"✅ Restored version {result.VersionId}: {result.RowsWritten} row(s) [{verifiedNote}]. "
            + "Your previous dataset was captured as an undo version (restore it to revert). "
            + "Rows are in canonical form (key order may be normalized).");
    }

    /// <summary>Refresh the version list (newest first) + a one-line integrity summary.
    /// Selection is preserved by version_id across refreshes.</summary>
    public void ApplyDatasetVersions(IReadOnlyList<DatasetVersionDisplayItem> items)
    {
        var selectedId = SelectedDatasetVersion?.Record.VersionId;
        DatasetVersions.Clear();
        foreach (var item in items)
        {
            DatasetVersions.Add(item);
        }
        SelectedDatasetVersion = DatasetVersions.FirstOrDefault(i => i.Record.VersionId == selectedId);

        if (items.Count == 0)
        {
            DatasetVersionSummary = "No versions captured yet. Capture the current dataset to start a history.";
            return;
        }
        var matches = items.Count(i => i.Integrity == "matches");
        var drifted = items.Count(i => i.Integrity == "drifted");
        var unreadable = items.Count(i => i.Integrity == "unreadable");
        DatasetVersionSummary =
            $"{items.Count} version(s): {matches} matching the current dataset, {drifted} drifted, {unreadable} unverifiable.";
    }

    /// <summary>Clear all version state on a project switch so nothing leaks across projects.</summary>
    public void Reset()
    {
        DatasetVersions.Clear();
        SelectedDatasetVersion = null;
        DatasetDiffBaseId = string.Empty;
        DatasetDiffBaseLabel = NoDiffBase;
        DatasetVersionSummary = DefaultSummary;
        DatasetVersionDetail = DefaultDetail;
    }
}
