using System;
using System.IO;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class DatasetVersionRestoreTests
{
    private static DatasetVersionDisplayItem Version(string id = "20260101T000000-1-abc", int rows = 5)
        => new(new DatasetVersionRecord { VersionId = id, RowCount = rows, CurrentIntegrity = "matches" });

    // --- pure confirmation / label / detail --------------------------------

    [Fact]
    public void BuildRestoreConfirmation_IsHonest()
    {
        var text = MainWindowViewModel.BuildRestoreConfirmation(Version("v1", 5), currentRowCount: 8);
        Assert.Contains("v1", text);
        Assert.Contains("8 row", text);  // current dataset
        Assert.Contains("5 row", text);  // target version
        Assert.Contains("undo", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("canonical", text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void BuildRestoreUndoLabel_Format()
    {
        Assert.Equal("before restore of v1", MainWindowViewModel.BuildRestoreUndoLabel(Version("v1")));
    }

    [Fact]
    public void ApplyRestoreResult_ReportsRowsVerifiedAndUndo()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyRestoreResult(new RestoreResult { VersionId = "v1", RowsWritten = 7, Verified = true });
        Assert.Contains("Restored version v1", vm.DatasetVersionDetail);
        Assert.Contains("7 row", vm.DatasetVersionDetail);
        Assert.Contains("verified", vm.DatasetVersionDetail);
        Assert.Contains("undo", vm.DatasetVersionDetail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ApplyRestoreResult_UnverifiedWhenVerifySkipped()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyRestoreResult(new RestoreResult { VersionId = "v1", RowsWritten = 3, Verified = false, VerifySkipped = true });
        Assert.Contains("unverified", vm.DatasetVersionDetail);
    }

    // --- atomic replace-from-file (real file IO, no WPF/engine) ------------

    [Fact]
    public void AtomicReplaceFromFile_ReplacesExistingTargetAndConsumesTemp()
    {
        using var dir = new TempProjectDirectory();
        var target = Path.Combine(dir.Path, "examples.jsonl");
        var temp = Path.Combine(dir.Path, "examples.jsonl.restore.tmp");
        File.WriteAllText(target, "OLD");
        File.WriteAllText(temp, "NEW");

        PythonEngineService.AtomicReplaceFromFile(temp, target);

        Assert.Equal("NEW", File.ReadAllText(target));
        Assert.False(File.Exists(temp));  // source consumed by the swap
    }

    [Fact]
    public void AtomicReplaceFromFile_MovesWhenTargetAbsent()
    {
        using var dir = new TempProjectDirectory();
        var target = Path.Combine(dir.Path, "examples.jsonl");
        var temp = Path.Combine(dir.Path, "examples.jsonl.restore.tmp");
        File.WriteAllText(temp, "NEW");

        PythonEngineService.AtomicReplaceFromFile(temp, target);

        Assert.Equal("NEW", File.ReadAllText(target));
        Assert.False(File.Exists(temp));
    }

    // --- hollow-undo safety gate (the audit's high-severity data-loss fix) --

    [Theory]
    [InlineData(true, 5, "abc", true, true)]    // rows stored => a real, restorable undo point
    [InlineData(true, 0, "abc", true, true)]    // empty but stored => fine
    [InlineData(false, 0, "abc", true, true)]   // genuinely empty (engine gives it a fingerprint) => nothing to lose
    [InlineData(false, 0, null, false, true)]   // current file missing/fresh => nothing to lose
    [InlineData(false, 0, null, true, false)]   // present-but-UNREADABLE (0 rows + null fp) => REFUSE (the fix)
    [InlineData(false, 5, "abc", true, false)]  // rows exist but NOT stored => hollow undo => abort the restore
    public void IsUndoRestorable_GatesHollowUndo(
        bool rowsStored, int rowCount, string? contentFingerprint, bool currentDatasetExists, bool expected)
    {
        var undo = new DatasetVersionRecord
        {
            VersionId = "u",
            RowsStored = rowsStored,
            RowCount = rowCount,
            ContentFingerprint = contentFingerprint,
        };
        Assert.Equal(expected, PythonEngineService.IsUndoRestorable(undo, currentDatasetExists));
    }

    [Fact]
    public void DatasetVersionRecord_MapsRowsStoredFromEngineJson()
    {
        const string json = """
            {"version_id": "v1", "created_at": "t", "updated_at": "t", "row_count": 5,
             "content_fingerprint": "abc", "rows_stored": true, "stored_row_count": 5}
            """;
        var record = PythonEngineService.ParseDatasetVersionRecord(json);
        Assert.True(record.RowsStored);
        Assert.Equal(5, record.StoredRowCount);
    }

    [Fact]
    public void ParseRestoreResult_ReadsEngineJson()
    {
        const string json = """
            {"version_id": "v1", "rows_written": 4, "verified": true, "verify_skipped": false, "output_path": "/x/out.jsonl"}
            """;
        var result = PythonEngineService.ParseRestoreResult(json);
        Assert.Equal("v1", result.VersionId);
        Assert.Equal(4, result.RowsWritten);
        Assert.True(result.Verified);
        Assert.False(result.VerifySkipped);
    }
}
