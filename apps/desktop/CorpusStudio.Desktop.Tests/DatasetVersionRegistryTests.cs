using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class DatasetVersionRegistryTests
{
    // A realistic dataset-version-list payload: newest first, mixed integrity, and
    // the engine's live current_integrity annotation.
    private const string ListJson = """
        {
          "versions": [
            {"version_id": "20260702T182000-000002-aaa", "created_at": "t", "updated_at": "t", "label": "after clean", "trigger": "manual", "row_count": 120, "content_fingerprint": "abc", "fingerprint_algo": "sha256-ordered-exact-v1", "row_signature_kind": "exact", "source_run_ids": ["run-1"], "artifact_ids": [], "eval_report_path": null, "gate_report_path": null, "notes": "", "current_integrity": "matches"},
            {"version_id": "20260702T181000-000001-bbb", "created_at": "t", "updated_at": "t", "label": "", "trigger": "pre_training", "row_count": 100, "content_fingerprint": "def", "fingerprint_algo": "sha256-ordered-exact-v1", "row_signature_kind": "exact", "source_run_ids": ["run-1"], "artifact_ids": ["art-1"], "eval_report_path": "/x/eval.json", "gate_report_path": null, "notes": "", "current_integrity": "drifted"}
          ]
        }
        """;

    // --- parsing (pure, no engine) -------------------------------------------

    [Fact]
    public void ParseList_ReadsRecordsIntegrityAndOrder()
    {
        var items = PythonEngineService.ParseDatasetVersionList(ListJson);
        Assert.Equal(2, items.Count);
        Assert.Equal("20260702T182000-000002-aaa", items[0].Record.VersionId);
        Assert.Equal("matches", items[0].Integrity);
        Assert.Equal(120, items[0].Record.RowCount);
        Assert.Equal("drifted", items[1].Integrity);
        Assert.Equal(2, items[1].LinkCount); // run-1 + art-1
    }

    [Fact]
    public void DisplayItem_TimelineHelpers_AreHonest()
    {
        // Additive timeline display helpers (Versions re-skin): real fields only — NO fabricated grade.
        var items = PythonEngineService.ParseDatasetVersionList(ListJson);
        var head = items[0];
        Assert.Equal("after clean", head.Title);            // the real label
        Assert.Equal("120 rows · fp: abc", head.MetaLine);  // real row count + short fingerprint
        Assert.Equal("#6bbf9a", head.IntegrityColor);       // matches -> Ok green
        Assert.Equal("#d9a35f", items[1].IntegrityColor);   // drifted -> Warn amber
        Assert.Equal("(no label)", items[1].Title);         // unlabeled -> neutral placeholder, never invented
    }

    [Fact]
    public void ApplyDatasetVersions_MarksOnlyTheNewestHeadAsCurrent()
    {
        var vm = new MainWindowViewModel();
        vm.Versions.ApplyDatasetVersions(PythonEngineService.ParseDatasetVersionList(ListJson));
        Assert.True(vm.Versions.DatasetVersions[0].IsCurrent);   // newest-first head = CURRENT
        Assert.False(vm.Versions.DatasetVersions[1].IsCurrent);
    }

    [Fact]
    public void ParseList_EmptyOrMissingVersionsYieldsEmpty()
    {
        Assert.Empty(PythonEngineService.ParseDatasetVersionList("""{"versions": []}"""));
        Assert.Empty(PythonEngineService.ParseDatasetVersionList("{}"));
    }

    [Fact]
    public void ParseRecord_BareCreateHasNullIntegrity()
    {
        const string json = """
            {"version_id": "20260702T182000-000002-aaa", "created_at": "t", "updated_at": "t", "label": "x", "trigger": "manual", "row_count": 5, "content_fingerprint": "abc", "fingerprint_algo": "sha256-ordered-exact-v1", "row_signature_kind": "exact", "source_run_ids": [], "artifact_ids": [], "eval_report_path": null, "gate_report_path": null, "notes": ""}
            """;
        var record = PythonEngineService.ParseDatasetVersionRecord(json);
        Assert.Equal("20260702T182000-000002-aaa", record.VersionId);
        Assert.Equal(5, record.RowCount);
        Assert.Null(record.CurrentIntegrity);
    }

    // --- display formatting --------------------------------------------------

    [Theory]
    [InlineData("matches", "✅ matches")]
    [InlineData("drifted", "⚠ drifted")]
    [InlineData("unreadable", "⛔ unreadable")]
    [InlineData(null, "⛔ unreadable")]
    public void DisplayItem_BadgePerIntegrity(string? integrity, string badge)
    {
        var record = new DatasetVersionRecord
        {
            VersionId = "v1", Label = "L", RowCount = 7, CurrentIntegrity = integrity,
        };
        var item = new DatasetVersionDisplayItem(record);
        Assert.Contains(badge, item.DisplayName);
        Assert.Contains("L", item.DisplayName);
        Assert.Contains("7 rows", item.DisplayName);
    }

    [Fact]
    public void DisplayItem_NoLabelAndLinkCount()
    {
        var record = new DatasetVersionRecord
        {
            VersionId = "v1", RowCount = 3, SourceRunIds = ["r1"], ArtifactIds = ["a1", "a2"],
            CurrentIntegrity = "matches",
        };
        var item = new DatasetVersionDisplayItem(record);
        Assert.Contains("(no label)", item.DisplayName);
        Assert.Contains("3 links", item.DisplayName);
        Assert.Equal(3, item.LinkCount);
    }

    // --- view model apply ----------------------------------------------------

    [Fact]
    public void ApplyDatasetVersions_SummarizesIntegrityCounts()
    {
        var vm = new MainWindowViewModel();
        vm.Versions.ApplyDatasetVersions(PythonEngineService.ParseDatasetVersionList(ListJson));
        Assert.Equal(2, vm.Versions.DatasetVersions.Count);
        Assert.Contains("2 version(s)", vm.Versions.DatasetVersionSummary);
        Assert.Contains("1 matching", vm.Versions.DatasetVersionSummary);
        Assert.Contains("1 drifted", vm.Versions.DatasetVersionSummary);
    }

    [Fact]
    public void ApplyDatasetVersions_PreservesSelectionByVersionId()
    {
        var vm = new MainWindowViewModel();
        vm.Versions.ApplyDatasetVersions(PythonEngineService.ParseDatasetVersionList(ListJson));
        vm.Versions.SelectedDatasetVersion = vm.Versions.DatasetVersions[1];
        var selectedId = vm.Versions.SelectedDatasetVersion!.Record.VersionId;

        vm.Versions.ApplyDatasetVersions(PythonEngineService.ParseDatasetVersionList(ListJson)); // refresh
        Assert.NotNull(vm.Versions.SelectedDatasetVersion);
        Assert.Equal(selectedId, vm.Versions.SelectedDatasetVersion!.Record.VersionId);
    }

    [Fact]
    public void ApplyDatasetVersions_EmptyShowsNone()
    {
        var vm = new MainWindowViewModel();
        vm.Versions.ApplyDatasetVersions([]);
        Assert.Contains("No versions captured", vm.Versions.DatasetVersionSummary);
        Assert.Empty(vm.Versions.DatasetVersions);
    }

    [Fact]
    public void SetDatasetVersionError_AndDetail()
    {
        var vm = new MainWindowViewModel();
        vm.Versions.SetDatasetVersionError("boom");
        Assert.Contains("boom", vm.Versions.DatasetVersionSummary);
        vm.Versions.SetDatasetVersionDetail("# Dataset Version Card — v1");
        Assert.Contains("Dataset Version Card", vm.Versions.DatasetVersionDetail);
    }

    // --- audit: honest capture confirmation ----------------------------------

    [Fact]
    public void FormatCaptureConfirmation_SuccessWhenFingerprinted()
    {
        var record = new DatasetVersionRecord { VersionId = "v1", RowCount = 12, ContentFingerprint = "abc" };
        var text = VersionsViewModel.FormatCaptureConfirmation(record);
        Assert.Contains("✅ Captured version v1", text);
        Assert.Contains("12 rows", text);
    }

    [Fact]
    public void FormatCaptureConfirmation_HonestWhenNoFingerprint()
    {
        // Missing/unreadable examples.jsonl -> engine records a null fingerprint that
        // is 'unreadable' forever; the confirmation must not read as a green success.
        var record = new DatasetVersionRecord { VersionId = "v1", RowCount = 0, ContentFingerprint = null };
        var text = VersionsViewModel.FormatCaptureConfirmation(record);
        Assert.DoesNotContain("✅", text);
        Assert.Contains("can never be verified", text);
    }

    // --- audit: unknown integrity is unreadable in badge AND summary ---------

    [Fact]
    public void UnknownIntegrity_TreatedAsUnreadableEverywhere()
    {
        var record = new DatasetVersionRecord { VersionId = "v1", RowCount = 1, CurrentIntegrity = "stale" };
        var item = new DatasetVersionDisplayItem(record);
        Assert.Equal("unreadable", item.Integrity);        // getter normalizes
        Assert.Contains("⛔ unreadable", item.DisplayName); // badge

        var vm = new MainWindowViewModel();
        vm.Versions.ApplyDatasetVersions([item]);
        // Buckets sum to the total: the unknown value is counted as unverifiable.
        Assert.Contains("1 version(s)", vm.Versions.DatasetVersionSummary);
        Assert.Contains("1 unverifiable", vm.Versions.DatasetVersionSummary);
    }
}
