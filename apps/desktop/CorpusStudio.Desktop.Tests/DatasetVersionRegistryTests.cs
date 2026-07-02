using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
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
        vm.ApplyDatasetVersions(PythonEngineService.ParseDatasetVersionList(ListJson));
        Assert.Equal(2, vm.DatasetVersions.Count);
        Assert.Contains("2 version(s)", vm.DatasetVersionSummary);
        Assert.Contains("1 matching", vm.DatasetVersionSummary);
        Assert.Contains("1 drifted", vm.DatasetVersionSummary);
    }

    [Fact]
    public void ApplyDatasetVersions_PreservesSelectionByVersionId()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyDatasetVersions(PythonEngineService.ParseDatasetVersionList(ListJson));
        vm.SelectedDatasetVersion = vm.DatasetVersions[1];
        var selectedId = vm.SelectedDatasetVersion!.Record.VersionId;

        vm.ApplyDatasetVersions(PythonEngineService.ParseDatasetVersionList(ListJson)); // refresh
        Assert.NotNull(vm.SelectedDatasetVersion);
        Assert.Equal(selectedId, vm.SelectedDatasetVersion!.Record.VersionId);
    }

    [Fact]
    public void ApplyDatasetVersions_EmptyShowsNone()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyDatasetVersions([]);
        Assert.Contains("No versions captured", vm.DatasetVersionSummary);
        Assert.Empty(vm.DatasetVersions);
    }

    [Fact]
    public void SetDatasetVersionError_AndDetail()
    {
        var vm = new MainWindowViewModel();
        vm.SetDatasetVersionError("boom");
        Assert.Contains("boom", vm.DatasetVersionSummary);
        vm.SetDatasetVersionDetail("# Dataset Version Card — v1");
        Assert.Contains("Dataset Version Card", vm.DatasetVersionDetail);
    }
}
