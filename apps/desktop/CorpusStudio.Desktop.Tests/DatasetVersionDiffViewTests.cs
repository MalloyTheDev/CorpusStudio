using System;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class DatasetVersionDiffViewTests
{
    private static DatasetVersionDisplayItem Version(string id)
        => new(new DatasetVersionRecord { VersionId = id, RowCount = 3, CurrentIntegrity = "matches" });

    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new DateTime(2026, 1, 1), new DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    [Fact]
    public void DatasetDiffBaseId_DefaultsEmpty()
    {
        Assert.Equal(string.Empty, new MainWindowViewModel().Versions.DatasetDiffBaseId);
    }

    [Fact]
    public void SetDatasetDiffBase_PinsBaseAndPromptsNextStep()
    {
        var vm = new MainWindowViewModel();
        vm.Versions.SetDatasetDiffBase(Version("v-base"));
        Assert.Equal("v-base", vm.Versions.DatasetDiffBaseId);
        Assert.Contains("v-base", vm.Versions.DatasetVersionDetail);
        Assert.Contains("Diff base", vm.Versions.DatasetVersionDetail);
    }

    [Fact]
    public void SetDatasetDiffBase_Repins()
    {
        var vm = new MainWindowViewModel();
        vm.Versions.SetDatasetDiffBase(Version("v1"));
        vm.Versions.SetDatasetDiffBase(Version("v2"));
        Assert.Equal("v2", vm.Versions.DatasetDiffBaseId);
    }

    [Fact]
    public void SetDatasetDiffBase_SurfacesPersistentLabel()
    {
        var vm = new MainWindowViewModel();
        Assert.Equal("No diff base pinned.", vm.Versions.DatasetDiffBaseLabel);
        vm.Versions.SetDatasetDiffBase(Version("v-base"));
        Assert.Contains("v-base", vm.Versions.DatasetDiffBaseLabel);
    }

    [Fact]
    public void SelectProject_ClearsDiffBaseAndVersionState()
    {
        // The audit's high finding: version state (list, selection, pinned base) must
        // not leak across a project switch.
        var vm = new MainWindowViewModel();
        vm.Versions.ApplyDatasetVersions(new[] { Version("v1") });
        vm.Versions.SelectedDatasetVersion = vm.Versions.DatasetVersions[0];
        vm.Versions.SetDatasetDiffBase(Version("v1"));

        vm.SelectProject(Project("other"));

        Assert.Equal(string.Empty, vm.Versions.DatasetDiffBaseId);
        Assert.Equal("No diff base pinned.", vm.Versions.DatasetDiffBaseLabel);
        Assert.Empty(vm.Versions.DatasetVersions);
        Assert.Null(vm.Versions.SelectedDatasetVersion);
    }
}
