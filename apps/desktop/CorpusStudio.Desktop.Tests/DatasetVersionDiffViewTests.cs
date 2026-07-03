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
        Assert.Equal(string.Empty, new MainWindowViewModel().DatasetDiffBaseId);
    }

    [Fact]
    public void SetDatasetDiffBase_PinsBaseAndPromptsNextStep()
    {
        var vm = new MainWindowViewModel();
        vm.SetDatasetDiffBase(Version("v-base"));
        Assert.Equal("v-base", vm.DatasetDiffBaseId);
        Assert.Contains("v-base", vm.DatasetVersionDetail);
        Assert.Contains("Diff base", vm.DatasetVersionDetail);
    }

    [Fact]
    public void SetDatasetDiffBase_Repins()
    {
        var vm = new MainWindowViewModel();
        vm.SetDatasetDiffBase(Version("v1"));
        vm.SetDatasetDiffBase(Version("v2"));
        Assert.Equal("v2", vm.DatasetDiffBaseId);
    }

    [Fact]
    public void SetDatasetDiffBase_SurfacesPersistentLabel()
    {
        var vm = new MainWindowViewModel();
        Assert.Equal("No diff base pinned.", vm.DatasetDiffBaseLabel);
        vm.SetDatasetDiffBase(Version("v-base"));
        Assert.Contains("v-base", vm.DatasetDiffBaseLabel);
    }

    [Fact]
    public void SelectProject_ClearsDiffBaseAndVersionState()
    {
        // The audit's high finding: version state (list, selection, pinned base) must
        // not leak across a project switch.
        var vm = new MainWindowViewModel();
        vm.ApplyDatasetVersions(new[] { Version("v1") });
        vm.SelectedDatasetVersion = vm.DatasetVersions[0];
        vm.SetDatasetDiffBase(Version("v1"));

        vm.SelectProject(Project("other"));

        Assert.Equal(string.Empty, vm.DatasetDiffBaseId);
        Assert.Equal("No diff base pinned.", vm.DatasetDiffBaseLabel);
        Assert.Empty(vm.DatasetVersions);
        Assert.Null(vm.SelectedDatasetVersion);
    }
}
