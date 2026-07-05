using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Import Quarantine tab view-model (Phase-2 extraction). List/selection/detail in isolation
/// plus the per-project reset forwarded by the shell. The retry→Writing-Studio bridge stays on the
/// shell and is covered by ImportDedupeAndQuarantineTests.Retry_ThenTakePendingRetryItem_ReturnsItemOnce.</summary>
public sealed class QuarantineTests
{
    private static ImportQuarantineItem Item(int row) => new()
    {
        RowNumber = row,
        SourcePath = "src.jsonl",
        Raw = $"{{\"bad\":{row}}}",
        QuarantinePath = $"q{row}.json",
    };

    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    [Fact]
    public void SetItems_PopulatesListSelectsFirstAndShowsDetail()
    {
        var vm = new QuarantineViewModel();
        vm.SetItems([Item(1), Item(2)]);
        Assert.Equal(2, vm.ImportQuarantineItems.Count);
        Assert.Same(vm.ImportQuarantineItems[0], vm.SelectedImportQuarantineItem);   // first auto-selected
        Assert.Equal(vm.ImportQuarantineItems[0].DetailText, vm.SelectedImportQuarantineDetail);
    }

    [Fact]
    public void SetItems_EmptyShowsNoneMessage()
    {
        var vm = new QuarantineViewModel();
        vm.SetItems([]);
        Assert.Empty(vm.ImportQuarantineItems);
        Assert.Null(vm.SelectedImportQuarantineItem);
        Assert.Contains("No rejected import rows are in quarantine", vm.SelectedImportQuarantineDetail);
    }

    [Fact]
    public void SelectingItem_UpdatesDetail_NullClearsToPrompt()
    {
        var vm = new QuarantineViewModel();
        vm.SetItems([Item(1), Item(2)]);
        vm.SelectedImportQuarantineItem = vm.ImportQuarantineItems[1];
        Assert.Equal(vm.ImportQuarantineItems[1].DetailText, vm.SelectedImportQuarantineDetail);

        vm.SelectedImportQuarantineItem = null;
        Assert.Contains("Select a rejected import row to inspect it", vm.SelectedImportQuarantineDetail);
    }

    [Fact]
    public void Reset_ClearsListAndSelection()
    {
        var vm = new QuarantineViewModel();
        vm.SetItems([Item(1)]);
        vm.Reset();
        Assert.Empty(vm.ImportQuarantineItems);
        Assert.Null(vm.SelectedImportQuarantineItem);
    }

    [Fact]
    public void SelectProject_ResetsQuarantineState()
    {
        // A project switch must not leave the previous project's quarantined rows on screen.
        var vm = new MainWindowViewModel();
        vm.Quarantine.SetItems([Item(1), Item(2)]);

        vm.SelectProject(Project("other"));

        Assert.Empty(vm.Quarantine.ImportQuarantineItems);
        Assert.Null(vm.Quarantine.SelectedImportQuarantineItem);
    }
}
