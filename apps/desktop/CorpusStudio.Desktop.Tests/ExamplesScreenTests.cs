using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Covers the structured Examples screen slice: the REAL per-row status derived from a row's
/// JSON (feeds the Clean/Flagged pills + the list status icon), the metadata parse (Tags/Source/Tokens/
/// Added → "—" when absent, never fabricated), and the tab-VM search + filter projection.</summary>
public sealed class ExamplesScreenTests
{
    private static SavedExampleItem Row(int row, string json) => new(row, "preview", json);

    // Mirror of the engine's sentinel for an unparseable line.
    private static SavedExampleItem BrokenRow(int row) => new(row, "Invalid JSON row", "{ not json");

    // --- per-row status derived from real content -----------------------------------

    [Fact]
    public void Status_ValidRowWithOutput_IsCleanOk()
    {
        var item = Row(1, """{"instruction":"hi","output":"there"}""");
        Assert.Equal(ExampleStatus.Ok, item.Status);
        Assert.True(item.IsOk);
        Assert.True(item.IsClean);
        Assert.False(item.IsFlagged);
        Assert.Equal("Clean", item.StatusLabel);
    }

    [Fact]
    public void Status_PresentButEmptyOutput_IsWarnFlagged()
    {
        var item = Row(1, """{"instruction":"hi","output":"  "}""");
        Assert.Equal(ExampleStatus.Warn, item.Status);
        Assert.True(item.IsWarn);
        Assert.True(item.IsFlagged);
        Assert.False(item.IsClean);
        Assert.Equal("Flagged", item.StatusLabel);
    }

    [Fact]
    public void Status_InvalidJson_IsBadFlagged()
    {
        var item = BrokenRow(1);
        Assert.Equal(ExampleStatus.Bad, item.Status);
        Assert.True(item.IsBad);
        Assert.True(item.IsFlagged);
        Assert.Equal("Broken", item.StatusLabel);
    }

    [Fact]
    public void Status_UnparseableJson_IsBad_EvenWithoutSentinelPreview()
    {
        var item = new SavedExampleItem(1, "Contact support", "{ not valid json");
        Assert.Equal(ExampleStatus.Bad, item.Status);
        Assert.True(item.IsBad);
    }

    [Fact]
    public void Status_ChatMessagesWithoutAssistantReply_IsFlagged()
    {
        var item = Row(1, """{"messages":[{"role":"user","content":"q"}]}""");
        Assert.Equal(ExampleStatus.Warn, item.Status);
    }

    [Fact]
    public void Status_TextOnlyRowWithNoOutputKey_IsClean()
    {
        // A pretraining-style row carries no output field — that absence must NOT be flagged.
        var item = Row(1, """{"text":"some corpus text"}""");
        Assert.Equal(ExampleStatus.Ok, item.Status);
    }

    // --- metadata parse: real fields, "—" when absent -------------------------------

    [Fact]
    public void Metadata_ReadsRealFields()
    {
        var item = Row(1, """{"instruction":"do","output":"done","tags":["account","auth"],"source":"handwritten","tokens":34,"created_at":"3 days ago"}""");
        Assert.Equal("do", item.Instruction);
        Assert.Equal("done", item.Output);
        Assert.Equal("account · auth", item.Tags);
        Assert.Equal("handwritten", item.Source);
        Assert.Equal("34", item.Tokens);
        Assert.Equal("3 days ago", item.Added);
    }

    [Fact]
    public void Metadata_AbsentFields_ShowMissingPlaceholder_NeverFabricated()
    {
        var item = Row(1, """{"instruction":"do","output":"done"}""");
        Assert.Equal(SavedExampleItem.MissingField, item.Tags);
        Assert.Equal(SavedExampleItem.MissingField, item.Source);
        Assert.Equal(SavedExampleItem.MissingField, item.Tokens);
        Assert.Equal(SavedExampleItem.MissingField, item.Added);
    }

    [Fact]
    public void Instruction_FallsBackToChatUserTurn()
    {
        var item = Row(1, """{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}""");
        Assert.Equal("hello", item.Instruction);
        Assert.Equal("hi", item.Output);
        Assert.Equal(ExampleStatus.Ok, item.Status);
    }

    // --- tab-VM counts, filter + search projection ----------------------------------

    [Fact]
    public void Counts_ReflectRealCleanFlaggedSplit()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems(
        [
            Row(1, """{"instruction":"a","output":"x"}"""),   // clean
            Row(2, """{"instruction":"b","output":"y"}"""),   // clean
            Row(3, """{"instruction":"c","output":""}"""),    // flagged (empty output)
            BrokenRow(4),                                       // flagged (broken)
        ]);

        Assert.Equal(4, vm.AllCount);
        Assert.Equal(2, vm.CleanCount);
        Assert.Equal(2, vm.FlaggedCount);
        Assert.Equal("Search 4 examples…", vm.SearchWatermark);
        Assert.Equal("All 4", vm.AllPillLabel);
        Assert.Equal("Clean 2", vm.CleanPillLabel);
        Assert.Equal("Flagged 2", vm.FlaggedPillLabel);
    }

    [Fact]
    public void Filter_Flagged_ShowsOnlyNonCleanRows()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems(
        [
            Row(1, """{"instruction":"a","output":"x"}"""),   // clean
            Row(2, """{"instruction":"b","output":""}"""),    // flagged
            BrokenRow(3),                                       // flagged
        ]);

        Assert.Equal(3, vm.FilteredItems.Count);
        Assert.True(vm.IsFilterAll);

        vm.SetFilterCommand.Execute("Flagged");
        Assert.True(vm.IsFilterFlagged);
        Assert.False(vm.IsFilterAll);
        Assert.Equal(2, vm.FilteredItems.Count);
        Assert.All(vm.FilteredItems, item => Assert.True(item.IsFlagged));

        vm.SetFilterCommand.Execute("Clean");
        Assert.Single(vm.FilteredItems);
        Assert.True(vm.FilteredItems[0].IsClean);

        vm.SetFilterCommand.Execute("All");
        Assert.Equal(3, vm.FilteredItems.Count);
    }

    [Fact]
    public void Search_FiltersByRowContent()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems(
        [
            new SavedExampleItem(1, "Reset my password", """{"instruction":"Reset my password","output":"ok"}"""),
            new SavedExampleItem(2, "Business hours", """{"instruction":"Business hours","output":"9-5"}"""),
        ]);

        vm.SearchText = "password";
        Assert.Single(vm.FilteredItems);
        Assert.Equal(1, vm.FilteredItems[0].RowNumber);

        vm.SearchText = "";
        Assert.Equal(2, vm.FilteredItems.Count);
    }

    [Fact]
    public void Search_And_Filter_Compose()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems(
        [
            new SavedExampleItem(1, "alpha clean", """{"instruction":"alpha","output":"ok"}"""),
            new SavedExampleItem(2, "alpha empty", """{"instruction":"alpha","output":""}"""),
            new SavedExampleItem(3, "beta empty", """{"instruction":"beta","output":""}"""),
        ]);

        vm.SetFilterCommand.Execute("Flagged");
        vm.SearchText = "alpha";
        Assert.Single(vm.FilteredItems);
        Assert.Equal(2, vm.FilteredItems[0].RowNumber);
    }

    [Fact]
    public void Reset_ClearsFilteredView_AndCounts()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems([Row(1, """{"instruction":"a","output":"x"}""")]);
        Assert.True(vm.HasAnyExamples);

        vm.Reset();
        Assert.Empty(vm.FilteredItems);
        Assert.Equal(0, vm.AllCount);
        Assert.False(vm.HasVisibleExamples);
        Assert.False(vm.HasSelectedExample);
    }
}
