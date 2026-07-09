using System;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Issue #190: parse a suite's run history, load it newest-first into the Suites VM, and raise
/// the selection event that triggers the trend load.</summary>
public sealed class SuiteHistoryTests
{
    [Fact]
    public void ParseSuiteHistory_ParsesEngineJsonArray()
    {
        const string json = """
        [{"generated_at":"t1","overall_status":"pass","total":5,"passed":5,"warned":0,"blocked":0,"errored":0,"summary":"s"},
         {"generated_at":"t2","overall_status":"block","total":5,"passed":3,"warned":0,"blocked":2,"errored":0,"summary":"s"}]
        """;

        var history = PythonEngineService.ParseSuiteHistory(json);

        Assert.Equal(2, history.Count);
        Assert.Equal("block", history[1].OverallStatus);
        Assert.Equal(3, history[1].Passed);
        Assert.Equal(2, history[1].Blocked);
    }

    [Fact]
    public void ParseSuiteHistory_EmptyOrBlank_ReturnsEmpty()
    {
        Assert.Empty(PythonEngineService.ParseSuiteHistory(""));
        Assert.Empty(PythonEngineService.ParseSuiteHistory("[]"));
    }

    [Fact]
    public void SetSuiteHistory_OrdersNewestFirst_AndSummarizes()
    {
        var vm = new SuitesViewModel();
        vm.SetSuiteHistory(
        [
            new SuiteHistoryEntry { GeneratedAt = "t1", OverallStatus = "pass" },
            new SuiteHistoryEntry { GeneratedAt = "t2", OverallStatus = "block" },
        ]);

        Assert.Equal(2, vm.SuiteHistory.Count);
        Assert.Equal("t2", vm.SuiteHistory[0].GeneratedAt); // newest first
        Assert.Contains("2 run", vm.SuiteHistorySummary);
    }

    [Fact]
    public void SetSuiteHistory_Empty_SetsNoHistorySummary()
    {
        var vm = new SuitesViewModel();
        vm.SetSuiteHistory(Array.Empty<SuiteHistoryEntry>());

        Assert.Empty(vm.SuiteHistory);
        Assert.Contains("No run history", vm.SuiteHistorySummary);
    }

    [Fact]
    public void HasSuiteHistory_TracksWhetherThereIsATrendToPlot()
    {
        var vm = new SuitesViewModel();
        Assert.False(vm.HasSuiteHistory);

        vm.SetSuiteHistory([new SuiteHistoryEntry { GeneratedAt = "t1", OverallStatus = "pass" }]);
        Assert.True(vm.HasSuiteHistory);

        vm.SetSuiteHistory(Array.Empty<SuiteHistoryEntry>());
        Assert.False(vm.HasSuiteHistory);
    }

    [Theory]
    [InlineData(8, 10, 0.8)]   // 80% pass → 4 + 0.8*32 = 29.6 px
    [InlineData(0, 10, 0.0)]   // 0% → the 4px floor
    [InlineData(0, 0, 0.0)]    // no cases → 0 rate, still the 4px floor
    [InlineData(10, 10, 1.0)]  // 100% → 36px
    public void SparkBarHeight_ScalesPassRateIntoAVisibleBar(int passed, int total, double expectedRate)
    {
        var entry = new SuiteHistoryEntry { Passed = passed, Total = total, OverallStatus = "pass" };
        Assert.Equal(expectedRate, entry.PassRate, 3);
        Assert.Equal(4 + expectedRate * 32, entry.SparkBarHeight, 3);
        Assert.True(entry.SparkBarHeight >= 4); // a 0% run is still a visible tick
    }

    [Fact]
    public void DisplayLine_IncludesVerdictCountsAndBlocked()
    {
        var entry = new SuiteHistoryEntry
        {
            GeneratedAt = "t1",
            OverallStatus = "block",
            Total = 5,
            Passed = 3,
            Blocked = 2,
        };

        Assert.Contains("BLOCK", entry.DisplayLine);
        Assert.Contains("3/5 passed", entry.DisplayLine);
        Assert.Contains("2 blocked", entry.DisplayLine);
    }

    [Fact]
    public void SelectedSuite_RaisesSuiteSelectedWithName()
    {
        var vm = new SuitesViewModel();
        string? selected = null;
        vm.SuiteSelected += name => selected = name;

        vm.SelectedSuite = new SuiteSummary { Name = "demo" };

        Assert.Equal("demo", selected);
    }
}
