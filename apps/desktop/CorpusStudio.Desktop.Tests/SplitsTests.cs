using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Splits tab view-model (Phase-2 extraction). Pure display/format logic plus the two
/// cross-cutting concerns wired by the shell: the ErrorReported event -> shared error banner, and
/// the per-project lifecycle (Reset summary + reload saved ratios) on project switch.</summary>
public sealed class SplitsTests
{
    private static SplitReport Report(int rowsShared = 0)
        => new()
        {
            Train = 90,
            Validation = 5,
            Test = 5,
            TrainRatio = 0.9,
            ValidationRatio = 0.05,
            TestRatio = 0.05,
            Seed = 42,
            RowsSharedAcrossSplits = rowsShared,
            OutputDirectory = "C:/out",
        };

    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    // --- pure display / format --------------------------------------------------------

    [Fact]
    public void ApplySplitSettings_MapsRatiosToPercentTextAndSeed()
    {
        var vm = new SplitsViewModel();
        vm.ApplySplitSettings(new SplitSettings { TrainRatio = 0.8, ValidationRatio = 0.1, Seed = 7 });
        Assert.Equal("80", vm.SplitTrainPercent);
        Assert.Equal("10", vm.SplitValidationPercent);
        Assert.Equal("7", vm.SplitSeed);
    }

    [Fact]
    public void ApplySplitReport_ShowsCountsRatiosAndOutputNoLeakage()
    {
        var vm = new SplitsViewModel();
        vm.ApplySplitReport(Report(rowsShared: 0));
        Assert.Contains("Train: 90", vm.SplitSummary);
        Assert.Contains("Validation: 5", vm.SplitSummary);
        Assert.Contains("Test: 5", vm.SplitSummary);
        Assert.Contains("Seed: 42", vm.SplitSummary);
        Assert.Contains("Output: C:/out", vm.SplitSummary);
        Assert.DoesNotContain("leakage", vm.SplitSummary);   // no shared rows -> no leakage note
    }

    [Fact]
    public void ApplySplitReport_FlagsLeakageAndSurfacesWarnings()
    {
        var vm = new SplitsViewModel();
        var report = new SplitReport
        {
            Train = 8, Validation = 1, Test = 1, Seed = 42,
            RowsSharedAcrossSplits = 2,
            Warnings = { "tiny validation set" },
        };
        vm.ApplySplitReport(report);
        Assert.Contains("Rows shared across splits: 2 (train/test leakage)", vm.SplitSummary);
        Assert.Contains("Warnings:", vm.SplitSummary);
        Assert.Contains("- tiny validation set", vm.SplitSummary);
    }

    [Fact]
    public void SetSplitInProgress_ShowsRatiosAndDerivedTest()
    {
        var vm = new SplitsViewModel();
        vm.SetSplitInProgress(0.9, 0.05, 42);
        Assert.Contains("Generating", vm.SplitSummary);
        Assert.Contains("Train: 90%", vm.SplitSummary);
        Assert.Contains("Validation: 5%", vm.SplitSummary);
        Assert.Contains("Test: 5%", vm.SplitSummary);        // 1 - 0.9 - 0.05 = 0.05 -> 5%
        Assert.Contains("Seed: 42", vm.SplitSummary);
    }

    // --- ErrorReported event (the tab holds no shell reference) ------------------------

    [Fact]
    public void SetSplitError_SetsSummaryAndRaisesErrorReported()
    {
        var vm = new SplitsViewModel();
        string? reported = null;
        vm.ErrorReported += m => reported = m;
        vm.SetSplitError("bad ratios");
        Assert.Contains("could not be generated", vm.SplitSummary);
        Assert.Contains("bad ratios", vm.SplitSummary);
        Assert.Equal("bad ratios", reported);   // event carries the message for the shell banner
    }

    // --- cross-cutting wiring through the shell ---------------------------------------

    [Fact]
    public void SetSplitError_RoutesToSharedErrorBanner()
    {
        // The shell wires Splits.ErrorReported -> ReportError, so a split failure lights the
        // shared, dismissible error banner without the tab referencing the shell.
        var vm = new MainWindowViewModel();
        vm.Splits.SetSplitError("boom");
        Assert.True(vm.HasError);
        Assert.Equal("boom", vm.ErrorMessage);
    }

    [Fact]
    public void SelectProject_ResetsSummaryAndReloadsSavedRatios()
    {
        // A project switch resets the summary to the pending state and reloads the new project's
        // saved split ratios (here the defaults), so nothing leaks across projects.
        var vm = new MainWindowViewModel();
        vm.Splits.ApplySplitReport(Report(rowsShared: 3));   // dirty state from a prior project
        vm.Splits.SplitTrainPercent = "70";

        vm.SelectProject(Project("p"));

        Assert.Equal("Generate splits after examples are saved.", vm.Splits.SplitSummary);
        Assert.Equal("90", vm.Splits.SplitTrainPercent);     // reloaded from the project's default settings
        Assert.Equal("5", vm.Splits.SplitValidationPercent);
        Assert.Equal("42", vm.Splits.SplitSeed);
    }
}
