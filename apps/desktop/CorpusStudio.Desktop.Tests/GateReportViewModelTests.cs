using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class GateReportViewModelTests
{
    private static GateReport Report(string overall, params GateResult[] results)
    {
        return new GateReport
        {
            Scope = "dataset",
            Target = "examples.jsonl",
            OverallStatus = overall,
            PassCount = System.Array.FindAll(results, r => r.Status == "pass").Length,
            WarnCount = System.Array.FindAll(results, r => r.Status == "warn").Length,
            BlockCount = System.Array.FindAll(results, r => r.Status == "block").Length,
            Results = results,
        };
    }

    [Fact]
    public void ApplyGateReport_ShowsOverallStatusAndCounts()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(Report(
            "pass",
            new GateResult { GateId = "schema", Name = "Schema validation", Status = "pass", Message = "All rows validate." }
        ));

        Assert.Contains("dataset gates: PASS", vm.GateSummary);
        Assert.Contains("1 pass, 0 warn, 0 block", vm.GateSummary);
        Assert.Contains("[PASS] Schema validation", vm.GateSummary);
    }

    [Fact]
    public void ApplyGateReport_ShowsBlockAndRepairForFailingGate()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(Report(
            "block",
            new GateResult { GateId = "pii", Name = "PII / secret leakage", Status = "block", Message = "Found api_key.", Repair = "Remove keys before continuing." }
        ));

        Assert.Contains("dataset gates: BLOCK", vm.GateSummary);
        Assert.Contains("[BLOCK] PII / secret leakage: Found api_key.", vm.GateSummary);
        Assert.Contains("fix: Remove keys before continuing.", vm.GateSummary);
    }

    [Fact]
    public void ApplyGateReport_OmitsRepairForPassingGate()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(Report(
            "pass",
            new GateResult { GateId = "schema", Name = "Schema validation", Status = "pass", Message = "ok", Repair = "should-not-show" }
        ));

        Assert.DoesNotContain("should-not-show", vm.GateSummary);
    }

    [Fact]
    public void SetGateError_ShowsMessage()
    {
        var vm = new MainWindowViewModel();
        vm.SetGateError("engine exploded");
        Assert.Contains("Gates could not run", vm.GateSummary);
        Assert.Contains("engine exploded", vm.GateSummary);
    }

    // ---- Problems panel (v1.2.6) -------------------------------------------------

    [Theory]
    [InlineData("block", "⛔", 0, "#DC2626")]
    [InlineData("warn", "⚠", 1, "#D97706")]
    [InlineData("pass", "✅", 2, "#16A34A")]
    [InlineData("weird", "•", 2, "#64748B")]
    public void ProblemItem_FromGateResult_MapsSeverity(string status, string icon, int rank, string color)
    {
        var item = ProblemItem.FromGateResult(
            new GateResult { Name = "n", Status = status, Message = "m" });
        Assert.Equal(icon, item.SeverityIcon);
        Assert.Equal(rank, item.SeverityRank);
        Assert.Equal(color, item.SeverityColor);
    }

    [Theory]
    [InlineData("block", true)]
    [InlineData("warn", true)]
    [InlineData("pass", false)]
    [InlineData("", false)]
    [InlineData("unknown", false)]
    public void ProblemItem_IsProblem_OnlyBlockAndWarn(string status, bool expected) =>
        Assert.Equal(expected, ProblemItem.IsProblem(new GateResult { Status = status }));

    [Fact]
    public void ProblemItem_HasFix_TracksRepairPresence()
    {
        Assert.True(ProblemItem.FromGateResult(new GateResult { Status = "warn", Repair = "do x" }).HasFix);
        Assert.False(ProblemItem.FromGateResult(new GateResult { Status = "warn", Repair = null }).HasFix);
    }

    [Fact]
    public void ApplyGateReport_PopulatesProblems_BlockFirst_PassesExcluded()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(Report(
            "block",
            new GateResult { Name = "Schema", Status = "pass", Message = "ok" },
            new GateResult { Name = "Quality", Status = "warn", Message = "thin coverage" },
            new GateResult { Name = "PII", Status = "block", Message = "found api_key", Repair = "remove keys" }
        ));

        // Passes are not problems; block sorts before warn.
        Assert.Equal(2, vm.Problems.Count);
        Assert.Equal("PII", vm.Problems[0].Name);
        Assert.Equal("Quality", vm.Problems[1].Name);
        Assert.False(vm.IsNoProblems);

        // Badge = block+warn count, coloured red because a block exists.
        Assert.Equal("2", vm.ProblemsBadge);
        Assert.True(vm.HasProblemsBadge);
        Assert.Equal("#DC2626", vm.ProblemsBadgeColor);
        Assert.Contains("2 problems", vm.ProblemsSummary);
        Assert.Contains("1 block", vm.ProblemsSummary);
        Assert.Contains("1 passed", vm.ProblemsSummary);
    }

    [Fact]
    public void ApplyGateReport_WarnOnly_BadgeIsAmber()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(Report(
            "warn",
            new GateResult { Name = "Quality", Status = "warn", Message = "thin" }
        ));

        Assert.Equal("1", vm.ProblemsBadge);
        Assert.Equal("#D97706", vm.ProblemsBadgeColor);
        Assert.Contains("1 problem ", vm.ProblemsSummary); // singular, trailing space
    }

    [Fact]
    public void ApplyGateReport_CleanReport_NoProblems_NoBadge()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(Report(
            "pass",
            new GateResult { Name = "Schema", Status = "pass", Message = "ok" },
            new GateResult { Name = "PII", Status = "pass", Message = "clean" }
        ));

        Assert.Empty(vm.Problems);
        Assert.True(vm.IsNoProblems);
        Assert.Equal(string.Empty, vm.ProblemsBadge);
        Assert.False(vm.HasProblemsBadge);
        Assert.Contains("No problems", vm.ProblemsSummary);
        Assert.Contains("all 2 checks passed", vm.ProblemsSummary);
        Assert.Contains("not approval", vm.ProblemsSummary); // clean gate != approved
    }

    [Fact]
    public void ResetProblems_ClearsListBadgeAndSummary()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(Report("block",
            new GateResult { Name = "PII", Status = "block", Message = "x" }));
        Assert.NotEmpty(vm.Problems);

        vm.ResetProblems();
        Assert.Empty(vm.Problems);
        Assert.True(vm.IsNoProblems);
        Assert.Equal(string.Empty, vm.ProblemsBadge);
        Assert.Contains("Run gates", vm.ProblemsSummary);
    }

    [Fact]
    public void ToggleProblemsPanel_FlipsVisibility()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.ProblemsPanelVisible);
        vm.ToggleProblemsPanel();
        Assert.True(vm.ProblemsPanelVisible);
        vm.ToggleProblemsPanel();
        Assert.False(vm.ProblemsPanelVisible);
    }
}
