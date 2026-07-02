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
}
