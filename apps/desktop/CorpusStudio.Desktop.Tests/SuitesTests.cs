using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Desktop Evaluation Suites tab (v1.3 M2, slice 3): model parsing + VM shaping.
/// The verdict is per-metric (never folded) and honest (an errored case is red, never green).</summary>
public sealed class SuitesTests
{
    private const string ReportJson = """
    {
      "suite": "release-gate",
      "generated_at": "2026-07-04T00:00:00Z",
      "cases": [
        {"case": "kw", "model": "llama3", "metric": "keyword_overlap", "dataset_fingerprint": "abc",
         "examples_tested": 10, "average_score": 82.0, "pass_rate": 0.9,
         "gate": {"scope": "evaluation_report", "target": "kw", "overall_status": "pass", "results": []},
         "error": null, "status": "pass"},
        {"case": "down", "model": "llama3", "metric": "llm_judge", "dataset_fingerprint": null,
         "examples_tested": null, "average_score": null, "pass_rate": null,
         "gate": null, "error": "backend down", "status": "error"}
      ],
      "per_metric": [
        {"metric": "keyword_overlap", "total": 1, "passed": 1, "warned": 0, "blocked": 0, "errored": 0},
        {"metric": "llm_judge", "total": 1, "passed": 0, "warned": 0, "blocked": 0, "errored": 1}
      ],
      "overall_status": "block",
      "summary": "Suite 'release-gate': 2 case(s) — 1 pass, 1 error."
    }
    """;

    private const string ListJson = """
    [
      {"name": "release-gate", "case_count": 2, "valid": true, "error": null},
      {"name": "broken", "case_count": 0, "valid": false, "error": "Invalid suite definition"}
    ]
    """;

    // --- model / parse ----------------------------------------------------------------

    [Fact]
    public void ParseSuiteReport_DeserializesCasesPerMetricAndNullGate()
    {
        var report = PythonEngineService.ParseSuiteReport(ReportJson);
        Assert.Equal("release-gate", report.Suite);
        Assert.Equal("block", report.OverallStatus);
        Assert.Equal(2, report.Cases.Count);

        var errored = report.Cases.Single(c => c.Status == "error");
        Assert.Null(errored.Gate);                 // an error case has no gate
        Assert.Equal("backend down", errored.Error);
        Assert.Equal(2, report.PerMetric.Count);   // per-metric, not folded
    }

    [Fact]
    public void ParseSuiteSummaries_HandlesValidInvalidAndEmpty()
    {
        var summaries = PythonEngineService.ParseSuiteSummaries(ListJson);
        Assert.Equal(2, summaries.Count);
        Assert.True(summaries[0].Valid && summaries[0].CaseCount == 2);
        Assert.False(summaries[1].Valid);
        Assert.Empty(PythonEngineService.ParseSuiteSummaries(""));
        Assert.Empty(PythonEngineService.ParseSuiteSummaries("[]"));
    }

    [Theory]
    [InlineData("pass", "#16A34A")]
    [InlineData("warn", "#D97706")]
    [InlineData("block", "#DC2626")]
    [InlineData("error", "#DC2626")]   // errored case is red, not neutral
    [InlineData("", "#64748B")]        // unknown -> gray, never green
    [InlineData(null, "#64748B")]
    public void ColorForStatus_MapsHonestly(string? status, string expected)
    {
        Assert.Equal(expected, SuiteReport.ColorForStatus(status));
    }

    // --- view-model shaping -----------------------------------------------------------

    [Fact]
    public void ApplySuites_PopulatesAndSummarizes()
    {
        var vm = new MainWindowViewModel();
        vm.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        Assert.Equal(2, vm.Suites.Count);
        Assert.Contains("2 suite(s)", vm.SuitesStatus);
        Assert.Contains("1 invalid", vm.SuitesStatus);
    }

    [Fact]
    public void ApplySuites_EmptyShowsCreateNote()
    {
        var vm = new MainWindowViewModel();
        vm.ApplySuites(PythonEngineService.ParseSuiteSummaries("[]"));
        Assert.Empty(vm.Suites);
        Assert.Contains("No suites defined", vm.SuitesStatus);
    }

    [Fact]
    public void ApplySuiteReport_ShowsPerMetricCasesAndOverall()
    {
        var vm = new MainWindowViewModel();
        vm.ApplySuiteReport(PythonEngineService.ParseSuiteReport(ReportJson));
        Assert.True(vm.HasSuiteReport);
        Assert.Equal(2, vm.SuiteMetricRows.Count);
        Assert.Equal(2, vm.SuiteCaseRows.Count);
        Assert.Equal("BLOCK", vm.SuiteOverallStatus);
        Assert.Equal("#DC2626", vm.SuiteOverallColor);   // block -> red
        Assert.Contains("2 case(s)", vm.SuiteReportSummary);
    }

    [Fact]
    public void SetSuitesError_CollapsesToNeutralNeverGreen()
    {
        var vm = new MainWindowViewModel();
        vm.ApplySuiteReport(PythonEngineService.ParseSuiteReport(ReportJson));
        vm.SetSuitesError("Engine failed.");
        Assert.False(vm.HasSuiteReport);
        Assert.Empty(vm.SuiteCaseRows);
        Assert.Equal("#64748B", vm.SuiteOverallColor);   // neutral gray, not green
        Assert.Equal("Engine failed.", vm.SuiteReportSummary);
    }

    [Fact]
    public void CanRunSuite_RequiresValidSelectionNotBusy()
    {
        var vm = new MainWindowViewModel();
        vm.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));

        vm.SelectedSuite = vm.Suites.Single(s => !s.Valid);
        Assert.False(vm.CanRunSuite);                    // invalid suite -> cannot run

        vm.SelectedSuite = vm.Suites.Single(s => s.Valid);
        vm.IsSuitesBusy = true;
        Assert.False(vm.CanRunSuite);                    // busy -> cannot run
        // (HasActiveProject is false here with no project, so CanRunSuite stays false — the
        //  project gate is covered by project-lifecycle tests.)
    }

    [Fact]
    public void ResetSuites_ClearsEverything()
    {
        var vm = new MainWindowViewModel();
        vm.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        vm.ApplySuiteReport(PythonEngineService.ParseSuiteReport(ReportJson));
        vm.ResetSuites();
        Assert.Empty(vm.Suites);
        Assert.Empty(vm.SuiteCaseRows);
        Assert.False(vm.HasSuiteReport);
        Assert.Null(vm.SelectedSuite);
    }
}
