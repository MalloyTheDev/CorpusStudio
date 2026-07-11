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

    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

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

    [Fact]
    public void SuiteSummary_DisplayHelpers_ReflectRealFieldsHonestly()
    {
        var summaries = PythonEngineService.ParseSuiteSummaries(ListJson);
        var valid = summaries.Single(s => s.Valid);
        var invalid = summaries.Single(s => !s.Valid);

        // Valid suite: neutral "valid" badge (NEVER a green "pass" — suite-list has no run score),
        // not-invalid, real case-count token.
        Assert.False(valid.IsInvalid);
        Assert.Equal("valid", valid.StatusBadgeText);
        Assert.Equal("2 case(s)", valid.CaseCountLabel);

        // Malformed suite: warn "invalid" badge + IsInvalid true so it can't read as healthy.
        Assert.True(invalid.IsInvalid);
        Assert.Equal("invalid", invalid.StatusBadgeText);
        Assert.Equal("0 case(s)", invalid.CaseCountLabel);
    }

    [Fact]
    public void HasSuites_TracksTheRegistry()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.Suites.HasSuites);                                  // empty at start

        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        Assert.True(vm.Suites.HasSuites);                                   // populated -> card list

        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries("[]"));
        Assert.False(vm.Suites.HasSuites);                                  // cleared -> empty state

        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        vm.Suites.Reset();
        Assert.False(vm.Suites.HasSuites);                                  // reset -> empty state
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
        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        Assert.Equal(2, vm.Suites.Suites.Count);
        Assert.Contains("2 suite(s)", vm.Suites.SuitesStatus);
        Assert.Contains("1 invalid", vm.Suites.SuitesStatus);
    }

    [Fact]
    public void ApplySuites_EmptyShowsCreateNote()
    {
        var vm = new MainWindowViewModel();
        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries("[]"));
        Assert.Empty(vm.Suites.Suites);
        Assert.Contains("No suites defined", vm.Suites.SuitesStatus);
    }

    [Fact]
    public void ApplySuiteReport_ShowsPerMetricCasesAndOverall()
    {
        var vm = new MainWindowViewModel();
        vm.Suites.ApplySuiteReport(PythonEngineService.ParseSuiteReport(ReportJson));
        Assert.True(vm.Suites.HasSuiteReport);
        Assert.Equal(2, vm.Suites.SuiteMetricRows.Count);
        Assert.Equal(2, vm.Suites.SuiteCaseRows.Count);
        Assert.Equal("BLOCK", vm.Suites.SuiteOverallStatus);
        Assert.Equal("#DC2626", vm.Suites.SuiteOverallColor);   // block -> red
        Assert.Contains("2 case(s)", vm.Suites.SuiteReportSummary);
    }

    [Fact]
    public void SetSuitesError_CollapsesToNeutralNeverGreen()
    {
        var vm = new MainWindowViewModel();
        vm.Suites.ApplySuiteReport(PythonEngineService.ParseSuiteReport(ReportJson));
        vm.Suites.SetSuitesError("Engine failed.");
        Assert.False(vm.Suites.HasSuiteReport);
        Assert.Empty(vm.Suites.SuiteCaseRows);
        Assert.Equal("#64748B", vm.Suites.SuiteOverallColor);   // neutral gray, not green
        Assert.Equal("Engine failed.", vm.Suites.SuiteReportSummary);
    }

    [Fact]
    public void CanRunSuite_RequiresValidSelectionNotBusy()
    {
        var vm = new MainWindowViewModel();
        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));

        vm.Suites.SelectedSuite = vm.Suites.Suites.Single(s => !s.Valid);
        Assert.False(vm.Suites.CanRunSuite);                    // invalid suite -> cannot run

        vm.Suites.SelectedSuite = vm.Suites.Suites.Single(s => s.Valid);
        vm.Suites.IsSuitesBusy = true;
        Assert.False(vm.Suites.CanRunSuite);                    // busy -> cannot run
        // (HasActiveProject is false here with no project, so CanRunSuite stays false — the
        //  project gate is covered by SelectProject_ResetsSuitesAndSyncsRunGate.)
    }

    [Fact]
    public void Reset_ClearsEverything()
    {
        var vm = new MainWindowViewModel();
        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        vm.Suites.ApplySuiteReport(PythonEngineService.ParseSuiteReport(ReportJson));
        vm.Suites.Reset();
        Assert.Empty(vm.Suites.Suites);
        Assert.Empty(vm.Suites.SuiteCaseRows);
        Assert.False(vm.Suites.HasSuiteReport);
        Assert.Null(vm.Suites.SelectedSuite);
    }

    // --- per-project lifecycle (Phase-2 extraction: the shell forwards Reset() and pushes
    //     HasActiveProject down so the Run gate tracks the open project) -------------------

    [Fact]
    public void SelectProject_ResetsSuitesAndSyncsRunGate()
    {
        // A project switch must clear the previous project's suites/report AND push the new
        // project-open state so CanRunSuite (the Run gate) reflects it — the cross-cutting
        // concern introduced when the tab moved out of the shell.
        var vm = new MainWindowViewModel();
        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        vm.Suites.ApplySuiteReport(PythonEngineService.ParseSuiteReport(ReportJson));

        vm.SelectProject(Project("p"));

        Assert.Empty(vm.Suites.Suites);              // reset cleared the previous project's suites
        Assert.False(vm.Suites.HasSuiteReport);
        Assert.True(vm.Suites.HasActiveProject);     // project-open flag pushed down by the shell

        // With a project open, a valid selected, non-busy suite can now run.
        vm.Suites.ApplySuites(PythonEngineService.ParseSuiteSummaries(ListJson));
        vm.Suites.SelectedSuite = vm.Suites.Suites.Single(s => s.Valid);
        Assert.True(vm.Suites.CanRunSuite);
    }
}
