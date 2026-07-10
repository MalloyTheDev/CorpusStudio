using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Desktop surfacing of the engine <c>provenance-gate</c>: the pure JSON parse and the
/// pure VM render (badge + per-teacher summary). Locks in the honesty framing (unknown ≠ green pass)
/// and that the engine's snake_case report maps cleanly to the C# model.</summary>
public sealed class ProvenanceGateSurfacingTests
{
    private const string EngineJson = """
    {
      "role": "trainable_output_generator",
      "teacher_field": "meta.teacher",
      "target": "examples.jsonl",
      "total_rows": 699,
      "trainable_rows": 172,
      "quarantined_rows": 450,
      "unknown_rows": 77,
      "strict": false,
      "overall_status": "block",
      "summary": "BLOCK — 450 row(s) from restricted teacher(s) must be removed or authorized.",
      "buckets": [
        {"teacher": "claude-opus-4-8", "provider_id": "anthropic", "status": "quarantined", "row_count": 450, "note": "Anthropic terms restrict training"},
        {"teacher": "(untagged)", "provider_id": "", "status": "unknown", "row_count": 77, "note": "untagged"},
        {"teacher": "z-ai/glm-5.2", "provider_id": "z-ai", "status": "pass", "row_count": 172, "note": "MIT"}
      ]
    }
    """;

    [Fact]
    public void ParseProvenanceGateReport_MapsEngineSnakeCaseJson()
    {
        var report = PythonEngineService.ParseProvenanceGateReport(EngineJson);

        Assert.Equal("block", report.OverallStatus);
        Assert.Equal(699, report.TotalRows);
        Assert.Equal(450, report.QuarantinedRows);
        Assert.Equal(172, report.TrainableRows);
        Assert.Equal(77, report.UnknownRows);
        Assert.Equal(3, report.Buckets.Count);
        Assert.Equal("claude-opus-4-8", report.Buckets[0].Teacher);
        Assert.Equal("quarantined", report.Buckets[0].Status);
        Assert.Equal(450, report.Buckets[0].RowCount);
    }

    [Fact]
    public void ApplyProvenanceGateReport_BlockReport_SetsRedBadgeAndPerTeacherSummary()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyProvenanceGateReport(PythonEngineService.ParseProvenanceGateReport(EngineJson));

        Assert.True(vm.HasProvenanceGateReport);
        Assert.Equal("BLOCK", vm.ProvenanceGateStatus);
        Assert.Equal(GateReport.StatusColor("block"), vm.ProvenanceGateColor); // red, never green
        Assert.Contains("[QUARANTINED]", vm.ProvenanceGateSummary);
        Assert.Contains("claude-opus-4-8", vm.ProvenanceGateSummary);
        Assert.Contains("450 quarantined", vm.ProvenanceGateSummary);
        Assert.Contains("DECLARED teacher", vm.ProvenanceGateSummary); // honesty note
    }

    [Fact]
    public void ApplyProvenanceGateReport_UnknownStatus_StaysNeutralNeverGreen()
    {
        var vm = new MainWindowViewModel();
        // A malformed/absent status must not read as a pass — neutral gray, not green.
        vm.ApplyProvenanceGateReport(new ProvenanceGateReport { OverallStatus = "", TotalRows = 0 });

        Assert.Equal("UNKNOWN", vm.ProvenanceGateStatus);
        Assert.Equal(GateReport.StatusColor(null), vm.ProvenanceGateColor);
        Assert.NotEqual(GateReport.StatusColor("pass"), vm.ProvenanceGateColor);
    }

    [Fact]
    public void SetProvenanceGateError_CollapsesToNeutralNonReportState()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyProvenanceGateReport(PythonEngineService.ParseProvenanceGateReport(EngineJson));
        Assert.True(vm.HasProvenanceGateReport);

        vm.SetProvenanceGateError("engine exploded");

        Assert.False(vm.HasProvenanceGateReport);
        Assert.Equal(string.Empty, vm.ProvenanceGateStatus);
        Assert.Contains("engine exploded", vm.ProvenanceGateSummary);
    }
}
