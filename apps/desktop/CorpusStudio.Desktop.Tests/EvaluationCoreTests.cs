using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Evaluation tab core view-model (backend-cluster slice 3, PR 3b). Covers the two cross-cutting
/// seams the extraction introduced — the ErrorReported event routed to the shell banner, and the run
/// reading the shared connection child — plus the SelectProject reset and the moved report-comparison
/// guard in isolation. The run-result / filter / failure-view paths are covered by the engine + the
/// existing EvaluationFailureFilter / TrainingComparison tests.</summary>
public sealed class EvaluationCoreTests
{
    [Fact]
    public void SetEvaluationError_RoutesToSharedErrorBanner()
    {
        // The shell wires Evaluation.ErrorReported -> ReportError, so a failed run lights the shared,
        // dismissible error banner without the tab referencing the shell.
        var vm = new MainWindowViewModel();
        vm.Evaluation.SetEvaluationError("engine boom");
        Assert.True(vm.HasError);
        Assert.Equal("engine boom", vm.ErrorMessage);
    }

    [Fact]
    public void SetEvaluationInProgress_ReadsBackendAndModelFromConnectionChild()
    {
        // The core is composed from the shared EvaluationConnection instance, so the run's status line
        // reflects the backend/model configured on that child.
        var vm = new MainWindowViewModel();
        vm.EvaluationConnection.EvaluationBackend = "lm-studio";
        vm.EvaluationConnection.EvaluationModel = "phi-3";

        vm.Evaluation.SetEvaluationInProgress();

        Assert.Contains("Backend: lm-studio", vm.Evaluation.EvaluationSummary);
        Assert.Contains("Model: phi-3", vm.Evaluation.EvaluationSummary);
    }

    [Fact]
    public void Reset_ClearsRunReportAndResultState()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());
        vm.EvaluationSummary = "stale summary from a previous project";

        vm.Reset();

        Assert.Contains("Run a local model", vm.EvaluationSummary);
        Assert.Contains("appear here after a run", vm.EvaluationReportJson);
        Assert.Empty(vm.EvaluationResults);
        Assert.Empty(vm.EvaluationReportHistory);
        Assert.Null(vm.SelectedEvaluationReportHistoryItem);
    }

    [Fact]
    public void CompareSelectedEvaluationReports_WithoutSelection_ReportsAndReturnsFalse()
    {
        // The comparison action moved to the child with its guard branches; with nothing selected it
        // writes the neutral prompt and returns false (no report I/O).
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());

        var compared = vm.CompareSelectedEvaluationReports();

        Assert.False(compared);
        Assert.Contains("Select a saved evaluation report", vm.EvaluationComparisonSummary);
    }

    [Fact]
    public void CompareReportsCommand_RunsTheComparison()
    {
        // The command wraps CompareSelectedEvaluationReports so both heads can bind it.
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());
        vm.CompareReportsCommand.Execute(null);
        Assert.Contains("Select a saved evaluation report", vm.EvaluationComparisonSummary);
    }

    [Fact]
    public void ApplyFailureFilterCommand_WithoutSelection_ReportsError()
    {
        // The guard logic moved out of the desktop code-behind into ApplySelectedFailureFilter.
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());
        vm.ApplyFailureFilterCommand.Execute(null);
        Assert.Contains("Select a saved failure filter", vm.EvaluationFailureFilterSummary);
    }
}
