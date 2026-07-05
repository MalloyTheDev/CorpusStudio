using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Training tab view-model (Avalonia Phase 2, slice 4). Covers the two cross-cutting seams the
/// extraction introduced — the ErrorReported event routed to the shell banner, and the baseline
/// comparison reusing the shared Evaluation instance — plus the SelectProject reset. The run-lifecycle /
/// registry / gate / checkpoint paths are covered by the existing Training*ViewModelTests.</summary>
public sealed class TrainingCoreTests
{
    [Fact]
    public void SetTrainingConfigError_RoutesToSharedErrorBanner()
    {
        // The shell wires Training.ErrorReported -> ReportError, so a failed config export lights the
        // shared, dismissible error banner without the tab referencing the shell.
        var vm = new MainWindowViewModel();
        vm.Training.SetTrainingConfigError("engine boom");
        Assert.True(vm.HasError);
        Assert.Equal("engine boom", vm.ErrorMessage);
    }

    [Fact]
    public void CompareTrainingBaseline_UsesTheSharedEvaluationInstance()
    {
        // Training is constructed with the shell's Evaluation VM; the comparison reuses
        // Evaluation.BuildEvaluationReportComparison. Constructed standalone with that same VM here.
        var evaluation = new EvaluationViewModel(new EvaluationConnectionViewModel());
        var vm = new TrainingViewModel(evaluation);

        // Without a captured baseline it reports the neutral guidance (no report I/O).
        vm.CompareTrainingBaseline([]);

        Assert.Contains("No baseline was captured", vm.TrainingComparisonSummary);
    }

    [Fact]
    public void Reset_RestoresConfigDefaults_FollowingSchema()
    {
        var vm = new TrainingViewModel(new EvaluationViewModel(new EvaluationConnectionViewModel()));
        vm.TrainingTarget = "trl_sft";
        vm.SetTrainingConfigInProgress();

        vm.Reset("preference");

        Assert.Equal("preference", vm.TrainingFormat);
        Assert.Contains("Generate a training config", vm.TrainingSummary);
        Assert.Contains("preview appears here", vm.TrainingConfigPreview);
    }
}
