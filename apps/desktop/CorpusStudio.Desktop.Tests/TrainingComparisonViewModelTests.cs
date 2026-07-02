using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class TrainingComparisonViewModelTests
{
    private static EvaluationReportHistoryItem Item(
        string path,
        double averageScore,
        int failed,
        DateTime modified,
        string model = "base-model"
    )
    {
        var report = new EvaluationReport
        {
            Dataset = "d",
            Model = model,
            ExamplesTested = 10,
            AverageScore = averageScore,
            FailedExamples = failed,
        };
        return new EvaluationReportHistoryItem(report, path, "{}", modified);
    }

    private static readonly DateTime Earlier = new(2026, 7, 1, 10, 0, 0);
    private static readonly DateTime Later = new(2026, 7, 1, 12, 0, 0);

    [Fact]
    public void SetTrainingBaseline_Null_ExplainsNoBaseline()
    {
        var vm = new MainWindowViewModel();
        vm.SetTrainingBaseline(null);
        Assert.Contains("No baseline", vm.TrainingComparisonSummary);
    }

    [Fact]
    public void SetTrainingBaseline_CapturesAndGuides()
    {
        var vm = new MainWindowViewModel();
        vm.SetTrainingBaseline(Item("a.json", 60, 4, Earlier));
        Assert.NotNull(vm.TrainingBaselineReport);
        Assert.Contains("Baseline captured", vm.TrainingComparisonSummary);
        Assert.Contains("Compare vs baseline", vm.TrainingComparisonSummary);
    }

    [Fact]
    public void Compare_WithoutBaseline_Explains()
    {
        var vm = new MainWindowViewModel();
        vm.CompareTrainingBaseline([Item("a.json", 60, 4, Earlier)]);
        Assert.Contains("No baseline was captured", vm.TrainingComparisonSummary);
    }

    [Fact]
    public void Compare_OnlyBaselineInHistory_AsksForAfterEval()
    {
        var vm = new MainWindowViewModel();
        var baseline = Item("a.json", 60, 4, Earlier);
        vm.SetTrainingBaseline(baseline);
        vm.CompareTrainingBaseline([baseline]);
        Assert.Contains("No post-training evaluation", vm.TrainingComparisonSummary);
    }

    [Fact]
    public void Compare_NewestOtherReportOlderThanBaseline_Explains()
    {
        var vm = new MainWindowViewModel();
        vm.SetTrainingBaseline(Item("baseline.json", 60, 4, Later));
        vm.CompareTrainingBaseline(
            [Item("baseline.json", 60, 4, Later), Item("old.json", 50, 6, Earlier)]
        );
        Assert.Contains("older than the baseline", vm.TrainingComparisonSummary);
    }

    [Fact]
    public void Compare_WithAfterReport_ShowsDeltas()
    {
        var vm = new MainWindowViewModel();
        var baseline = Item("before.json", 60, 4, Earlier, model: "base");
        var after = Item("after.json", 75, 1, Later, model: "trained");
        vm.SetTrainingBaseline(baseline);

        // History is newest-first: after, then baseline.
        vm.CompareTrainingBaseline([after, baseline]);

        Assert.Contains("Before/after", vm.TrainingComparisonSummary);
        Assert.Contains("trained", vm.TrainingComparisonSummary);
        Assert.Contains("+15", vm.TrainingComparisonSummary); // average score delta
        Assert.Contains("-3", vm.TrainingComparisonSummary); // failed examples delta
    }
}
