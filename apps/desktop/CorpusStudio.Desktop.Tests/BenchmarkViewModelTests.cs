using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class BenchmarkViewModelTests
{
    [Fact]
    public void GetBenchmarkModels_SplitsLinesAndCommas_TrimsAndDedupes()
    {
        var vm = new MainWindowViewModel
        {
            BenchmarkModelsInput = "llama3\n qwen2.5 , llama3\n\nmistral",
        };

        Assert.Equal(new[] { "llama3", "qwen2.5", "mistral" }, vm.GetBenchmarkModels());
    }

    [Fact]
    public void GetBenchmarkModels_EmptyInput_ReturnsEmpty()
    {
        var vm = new MainWindowViewModel { BenchmarkModelsInput = "   \n , \n" };
        Assert.Empty(vm.GetBenchmarkModels());
    }

    [Fact]
    public void ApplyBenchmarkReport_FormatsRankingAndCommonFailures()
    {
        var vm = new MainWindowViewModel();
        var report = new BenchmarkReport
        {
            Dataset = "d",
            ModelCount = 2,
            ExamplesTested = 3,
            BestModel = "good",
            WorstModel = "bad",
            ScoreSpread = 55,
            Models =
            [
                new BenchmarkModelSummary { Model = "good", Rank = 1, AverageScore = 90, PassRate = 100, FailedExamples = 0, ScoreDeltaVsBest = 0 },
                new BenchmarkModelSummary { Model = "bad", Rank = 2, AverageScore = 35, PassRate = 33.3, FailedExamples = 2, ScoreDeltaVsBest = -55 },
            ],
            CommonlyFailedExamples = ["row-2"],
        };

        vm.ApplyBenchmarkReport(report);

        Assert.Contains("Best: good", vm.BenchmarkSummary);
        Assert.Contains("#1 good", vm.BenchmarkSummary);
        Assert.Contains("#2 bad", vm.BenchmarkSummary);
        Assert.Contains("(-55 vs best)", vm.BenchmarkSummary);
        Assert.Contains("Failed by every model (1): row-2", vm.BenchmarkSummary);
    }

    [Fact]
    public void SetBenchmarkError_SurfacesMessage()
    {
        var vm = new MainWindowViewModel();
        vm.SetBenchmarkError("boom");
        Assert.Contains("boom", vm.BenchmarkSummary);
    }
}
