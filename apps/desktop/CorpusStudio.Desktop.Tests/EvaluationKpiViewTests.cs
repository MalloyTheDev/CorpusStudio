using System.Collections.Generic;
using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Evaluation production-UI re-skin: the KPI stat-card display strings, the segmented
/// results-filter (counts + active states + commands), and the per-example score-bar model helpers.
/// All bind real report data — a pre-run VM shows the hidden/neutral state, never faked numbers.</summary>
public sealed class EvaluationKpiViewTests
{
    private static EvaluationExampleResult Result(string id, double score, bool passed) =>
        new() { ExampleId = id, Score = score, Passed = passed };

    private static EvaluationRunResult SampleRun(
        double averageScore = 78.4,
        int tested = 19,
        int failed = 5,
        string metric = "keyword_overlap",
        double threshold = 70.0)
    {
        var passed = tested - failed;
        var results = new List<EvaluationExampleResult>();
        for (var i = 0; i < passed; i++)
        {
            results.Add(Result($"pass-{i}", 90, passed: true));
        }

        for (var i = 0; i < failed; i++)
        {
            results.Add(Result($"fail-{i}", 40, passed: false));
        }

        var report = new EvaluationReport
        {
            Dataset = "support",
            Model = "llama3.1:8b",
            Metric = metric,
            ExamplesTested = tested,
            AverageScore = averageScore,
            FailedExamples = failed,
            RunSettings = new EvaluationRunSettings { ScoreThreshold = threshold },
            Results = results,
        };
        return new EvaluationRunResult(report, "report.json", "{}");
    }

    [Fact]
    public void PreRun_KpiCardsHiddenAndNeutral()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());

        Assert.False(vm.HasEvaluationReport);
        Assert.Equal("—", vm.AverageScoreDisplay);
        Assert.Equal("—", vm.PassRateDisplay);
        Assert.Equal(string.Empty, vm.PassRateDetail);
        Assert.Equal("—", vm.EvaluatedDisplay);
        Assert.Equal("—", vm.MetricDisplay);
    }

    [Fact]
    public void ApplyRunResult_PopulatesKpiCardsFromReport()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());

        vm.ApplyEvaluationRunResult(SampleRun());

        Assert.True(vm.HasEvaluationReport);
        Assert.Equal("78.4", vm.AverageScoreDisplay);
        // 14 of 19 passed -> 73.68% rounds to 74%.
        Assert.Equal("74%", vm.PassRateDisplay);
        Assert.Equal("14 / 19 ≥ 70", vm.PassRateDetail);
        Assert.Equal("19", vm.EvaluatedDisplay);
        Assert.Equal("keyword overlap", vm.MetricDisplay);
    }

    [Fact]
    public void ApplyRunResult_LlmJudgeMetric_ReadsAsFriendlyLabel()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());

        vm.ApplyEvaluationRunResult(SampleRun(metric: "llm_judge"));

        Assert.Equal("LLM judge", vm.MetricDisplay);
    }

    [Fact]
    public void ApplyRunResult_ZeroExamples_KeepsCardsHidden()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());

        vm.ApplyEvaluationRunResult(SampleRun(tested: 0, failed: 0));

        Assert.False(vm.HasEvaluationReport);
    }

    [Fact]
    public void ApplyRunResult_SetsSegmentedPassFailCounts()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());

        vm.ApplyEvaluationRunResult(SampleRun());

        Assert.Equal(14, vm.EvaluationPassCount);
        Assert.Equal(5, vm.EvaluationFailCount);
    }

    [Fact]
    public void Reset_RestoresPreRunNeutralState()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());
        vm.ApplyEvaluationRunResult(SampleRun());

        vm.Reset();

        Assert.False(vm.HasEvaluationReport);
        Assert.Equal("—", vm.AverageScoreDisplay);
        Assert.Equal(0, vm.EvaluationPassCount);
        Assert.Equal(0, vm.EvaluationFailCount);
    }

    [Fact]
    public void SegmentedActiveStates_TrackTheResultFilter()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());

        // Default is "All".
        Assert.True(vm.IsAllResultsFilterActive);
        Assert.False(vm.IsPassedResultsFilterActive);
        Assert.False(vm.IsFailedResultsFilterActive);

        vm.ShowPassedResultsCommand.Execute(null);
        Assert.Equal("Passed", vm.EvaluationResultFilter);
        Assert.False(vm.IsAllResultsFilterActive);
        Assert.True(vm.IsPassedResultsFilterActive);

        vm.ShowFailedResultsCommand.Execute(null);
        Assert.Equal("Failed", vm.EvaluationResultFilter);
        Assert.True(vm.IsFailedResultsFilterActive);

        vm.ShowAllResultsCommand.Execute(null);
        Assert.Equal("All", vm.EvaluationResultFilter);
        Assert.True(vm.IsAllResultsFilterActive);
    }

    [Fact]
    public void SegmentedActiveStates_RaisePropertyChanged()
    {
        var vm = new EvaluationViewModel(new EvaluationConnectionViewModel());
        var raised = new List<string?>();
        vm.PropertyChanged += (_, e) => raised.Add(e.PropertyName);

        vm.ShowPassedResultsCommand.Execute(null);

        Assert.Contains(nameof(vm.IsAllResultsFilterActive), raised);
        Assert.Contains(nameof(vm.IsPassedResultsFilterActive), raised);
        Assert.Contains(nameof(vm.IsFailedResultsFilterActive), raised);
    }

    [Theory]
    [InlineData(0, 0.0)]
    [InlineData(50, 65.0)]
    [InlineData(100, 130.0)]
    // Out-of-range scores clamp to the track so the fill can't overflow.
    [InlineData(140, 130.0)]
    [InlineData(-10, 0.0)]
    public void ScoreBarWidth_IsScoreFractionOfTrack(double score, double expectedWidth)
    {
        var result = new EvaluationExampleResult { Score = score, Passed = score >= 70 };
        Assert.Equal(expectedWidth, result.ScoreBarWidth, precision: 3);
    }

    [Fact]
    public void ScoreBarColor_OkWhenPassedElseWarn()
    {
        Assert.Equal("#6bbf9a", new EvaluationExampleResult { Score = 88, Passed = true }.ScoreBarColor);
        Assert.Equal("#d9a35f", new EvaluationExampleResult { Score = 58, Passed = false }.ScoreBarColor);
    }
}
