using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class ArenaViewModelTests
{
    [Fact]
    public void ParseModelList_SplitsCommaAndNewline_TrimsAndDedupes()
    {
        var models = ArenaViewModel.ParseModelList("alpha, beta\nalpha\n  gamma  ");
        Assert.Equal(new[] { "alpha", "beta", "gamma" }, models);
    }

    [Fact]
    public void ParseModelList_EmptyInput_IsEmpty()
    {
        Assert.Empty(ArenaViewModel.ParseModelList("  ,\n , "));
    }

    private static ArenaReport TwoModelReport(bool judged)
    {
        return new ArenaReport
        {
            PromptCount = 1,
            Models = ["alpha", "beta"],
            Prompts = [new ArenaPromptItem { Id = "p1", Prompt = "Explain recursion." }],
            Responses =
            [
                new ArenaResponse { PromptId = "p1", Model = "alpha", Text = "A function calls itself." },
                new ArenaResponse { PromptId = "p1", Model = "beta", Text = "" },
            ],
            ModelSummaries =
            [
                new ArenaModelSummary { Model = "alpha", ResponseCount = 1, WinCount = judged ? 1 : 0, AverageJudgeScore = judged ? 90 : null },
                new ArenaModelSummary { Model = "beta", ResponseCount = 1, EmptyResponseCount = 1, WinCount = 0, AverageJudgeScore = judged ? 40 : null },
            ],
            JudgeModel = judged ? "judge" : null,
            Judgments = judged
                ? [new ArenaJudgment { PromptId = "p1", Winner = "alpha", Rationale = "clearer", Parsed = true }]
                : [],
        };
    }

    [Fact]
    public void ApplyArenaReport_Unjudged_ShowsResponsesAndNoWinner()
    {
        var vm = new ArenaViewModel();
        vm.ApplyArenaReport(TwoModelReport(judged: false));

        Assert.Contains("Arena: 2 model(s) × 1 prompt(s)", vm.ArenaSummary);
        Assert.DoesNotContain("judge:", vm.ArenaSummary);
        Assert.Contains("A function calls itself.", vm.ArenaSummary);
        Assert.Contains("(empty response)", vm.ArenaSummary);
        Assert.DoesNotContain("🏆", vm.ArenaSummary);
    }

    [Fact]
    public void ApplyArenaReport_Judged_MarksWinnerAndShowsScores()
    {
        var vm = new ArenaViewModel();
        vm.ApplyArenaReport(TwoModelReport(judged: true));

        Assert.Contains("judge: judge", vm.ArenaSummary);
        Assert.Contains("🏆 alpha:", vm.ArenaSummary);
        Assert.Contains("alpha — 1 win(s), avg 90", vm.ArenaSummary);
        Assert.Contains("judge: alpha — clearer", vm.ArenaSummary);
    }

    [Fact]
    public void ApplyArenaReport_UnparseableJudgment_ShownHonestly()
    {
        var vm = new ArenaViewModel();
        var report = TwoModelReport(judged: true);
        var unparseable = new ArenaReport
        {
            PromptCount = report.PromptCount,
            Models = report.Models,
            Prompts = report.Prompts,
            Responses = report.Responses,
            ModelSummaries = report.ModelSummaries,
            JudgeModel = "judge",
            Judgments = [new ArenaJudgment { PromptId = "p1", Winner = "", Parsed = false }],
        };
        vm.ApplyArenaReport(unparseable);

        Assert.Contains("could not parse judge output", vm.ArenaSummary);
    }

    [Fact]
    public void SetArenaError_ShowsMessage()
    {
        var vm = new ArenaViewModel();
        vm.SetArenaError("engine not found");
        Assert.Contains("Arena could not run", vm.ArenaSummary);
        Assert.Contains("engine not found", vm.ArenaSummary);
    }
}
