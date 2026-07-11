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

    private static ArenaReport ThreeModelJudgedReport()
    {
        // alpha & beta each win one prompt; p3 is a judge tie (no winner). gamma wins none.
        return new ArenaReport
        {
            PromptCount = 3,
            Models = ["alpha", "beta", "gamma"],
            Prompts =
            [
                new ArenaPromptItem { Id = "p1", Prompt = "Reset my password" },
                new ArenaPromptItem { Id = "p2", Prompt = "Explain the warranty" },
                new ArenaPromptItem { Id = "p3", Prompt = "Downgrade my plan" },
            ],
            Responses =
            [
                new ArenaResponse { PromptId = "p1", Model = "alpha", Text = "a1" },
                new ArenaResponse { PromptId = "p1", Model = "beta", Text = "b1" },
                new ArenaResponse { PromptId = "p1", Model = "gamma", Text = "g1" },
            ],
            ModelSummaries =
            [
                new ArenaModelSummary { Model = "alpha", ResponseCount = 3, WinCount = 1, AverageJudgeScore = 80 },
                new ArenaModelSummary { Model = "beta", ResponseCount = 3, WinCount = 1, AverageJudgeScore = 70 },
                new ArenaModelSummary { Model = "gamma", ResponseCount = 3, WinCount = 0, AverageJudgeScore = 40 },
            ],
            JudgeModel = "judge",
            Judgments =
            [
                new ArenaJudgment { PromptId = "p1", Winner = "alpha", Parsed = true },
                new ArenaJudgment { PromptId = "p2", Winner = "beta", Parsed = true },
                new ArenaJudgment { PromptId = "p3", Winner = "", Parsed = true },
            ],
        };
    }

    [Fact]
    public void ApplyArenaReport_BuildsRankedStandings_WithHonestWinRateAndRecord()
    {
        var vm = new ArenaViewModel();
        vm.ApplyArenaReport(ThreeModelJudgedReport());

        Assert.True(vm.HasArenaReport);
        Assert.Equal(3, vm.Standings.Count);

        // Wins-desc, then avg-score-desc tie-break: alpha (avg 80) over beta (avg 70), gamma last.
        Assert.Equal(new[] { "alpha", "beta", "gamma" }, vm.Standings.Select(s => s.Model).ToArray());
        Assert.Equal(new[] { 1, 2, 3 }, vm.Standings.Select(s => s.Rank).ToArray());
        Assert.True(vm.Standings[0].IsFirst);
        Assert.False(vm.Standings[1].IsFirst);

        // alpha: 1 win of 3 judged, 1 tie → 1L. Win-rate 1/3 = 33%.
        var alpha = vm.Standings[0];
        Assert.Equal(1, alpha.Wins);
        Assert.Equal(1, alpha.Ties);
        Assert.Equal(1, alpha.Losses);
        Assert.Equal("33%", alpha.WinRateDisplay);
        Assert.Equal("1W · 1L · 1T", alpha.RecordDisplay);
        Assert.Equal(1d / 3d, alpha.WinRate, 3);

        // gamma: 0 wins, 1 tie → 2 losses.
        var gamma = vm.Standings[2];
        Assert.Equal("0%", gamma.WinRateDisplay);
        Assert.Equal("0W · 2L · 1T", gamma.RecordDisplay);
    }

    [Fact]
    public void ApplyArenaReport_BuildsHeadToHead_ForTopTwoModels()
    {
        var vm = new ArenaViewModel();
        vm.ApplyArenaReport(ThreeModelJudgedReport());

        Assert.True(vm.HasHeadToHead);
        Assert.Equal("HEAD-TO-HEAD · alpha vs beta", vm.HeadToHeadLabel);
        Assert.Equal(3, vm.HeadToHead.Count);

        Assert.Equal("alpha wins", vm.HeadToHead[0].ResultLabel);
        Assert.True(vm.HeadToHead[0].IsWin);
        Assert.Equal("beta wins", vm.HeadToHead[1].ResultLabel);
        Assert.True(vm.HeadToHead[1].IsWin);
        // p3 had no winner → a tie chip in the A-vs-B lens.
        Assert.Equal("tie", vm.HeadToHead[2].ResultLabel);
        Assert.False(vm.HeadToHead[2].IsWin);

        Assert.Contains("3 prompts · 3 models · pairwise judged", vm.ArenaResultsLine);
    }

    [Fact]
    public void ApplyArenaReport_Unjudged_StandingsAreNeutral_NoHeadToHead()
    {
        var vm = new ArenaViewModel();
        vm.ApplyArenaReport(TwoModelReport(judged: false));

        Assert.True(vm.HasArenaReport);
        Assert.False(vm.HasHeadToHead);
        Assert.Empty(vm.HeadToHead);
        Assert.Contains("not yet judged", vm.ArenaResultsLine);

        Assert.Equal(2, vm.Standings.Count);
        Assert.All(vm.Standings, s => Assert.False(s.IsJudged));
        Assert.Equal("—", vm.Standings[0].WinRateDisplay);
        Assert.Equal("not yet judged", vm.Standings[0].RecordDisplay);
        Assert.Equal(0d, vm.Standings[0].WinRate);
    }

    [Fact]
    public void SetArenaInProgress_And_Error_ResetResultsSurfaces()
    {
        var vm = new ArenaViewModel();
        vm.ApplyArenaReport(ThreeModelJudgedReport());
        Assert.True(vm.HasArenaReport);

        vm.SetArenaInProgress();
        Assert.False(vm.HasArenaReport);
        Assert.Empty(vm.Standings);
        Assert.Empty(vm.HeadToHead);
        Assert.False(vm.HasHeadToHead);

        vm.ApplyArenaReport(ThreeModelJudgedReport());
        vm.SetArenaError("boom");
        Assert.False(vm.HasArenaReport);
        Assert.Empty(vm.Standings);
    }

    [Fact]
    public void ArenaStandingItem_ComputesWinRateDisplayAndRecord()
    {
        var judged = new ArenaStandingItem { Rank = 1, Model = "m", Wins = 8, Losses = 3, Ties = 3, JudgedPrompts = 14 };
        Assert.True(judged.IsJudged);
        Assert.True(judged.IsFirst);
        Assert.Equal("57%", judged.WinRateDisplay); // 8/14 = 0.571
        Assert.Equal("8W · 3L · 3T", judged.RecordDisplay);

        var unjudged = new ArenaStandingItem { Rank = 2, Model = "m2", JudgedPrompts = 0 };
        Assert.False(unjudged.IsJudged);
        Assert.False(unjudged.IsFirst);
        Assert.Equal("—", unjudged.WinRateDisplay);
        Assert.Equal("not yet judged", unjudged.RecordDisplay);
        Assert.Equal(0d, unjudged.WinRate);
    }

    [Fact]
    public void ApplyArenaReport_BackendError_ShownPerResponseAndInSummary()
    {
        var report = new ArenaReport
        {
            PromptCount = 1,
            Models = ["alpha", "beta"],
            Prompts = [new ArenaPromptItem { Id = "p1", Prompt = "Explain recursion." }],
            Responses =
            [
                new ArenaResponse { PromptId = "p1", Model = "alpha", Text = "A function calls itself." },
                new ArenaResponse { PromptId = "p1", Model = "beta", Text = "", Error = "HTTP 503 Service Unavailable" },
            ],
            ModelSummaries =
            [
                new ArenaModelSummary { Model = "alpha", ResponseCount = 1 },
                new ArenaModelSummary { Model = "beta", ResponseCount = 1, ErrorCount = 1 },
            ],
        };

        var vm = new ArenaViewModel();
        vm.ApplyArenaReport(report);

        // The failed model's response shows the error, not a blank "(empty response)".
        Assert.Contains("backend error: HTTP 503 Service Unavailable", vm.ArenaSummary);
        Assert.DoesNotContain("(empty response)", vm.ArenaSummary);
        // The per-model summary line surfaces the error count.
        Assert.Contains("1 error(s)", vm.ArenaSummary);
        // The healthy model still shows its answer.
        Assert.Contains("A function calls itself.", vm.ArenaSummary);
    }
}
