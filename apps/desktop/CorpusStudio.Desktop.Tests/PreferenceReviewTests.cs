using System.Collections.Generic;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Preference Review tab view-model (Phase-2 extraction). Contrast ranking/filtering + pane
/// population in isolation, plus the two cross-cutting concerns wired by the shell: the ActiveSchemaId
/// push-down + SetItems/Reset lifecycle, and the AI-Assist handoff bridge that reaches into the tab.</summary>
public sealed class PreferenceReviewTests
{
    // overlap 1.0 -> weak; ~0.75 -> moderate; 0.0 -> strong (see BuildPreferenceContrastMetrics).
    private const string WeakJson =
        """{"prompt":"P1","chosen":"the quick brown fox jumps over","rejected":"the quick brown fox jumps over!"}""";
    private const string ModerateJson =
        """{"prompt":"P2","chosen":"the quick brown fox","rejected":"the quick brown cat","reason":"tone"}""";
    private const string StrongJson =
        """{"prompt":"P3","chosen":"alpha beta gamma delta","rejected":"xxxx yyyy zzzz wwww"}""";

    private static SavedExampleItem Example(int row, string json) => new(row, $"preview {row}", json);

    private static List<SavedExampleItem> ThreeContrastExamples() =>
    [
        Example(1, StrongJson),
        Example(2, WeakJson),
        Example(3, ModerateJson),
    ];

    private static PreferenceReviewViewModel PreferenceVm()
        => new() { ActiveSchemaId = "preference" };

    private static DatasetProjectListItem Project(string schemaId, string id = "p1")
        => new(
            new DatasetProject(id, id, schemaId, new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    // --- ranking / filtering (isolated VM) --------------------------------------------

    [Fact]
    public void SetItems_RanksWeakPairsFirstAndSummarizes()
    {
        var vm = PreferenceVm();
        vm.SetItems(ThreeContrastExamples());

        Assert.Equal(3, vm.PreferenceReviewItems.Count);
        Assert.Equal("weak", vm.PreferenceReviewItems[0].Contrast);       // weakest contrast surfaces first
        Assert.Equal("moderate", vm.PreferenceReviewItems[1].Contrast);
        Assert.Equal("strong", vm.PreferenceReviewItems[2].Contrast);
        Assert.Contains("1 weak, 1 moderate, 1 strong", vm.PreferenceRankingSummary);
        Assert.Contains("showing 3 of 3", vm.PreferenceRankingSummary);
    }

    [Fact]
    public void SetItems_NonPreferenceSchema_BuildsNothing()
    {
        var vm = new PreferenceReviewViewModel { ActiveSchemaId = "chat" };
        vm.SetItems([Example(1, WeakJson)]);
        Assert.Empty(vm.PreferenceReviewItems);
        Assert.Contains("available for preference projects", vm.PreferenceRankingSummary);
    }

    [Fact]
    public void SetItems_SkipsRowsMissingRequiredFields()
    {
        var vm = PreferenceVm();
        vm.SetItems(
        [
            Example(1, WeakJson),
            Example(2, """{"prompt":"only prompt"}"""),   // missing chosen/rejected -> skipped
            Example(3, "not json at all"),                // invalid JSON -> skipped
        ]);
        Assert.Single(vm.PreferenceReviewItems);
        Assert.Equal(1, vm.PreferenceReviewItems[0].RowNumber);
    }

    [Fact]
    public void SelectedItem_PopulatesPromptChosenRejectedReasonPanes()
    {
        var vm = PreferenceVm();
        vm.SetItems([Example(2, ModerateJson)]);   // auto-selected (only item)

        Assert.NotNull(vm.SelectedPreferenceReviewItem);
        Assert.Equal("the quick brown fox", vm.PreferenceChosenText);
        Assert.Equal("the quick brown cat", vm.PreferenceRejectedText);
        Assert.Equal("tone", vm.PreferenceReasonText);
        Assert.Contains("Example 2", vm.PreferenceReviewSummary);
    }

    [Fact]
    public void ContrastFilter_ShowsOnlyMatchingContrast()
    {
        var vm = PreferenceVm();
        vm.SetItems(ThreeContrastExamples());

        vm.PreferenceContrastFilter = "Strong";
        Assert.Single(vm.PreferenceReviewItems);
        Assert.Equal("strong", vm.PreferenceReviewItems[0].Contrast);
        Assert.Contains("showing 1 of 3", vm.PreferenceRankingSummary);
    }

    [Fact]
    public void ContrastLine_DerivesSecondLineFromContrastBand()
    {
        var vm = PreferenceVm();
        vm.SetItems(ThreeContrastExamples());

        // The pair-list item template's second line is bound to ContrastLine — a real, honest
        // derivation of the Contrast field (weak/moderate/strong), never a relabel to high/med/low.
        Assert.Equal("weak contrast", vm.PreferenceReviewItems[0].ContrastLine);
        Assert.Equal("moderate contrast", vm.PreferenceReviewItems[1].ContrastLine);
        Assert.Equal("strong contrast", vm.PreferenceReviewItems[2].ContrastLine);
    }

    // --- export formatting ------------------------------------------------------------

    [Fact]
    public void ApplyPreferenceTrainingExport_ReportsRowsDroppedAndWarnings()
    {
        var vm = PreferenceVm();
        vm.ApplyPreferenceTrainingExport(new PreferenceExportResult
        {
            Format = "dpo", OutputRows = 7, OutputPath = "C:/out.jsonl",
            DroppedDegenerate = 2,
            Warnings = { "2 pairs had identical chosen/rejected" },
        });
        Assert.Contains("Exported 7 row(s) as dpo: C:/out.jsonl", vm.PreferenceReviewSummary);
        Assert.Contains("Dropped 2 degenerate pair(s).", vm.PreferenceReviewSummary);
        Assert.Contains("- 2 pairs had identical chosen/rejected", vm.PreferenceReviewSummary);
    }

    [Fact]
    public void ApplyPreferenceRankingExport_AndError_Format()
    {
        var vm = PreferenceVm();
        vm.ApplyPreferenceRankingExport("C:/rank.jsonl", 5);
        Assert.Contains("Exported 5 visible preference ranking item(s)", vm.PreferenceReviewSummary);

        vm.SetPreferenceRankingExportError("disk full");
        Assert.Contains("export failed", vm.PreferenceReviewSummary);
        Assert.Contains("disk full", vm.PreferenceReviewSummary);
    }

    [Fact]
    public void Reset_ClearsPairsSelectionAndPanes()
    {
        var vm = PreferenceVm();
        vm.SetItems(ThreeContrastExamples());
        vm.PreferenceContrastFilter = "Strong";

        vm.Reset();

        Assert.Empty(vm.PreferenceReviewItems);
        Assert.Null(vm.SelectedPreferenceReviewItem);
        Assert.Equal("All", vm.PreferenceContrastFilter);
        Assert.Contains("Select a saved preference example", vm.PreferencePromptText);
    }

    // --- cross-cutting: shell lifecycle + AI-Assist bridge ----------------------------

    [Fact]
    public void SelectProject_PushesSchemaDown_ThenSetExamplesBuildsPairs()
    {
        var vm = new MainWindowViewModel();
        vm.SelectProject(Project("preference"));
        Assert.Equal("preference", vm.PreferenceReview.ActiveSchemaId);   // schema pushed down by the shell

        vm.SetExamples(ThreeContrastExamples());
        Assert.Equal(3, vm.PreferenceReview.PreferenceReviewItems.Count);

        // Switching to a non-preference project resets and re-gates the tab.
        vm.SelectProject(Project("chat", "p2"));
        Assert.Empty(vm.PreferenceReview.PreferenceReviewItems);
        Assert.Equal("chat", vm.PreferenceReview.ActiveSchemaId);
    }

    [Fact]
    public void PreparePreferenceJudgeReview_ReadsChildSelectionAndDrivesAiAssist()
    {
        // The AI-Assist handoff stays on the shell but reaches into the extracted tab for the
        // selected pair and to set the review summary.
        var vm = new MainWindowViewModel();
        vm.SelectProject(Project("preference"));
        vm.SetExamples([Example(2, WeakJson)]);   // one pair, auto-selected

        var ok = vm.PreparePreferenceJudgeReview();

        Assert.True(ok);
        Assert.Contains("Prepared Example 2", vm.PreferenceReview.PreferenceReviewSummary);
        Assert.Contains("Judge preference strength", vm.AiAssist.AiAssistInstruction);
    }
}
