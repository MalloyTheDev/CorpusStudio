using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>AI-Assist saved rewrite-batches sub-view-model (backend-cluster slice 2, PR 1). Isolated
/// list/selection/summary + the last-prepared handoff, plus the per-project reset forwarded by the
/// shell. The Resume→Writing-Studio bridge stays on the shell (ResumeAiAssistRewriteBatch).</summary>
public sealed class AiAssistRewriteBatchesTests
{
    private static AiAssistRewriteBatch Batch(string id, params int[] rows) => new()
    {
        BatchId = id,
        SchemaId = "instruction",
        Action = "rewrite-output",
        RowNumbers = [.. rows],
        Instruction = "rewrite",
        SourceDraft = "{}",
    };

    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    [Fact]
    public void SetBatches_PopulatesSelectsFirstAndSummarizes()
    {
        var vm = new AiAssistRewriteBatchesViewModel();
        vm.SetAiAssistRewriteBatches([Batch("b1", 3, 5), Batch("b2", 7)]);
        Assert.Equal(2, vm.AiAssistRewriteBatches.Count);
        Assert.Same(vm.AiAssistRewriteBatches[0], vm.SelectedAiAssistRewriteBatch);
        Assert.Contains("Saved rewrite batches: 2", vm.AiAssistRewriteBatchSummary);
    }

    [Fact]
    public void SetBatches_EmptyShowsNoneMessage()
    {
        var vm = new AiAssistRewriteBatchesViewModel();
        vm.SetAiAssistRewriteBatches([]);
        Assert.Empty(vm.AiAssistRewriteBatches);
        Assert.Contains("No prepared rewrite batches", vm.AiAssistRewriteBatchSummary);
    }

    [Fact]
    public void SelectingBatch_SummarizesRows()
    {
        var vm = new AiAssistRewriteBatchesViewModel();
        vm.SetAiAssistRewriteBatches([Batch("b1", 3, 5), Batch("b2", 7)]); // b1 auto-selected
        vm.SelectedAiAssistRewriteBatch = vm.AiAssistRewriteBatches[1];    // switch to b2
        Assert.Contains("Selected rewrite batch for rows 7", vm.AiAssistRewriteBatchSummary);
    }

    [Fact]
    public void LastPrepared_HandoffIsConsumedOnce()
    {
        var vm = new AiAssistRewriteBatchesViewModel();
        Assert.False(vm.TryGetLastPreparedAiAssistRewriteBatch(out _, out var err));
        Assert.Contains("Prepare a synthetic batch rewrite", err);

        vm.SetLastPrepared(Batch("prep", 2));
        Assert.True(vm.TryGetLastPreparedAiAssistRewriteBatch(out var batch, out _));
        Assert.Equal("prep", batch.BatchId);
    }

    [Fact]
    public void FormatRowNumbers_IsDistinctOrderedAndPositive()
    {
        Assert.Equal("2, 3, 5", AiAssistRewriteBatchesViewModel.FormatRowNumbers([5, 3, 2, 3, 0, -1]));
        Assert.Equal("none", AiAssistRewriteBatchesViewModel.FormatRowNumbers([0, -4]));
    }

    [Fact]
    public void SelectProject_ResetsBatchesState()
    {
        var vm = new MainWindowViewModel();
        vm.RewriteBatches.SetAiAssistRewriteBatches([Batch("b1", 1)]);
        vm.RewriteBatches.SetLastPrepared(Batch("prep", 2));

        vm.SelectProject(Project("other"));

        Assert.Empty(vm.RewriteBatches.AiAssistRewriteBatches);
        Assert.Null(vm.RewriteBatches.SelectedAiAssistRewriteBatch);
        Assert.False(vm.RewriteBatches.TryGetLastPreparedAiAssistRewriteBatch(out _, out _)); // last-prepared cleared
    }
}
