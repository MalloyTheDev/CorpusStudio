using System.Collections.Generic;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The AI-Assist bulk-triage undo stack, migrated from the desktop code-behind (#247).</summary>
public sealed class AiAssistBulkUndoTests
{
    private static IAiAssistViewModel NewAiAssist() => new MainWindowViewModel().AiAssist;

    private static Dictionary<string, string> Step(string id, string state) =>
        new() { [id] = state };

    [Fact]
    public void PushThenPeek_ReturnsTheLastSnapshot_AndDepthTracks()
    {
        var vm = NewAiAssist();
        Assert.Equal(0, vm.BulkUndoStackDepth);
        Assert.Null(vm.PeekBulkUndoStep());

        vm.PushBulkUndoStep(Step("a", "accepted"));
        vm.PushBulkUndoStep(Step("b", "rejected"));

        Assert.Equal(2, vm.BulkUndoStackDepth);
        Assert.Equal("rejected", vm.PeekBulkUndoStep()!["b"]);
    }

    [Fact]
    public void PushEmptySnapshot_IsIgnored()
    {
        var vm = NewAiAssist();
        vm.PushBulkUndoStep(new Dictionary<string, string>());
        Assert.Equal(0, vm.BulkUndoStackDepth);
    }

    [Fact]
    public void RemoveLast_PopsAndClear_Empties()
    {
        var vm = NewAiAssist();
        vm.PushBulkUndoStep(Step("a", "accepted"));
        vm.PushBulkUndoStep(Step("b", "rejected"));

        vm.RemoveLastBulkUndoStep();
        Assert.Equal(1, vm.BulkUndoStackDepth);
        Assert.Equal("accepted", vm.PeekBulkUndoStep()!["a"]);

        vm.ClearBulkUndoStack();
        Assert.Equal(0, vm.BulkUndoStackDepth);
        Assert.Null(vm.PeekBulkUndoStep());
    }

    [Fact]
    public void RemoveLast_OnEmpty_IsSafe()
    {
        var vm = NewAiAssist();
        vm.RemoveLastBulkUndoStep(); // no throw
        Assert.Equal(0, vm.BulkUndoStackDepth);
    }

    [Fact]
    public void Push_IsCappedAtTwenty_EvictingOldest()
    {
        var vm = NewAiAssist();
        for (var i = 0; i < 25; i++)
        {
            vm.PushBulkUndoStep(Step("k", $"state-{i}"));
        }

        Assert.Equal(20, vm.BulkUndoStackDepth);          // capped
        Assert.Equal("state-24", vm.PeekBulkUndoStep()!["k"]); // newest kept
    }

    [Fact]
    public void Push_StoresADefensiveCopy()
    {
        var vm = NewAiAssist();
        var mutable = Step("a", "accepted");
        vm.PushBulkUndoStep(mutable);
        mutable["a"] = "mutated";

        Assert.Equal("accepted", vm.PeekBulkUndoStep()!["a"]); // snapshot unaffected
    }
}
