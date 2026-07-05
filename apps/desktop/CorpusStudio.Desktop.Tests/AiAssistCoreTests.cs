using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>AI-Assist tab core view-model (backend-cluster slice 2, PR 3/3). Covers the two cross-cutting
/// seams the extraction introduced — the ErrorReported event routed to the shell banner, and the run
/// reading the shared connection child — plus the review-queue filter logic in isolation. The gate /
/// run-result / SelectProject-reset paths are covered by AiAssistCandidateGateTests.</summary>
public sealed class AiAssistCoreTests
{
    private static AiAssistReviewQueueItem Item(string id, string state) => new()
    {
        ReviewId = id,
        ReviewState = state,
        Action = "review",
        Model = "qwen",
        SuggestedJsonl = "{}",
    };

    [Fact]
    public void SetAiAssistError_RoutesToSharedErrorBanner()
    {
        // The shell wires AiAssist.ErrorReported -> ReportError, so a failed run lights the shared,
        // dismissible error banner without the tab referencing the shell.
        var vm = new MainWindowViewModel();
        vm.AiAssist.SetAiAssistError("engine boom");
        Assert.True(vm.HasError);
        Assert.Equal("engine boom", vm.ErrorMessage);
    }

    [Fact]
    public void SetAiAssistInProgress_ReadsBackendAndModelFromConnectionChild()
    {
        // The core is composed from the shared AiAssistConnection instance, so the run's status line
        // reflects the backend/model configured on that child.
        var vm = new MainWindowViewModel();
        vm.AiAssistConnection.AiAssistBackend = "lm-studio";
        vm.AiAssistConnection.AiAssistModel = "phi-3";

        vm.AiAssist.SetAiAssistInProgress();

        Assert.Contains("Backend: lm-studio", vm.AiAssist.AiAssistSummary);
        Assert.Contains("Model: phi-3", vm.AiAssist.AiAssistSummary);
    }

    [Fact]
    public void ReviewQueue_FilterAndSearchNarrowTheVisibleQueue()
    {
        var vm = new AiAssistViewModel(new AiAssistConnectionViewModel());
        vm.SetAiAssistReviewQueue(
        [
            Item("a", "review_required"),
            Item("b", "accepted"),
            Item("c", "rejected"),
        ]);
        Assert.Equal(3, vm.AiAssistReviewQueue.Count);
        Assert.Contains("3 of 3", vm.AiAssistQueueSummary);

        vm.AiAssistQueueFilter = "Accepted";
        Assert.Single(vm.AiAssistReviewQueue);
        Assert.Equal("b", vm.AiAssistReviewQueue[0].ReviewId);

        vm.AiAssistQueueFilter = "All";
        vm.AiAssistQueueSearch = "rejected";   // matches only item c's review state
        Assert.Single(vm.AiAssistReviewQueue);
        Assert.Equal("c", vm.AiAssistReviewQueue[0].ReviewId);
    }

    [Fact]
    public void ApplyActionPresets_ScopesActionsToSchema_AndKeepsCurrentIfValid()
    {
        var vm = new AiAssistViewModel(new AiAssistConnectionViewModel());
        vm.ApplyAiAssistActionPresets("preference");
        Assert.Contains("judge-preference-strength", vm.AiAssistActionPresets);

        vm.ApplyAiAssistActionPresets("raw_text");
        Assert.DoesNotContain("judge-preference-strength", vm.AiAssistActionPresets);
        Assert.Equal("review", vm.AiAssistAction); // fell back to a valid preset
    }
}
