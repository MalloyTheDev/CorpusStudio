using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Quality panel view-model (Avalonia Phase 2, slice 5). Covers the cross-cutting seam the
/// extraction introduced — the ErrorReported event routed to the shell banner — and the SelectProject
/// reset. The metric-grid / status / debt-trend paths are covered by QualityMetricTests + ExamplesTests.</summary>
public sealed class QualityCoreTests
{
    [Fact]
    public void SetQualityError_RoutesToSharedErrorBanner()
    {
        // The shell wires Quality.ErrorReported -> ReportError, so a failed quality run lights the
        // shared, dismissible error banner without the panel referencing the shell.
        var vm = new MainWindowViewModel();
        vm.Quality.SetQualityError("engine boom");
        Assert.True(vm.HasError);
        Assert.Equal("engine boom", vm.ErrorMessage);
    }

    [Fact]
    public void Reset_ClearsMetricsTrendAndSyntheticState()
    {
        var vm = new QualityViewModel();
        vm.QualitySummary = "stale summary from a previous project";

        vm.Reset();

        Assert.Contains("Quality checks will appear", vm.QualitySummary);
        Assert.False(vm.HasQualityMetrics);
        Assert.Empty(vm.QualityMetrics);
        Assert.Empty(vm.DebtTrend);
        Assert.Empty(vm.SyntheticPatternIssues);
        Assert.Null(vm.SelectedSyntheticPatternIssue);
    }
}
