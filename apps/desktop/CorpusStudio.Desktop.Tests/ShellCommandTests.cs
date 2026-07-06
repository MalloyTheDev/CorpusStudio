using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The shell-navigation ICommands (Avalonia Phase 3 polish). Both heads can bind
/// Command="{Binding X}" instead of per-head code-behind, so these are a shared, testable seam.</summary>
public sealed class ShellCommandTests
{
    [Fact]
    public void ModeCommands_SwitchTheShellMode()
    {
        var vm = new MainWindowViewModel();

        vm.ShowStudioCommand.Execute(null);
        Assert.True(vm.IsStudio);

        vm.ShowFilesCommand.Execute(null);
        Assert.True(vm.IsFiles);

        vm.ShowStartCenterCommand.Execute(null);
        Assert.True(vm.IsStartCenter);
    }

    [Fact]
    public void PanelToggleCommands_ToggleVisibility()
    {
        var vm = new MainWindowViewModel();

        Assert.False(vm.ProblemsPanelVisible);
        vm.ToggleProblemsPanelCommand.Execute(null);
        Assert.True(vm.ProblemsPanelVisible);

        Assert.False(vm.OutputPanelVisible);
        vm.ToggleOutputPanelCommand.Execute(null);
        Assert.True(vm.OutputPanelVisible);
    }

    [Fact]
    public void DismissErrorCommand_ClearsTheErrorBanner()
    {
        var vm = new MainWindowViewModel();
        vm.Evaluation.SetEvaluationError("boom"); // lights the shared banner via ErrorReported
        Assert.True(vm.HasError);

        vm.DismissErrorCommand.Execute(null);
        Assert.False(vm.HasError);
    }

    [Fact]
    public void NavigationCommands_SwitchToStudioAndSelectTheTab()
    {
        var vm = new MainWindowViewModel();
        vm.ShowStartCenter();

        vm.GoToDebtCommand.Execute(null);
        Assert.True(vm.IsStudio);
        Assert.Equal((int)StudioTab.Debt, vm.SelectedStudioTabIndex);

        vm.GoToEvaluationCommand.Execute(null);
        Assert.Equal((int)StudioTab.Evaluation, vm.SelectedStudioTabIndex);

        vm.GoToTrainingCommand.Execute(null);
        Assert.Equal((int)StudioTab.Training, vm.SelectedStudioTabIndex);
    }
}
