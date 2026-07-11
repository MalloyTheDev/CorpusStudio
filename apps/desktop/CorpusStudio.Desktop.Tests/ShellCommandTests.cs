using System.Threading.Tasks;
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

    [Fact]
    public void SelectStudioTabCommand_SwitchesToStudioAndSelectsTheNamedTab()
    {
        // The grouped-IA nav rows (slice 2) bind this ONE command with the StudioTab name as the
        // CommandParameter — the shared target that drives the Avalonia content-switcher.
        var vm = new MainWindowViewModel();
        vm.ShowStartCenter();

        vm.SelectStudioTabCommand.Execute("Artifacts");
        Assert.True(vm.IsStudio);
        Assert.Equal((int)StudioTab.Artifacts, vm.SelectedStudioTabIndex);

        vm.SelectStudioTabCommand.Execute("Quarantine");
        Assert.Equal((int)StudioTab.Quarantine, vm.SelectedStudioTabIndex);

        // An unknown/garbage parameter is a safe no-op — the selection stays put, never throws.
        vm.SelectStudioTabCommand.Execute("NotARealTab");
        Assert.Equal((int)StudioTab.Quarantine, vm.SelectedStudioTabIndex);
    }

    [Fact]
    public void ContextBar_TitleTracksSelectedTab_AndOverflowToggles()
    {
        // slice 4: the context bar's title/subtitle are derived from the selected tab, and the
        // "⋯ More" overflow is a simple toggle (collapsed by default).
        var vm = new MainWindowViewModel();

        vm.SelectStudioTabCommand.Execute("Debt");
        Assert.Equal("Dataset Debt", vm.StudioViewTitle);
        Assert.False(string.IsNullOrWhiteSpace(vm.StudioViewSubtitle));

        vm.SelectStudioTabCommand.Execute("Training");
        Assert.Equal("Training", vm.StudioViewTitle);

        Assert.False(vm.ProjectActionsExpanded);
        vm.ToggleProjectActionsCommand.Execute(null);
        Assert.True(vm.ProjectActionsExpanded);
        vm.ToggleProjectActionsCommand.Execute(null);
        Assert.False(vm.ProjectActionsExpanded);
    }

    [Fact]
    public void AsyncRelayCommandOfT_PassesTypedParameter_AndCoercesMismatchToNull()
    {
        string? seen = "unset";
        var command = new AsyncRelayCommand<string>(p => { seen = p; return Task.CompletedTask; });

        command.Execute("hello");
        Assert.Equal("hello", seen);

        command.Execute(42); // wrong type → coerced to default(string) = null, not a throw
        Assert.Null(seen);
    }

    [Fact]
    public void AsyncRelayCommandOfT_GuardBlocksExecuteWhenCanExecuteFalse()
    {
        var ran = false;
        var command = new AsyncRelayCommand<string>(_ => { ran = true; return Task.CompletedTask; }, _ => false);

        Assert.False(command.CanExecute("x"));
        command.Execute("x");
        Assert.False(ran); // a false guard blocks the run
    }
}
