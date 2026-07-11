using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Read-only display reflections added for the production Training screen (Nocturne re-skin):
/// the derived Method selector, the Run-header run label + status-chip signals, and the regression-gate
/// callout gate. All must reflect real config/run/gate state and stay in a neutral empty state when the
/// backing data is absent — never fabricated.</summary>
public sealed class TrainingScreenDisplayTests
{
    private static TrainingViewModel NewVm() =>
        new(new EvaluationViewModel(new EvaluationConnectionViewModel()));

    // ---- Method selector (derived read-only from the real LoRA-rank config) ----

    [Fact]
    public void Method_DefaultLoraRank_ReflectsLora_NeverQlora()
    {
        var vm = NewVm(); // default TrainingLoraR = "16"
        Assert.Equal("lora", vm.TrainingMethod);
        Assert.True(vm.IsMethodLora);
        Assert.False(vm.IsMethodFull);
        // QLoRA cannot be honestly asserted (no 4-bit signal on the VM) — the segment stays inactive.
        Assert.False(vm.IsMethodQlora);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("")]
    [InlineData("not-a-number")]
    public void Method_ZeroOrUnparseableRank_ReflectsFull(string rank)
    {
        var vm = NewVm();
        vm.TrainingLoraR = rank;
        Assert.Equal("full", vm.TrainingMethod);
        Assert.True(vm.IsMethodFull);
        Assert.False(vm.IsMethodLora);
        Assert.False(vm.IsMethodQlora);
    }

    [Fact]
    public void Method_RaisesChangeNotification_WhenLoraRankChanges()
    {
        var vm = NewVm();
        var raised = new List<string?>();
        vm.PropertyChanged += (_, e) => raised.Add(e.PropertyName);

        vm.TrainingLoraR = "0";

        Assert.Contains(nameof(vm.TrainingMethod), raised);
        Assert.Contains(nameof(vm.IsMethodLora), raised);
        Assert.Contains(nameof(vm.IsMethodFull), raised);
    }

    // ---- Run header: run label + HasRunStarted ----

    [Fact]
    public void RunLabel_NeutralBeforeAnyLaunch()
    {
        var vm = NewVm();
        Assert.Equal(0, vm.TrainingRunId);
        Assert.False(vm.HasRunStarted);
        Assert.Equal("—", vm.TrainingRunLabel);
    }

    [Fact]
    public void RunLabel_ReflectsSessionRunCounter_AfterLaunch()
    {
        var vm = NewVm();
        vm.BeginTrainingRun();
        Assert.True(vm.HasRunStarted);
        Assert.Equal("run 1", vm.TrainingRunLabel);

        vm.BeginTrainingRun();
        Assert.Equal("run 2", vm.TrainingRunLabel);
    }

    // ---- Status chip signals ----

    [Fact]
    public void StatusChip_CompleteOnlyForCleanExit()
    {
        var vm = NewVm();
        Assert.False(vm.IsTrainingComplete); // idle
        Assert.False(vm.IsTrainingFailed);

        vm.BeginTrainingRun();
        Assert.False(vm.IsTrainingComplete); // running

        vm.CompleteTrainingRun(0);
        Assert.True(vm.IsTrainingComplete);
        Assert.False(vm.IsTrainingFailed);
    }

    [Fact]
    public void StatusChip_FailedExit_IsFailedNotComplete()
    {
        var vm = NewVm();
        vm.BeginTrainingRun();
        vm.CompleteTrainingRun(1);
        Assert.False(vm.IsTrainingComplete);
        Assert.True(vm.IsTrainingFailed);
    }

    [Fact]
    public void StatusChip_RunError_IsFailed()
    {
        var vm = NewVm();
        vm.BeginTrainingRun();
        vm.SetTrainingRunError("boom");
        Assert.True(vm.IsTrainingFailed);
        Assert.False(vm.IsTrainingComplete);
    }

    // ---- Regression-gate callout gate ----

    [Fact]
    public void HasGateWarning_FalseForNeutralPromptAndPass()
    {
        var vm = NewVm();
        Assert.False(vm.HasGateWarning); // neutral "Gate a run…" prompt

        vm.ApplyTrainingRunGate(new GateReport { OverallStatus = "pass" });
        Assert.False(vm.HasGateWarning);
    }

    [Fact]
    public void HasGateWarning_TrueForWarnBlockAndError()
    {
        var vm = NewVm();

        vm.ApplyTrainingRunGate(new GateReport
        {
            OverallStatus = "warn",
            Results = [new GateResult { Message = "loss regressed" }],
        });
        Assert.True(vm.HasGateWarning);

        vm.ApplyTrainingRunGate(new GateReport { OverallStatus = "block" });
        Assert.True(vm.HasGateWarning);

        vm.SetTrainingRunGateError("engine boom");
        Assert.True(vm.HasGateWarning);
    }
}
