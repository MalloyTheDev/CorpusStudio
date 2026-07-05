using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class TrainingRunViewModelTests
{
    private static TrainingConfigExportResult ConfigWithLaunch() => new()
    {
        Target = "axolotl_yaml",
        OutputPath = "C:/proj/exports/x/config.yaml",
        Launch = new TrainingLaunchPlan
        {
            Target = "axolotl_yaml",
            Command = "accelerate launch -m axolotl.cli.train \"config.yaml\"",
            Argv = ["accelerate", "launch", "-m", "axolotl.cli.train", "config.yaml"],
        },
    };

    [Fact]
    public void CanLaunchTraining_FalseUntilConfigGenerated()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.Training.CanLaunchTraining);

        vm.Training.ApplyTrainingConfigExportResult(ConfigWithLaunch());

        Assert.True(vm.Training.CanLaunchTraining);
        Assert.Equal(5, vm.Training.TrainingLaunchArgv.Count);
        Assert.EndsWith("exports\\x", vm.Training.TrainingLaunchWorkingDirectory.Replace('/', '\\'));
    }

    [Fact]
    public void RunLifecycle_TracksStatusAndLog()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingConfigExportResult(ConfigWithLaunch());

        vm.Training.BeginTrainingRun();
        Assert.True(vm.Training.IsTrainingRunning);
        Assert.False(vm.Training.CanLaunchTraining); // cannot launch while running

        vm.Training.AppendTrainingRunLog("epoch 1");
        vm.Training.AppendTrainingRunLog("epoch 2");
        Assert.Contains("epoch 1", vm.Training.TrainingRunLog);
        Assert.Contains("epoch 2", vm.Training.TrainingRunLog);

        vm.Training.CompleteTrainingRun(0);
        Assert.False(vm.Training.IsTrainingRunning);
        Assert.Contains("Completed", vm.Training.TrainingRunStatus);
        Assert.True(vm.Training.CanLaunchTraining); // launchable again
    }

    [Fact]
    public void CompleteTrainingRun_NonZero_MarksFailed()
    {
        var vm = new MainWindowViewModel();
        vm.Training.BeginTrainingRun();
        vm.Training.CompleteTrainingRun(1);
        Assert.Contains("Failed", vm.Training.TrainingRunStatus);
    }

    [Fact]
    public void Cancel_SetsCancelledStatus()
    {
        var vm = new MainWindowViewModel();
        vm.Training.BeginTrainingRun();
        vm.Training.SetTrainingRunCancelled();
        Assert.False(vm.Training.IsTrainingRunning);
        Assert.Equal("Cancelled", vm.Training.TrainingRunStatus);
    }

    [Fact]
    public void BeginTrainingRun_IncrementsRunId()
    {
        var vm = new MainWindowViewModel();
        Assert.NotEqual(vm.Training.BeginTrainingRun(), vm.Training.BeginTrainingRun());
    }

    [Fact]
    public void AppendTrainingRunLogBatch_DropsStaleRunId()
    {
        var vm = new MainWindowViewModel();
        var runId = vm.Training.BeginTrainingRun();

        vm.Training.AppendTrainingRunLogBatch(runId, new[] { "current-line" });
        vm.Training.AppendTrainingRunLogBatch(runId - 1, new[] { "stale-line" });

        Assert.Contains("current-line", vm.Training.TrainingRunLog);
        Assert.DoesNotContain("stale-line", vm.Training.TrainingRunLog);
    }

    [Fact]
    public void AppendTrainingRunLogBatch_AppendsAndCaps()
    {
        var vm = new MainWindowViewModel();
        var runId = vm.Training.BeginTrainingRun();
        vm.Training.AppendTrainingRunLogBatch(runId, new[] { "a", "b", "c" });
        Assert.Contains("a", vm.Training.TrainingRunLog);
        Assert.Contains("c", vm.Training.TrainingRunLog);
    }

    [Fact]
    public void AppendTrainingRunLog_CapsAtMaxLines()
    {
        var vm = new MainWindowViewModel();
        vm.Training.BeginTrainingRun();
        var total = TrainingViewModel.TrainingLogMaxLines + 500;
        for (var index = 0; index < total; index++)
        {
            vm.Training.AppendTrainingRunLog($"line-{index}");
        }

        var lineCount = vm.Training.TrainingRunLog.Split('\n').Length;
        Assert.True(lineCount <= TrainingViewModel.TrainingLogMaxLines);
        Assert.Contains($"line-{total - 1}", vm.Training.TrainingRunLog); // newest kept
        Assert.DoesNotContain("line-0\r", vm.Training.TrainingRunLog); // oldest dropped
    }
}
