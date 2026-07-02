using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
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
        Assert.False(vm.CanLaunchTraining);

        vm.ApplyTrainingConfigExportResult(ConfigWithLaunch());

        Assert.True(vm.CanLaunchTraining);
        Assert.Equal(5, vm.TrainingLaunchArgv.Count);
        Assert.EndsWith("exports\\x", vm.TrainingLaunchWorkingDirectory.Replace('/', '\\'));
    }

    [Fact]
    public void RunLifecycle_TracksStatusAndLog()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingConfigExportResult(ConfigWithLaunch());

        vm.BeginTrainingRun();
        Assert.True(vm.IsTrainingRunning);
        Assert.False(vm.CanLaunchTraining); // cannot launch while running

        vm.AppendTrainingRunLog("epoch 1");
        vm.AppendTrainingRunLog("epoch 2");
        Assert.Contains("epoch 1", vm.TrainingRunLog);
        Assert.Contains("epoch 2", vm.TrainingRunLog);

        vm.CompleteTrainingRun(0);
        Assert.False(vm.IsTrainingRunning);
        Assert.Contains("Completed", vm.TrainingRunStatus);
        Assert.True(vm.CanLaunchTraining); // launchable again
    }

    [Fact]
    public void CompleteTrainingRun_NonZero_MarksFailed()
    {
        var vm = new MainWindowViewModel();
        vm.BeginTrainingRun();
        vm.CompleteTrainingRun(1);
        Assert.Contains("Failed", vm.TrainingRunStatus);
    }

    [Fact]
    public void Cancel_SetsCancelledStatus()
    {
        var vm = new MainWindowViewModel();
        vm.BeginTrainingRun();
        vm.SetTrainingRunCancelled();
        Assert.False(vm.IsTrainingRunning);
        Assert.Equal("Cancelled", vm.TrainingRunStatus);
    }

    [Fact]
    public void AppendTrainingRunLog_CapsAtMaxLines()
    {
        var vm = new MainWindowViewModel();
        vm.BeginTrainingRun();
        var total = MainWindowViewModel.TrainingLogMaxLines + 500;
        for (var index = 0; index < total; index++)
        {
            vm.AppendTrainingRunLog($"line-{index}");
        }

        var lineCount = vm.TrainingRunLog.Split('\n').Length;
        Assert.True(lineCount <= MainWindowViewModel.TrainingLogMaxLines);
        Assert.Contains($"line-{total - 1}", vm.TrainingRunLog); // newest kept
        Assert.DoesNotContain("line-0\r", vm.TrainingRunLog); // oldest dropped
    }
}
