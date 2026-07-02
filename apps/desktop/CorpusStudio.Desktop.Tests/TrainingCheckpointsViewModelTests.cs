using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class TrainingCheckpointsViewModelTests
{
    private static TrainingCheckpointsResult WithCheckpoints(bool resumeSupported = true) => new()
    {
        OutputDirectory = "out",
        Checkpoints = ["checkpoint-10", "checkpoint-40"],
        LatestCheckpoint = "checkpoint-40",
        ResumeCommand = "accelerate launch -m axolotl.cli.train \"c.yaml\" --resume_from_checkpoint=\"out/checkpoint-40\"",
        ResumeArgv = resumeSupported
            ? ["accelerate", "launch", "-m", "axolotl.cli.train", "c.yaml", "--resume_from_checkpoint", "out/checkpoint-40"]
            : ["python", "c.py"],
        ResumeSupported = resumeSupported,
    };

    [Fact]
    public void ApplyCheckpoints_EnablesResume_ForSupportedTarget()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingCheckpoints(WithCheckpoints());

        Assert.True(vm.CanResumeTraining);
        Assert.Contains("2", vm.TrainingCheckpointsSummary);
        Assert.Contains("checkpoint-40", vm.TrainingCheckpointsSummary);
        Assert.Equal("--resume_from_checkpoint", vm.TrainingResumeArgv[^2]);
    }

    [Fact]
    public void ApplyCheckpoints_ConfigDrivenTarget_DisablesResume_WithNote()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingCheckpoints(WithCheckpoints(resumeSupported: false));

        Assert.False(vm.CanResumeTraining);
        Assert.Contains("config-driven", vm.TrainingCheckpointsSummary);
    }

    [Fact]
    public void ApplyCheckpoints_Empty_DisablesResume()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingCheckpoints(new TrainingCheckpointsResult());

        Assert.False(vm.CanResumeTraining);
        Assert.Contains("No checkpoints", vm.TrainingCheckpointsSummary);
    }

    [Fact]
    public void ResumeDisabled_WhileRunActive()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingCheckpoints(WithCheckpoints());
        Assert.True(vm.CanResumeTraining);

        vm.BeginTrainingRun();
        Assert.False(vm.CanResumeTraining);

        vm.CompleteTrainingRun(0);
        Assert.True(vm.CanResumeTraining);
    }

    [Fact]
    public void NewConfigExport_ClearsResumeState_AndStoresOutputDir()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingCheckpoints(WithCheckpoints());
        Assert.True(vm.CanResumeTraining);

        vm.ApplyTrainingConfigExportResult(new TrainingConfigExportResult
        {
            Target = "axolotl_yaml",
            OutputPath = "C:/exports/p/training/config.yaml",
            TrainingOutputDirectory = "C:/exports/p/training/output",
        });

        Assert.False(vm.CanResumeTraining); // stale resume argv cleared
        Assert.Equal("C:/exports/p/training/output", vm.TrainingOutputDirectory);
        Assert.Equal("C:/exports/p/training/config.yaml", vm.TrainingConfigPath);
    }
}
