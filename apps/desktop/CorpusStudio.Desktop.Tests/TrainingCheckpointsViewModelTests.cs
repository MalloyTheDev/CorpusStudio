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
        vm.Training.ApplyTrainingCheckpoints(WithCheckpoints());

        Assert.True(vm.Training.CanResumeTraining);
        Assert.Contains("2", vm.Training.TrainingCheckpointsSummary);
        Assert.Contains("checkpoint-40", vm.Training.TrainingCheckpointsSummary);
        Assert.Equal("--resume_from_checkpoint", vm.Training.TrainingResumeArgv[^2]);
    }

    [Fact]
    public void ApplyCheckpoints_ConfigDrivenTarget_DisablesResume_WithNote()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingCheckpoints(WithCheckpoints(resumeSupported: false));

        Assert.False(vm.Training.CanResumeTraining);
        Assert.Contains("config-driven", vm.Training.TrainingCheckpointsSummary);
    }

    [Fact]
    public void ApplyCheckpoints_Empty_DisablesResume()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingCheckpoints(new TrainingCheckpointsResult());

        Assert.False(vm.Training.CanResumeTraining);
        Assert.Contains("No checkpoints", vm.Training.TrainingCheckpointsSummary);
    }

    [Fact]
    public void ResumeDisabled_WhileRunActive()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingCheckpoints(WithCheckpoints());
        Assert.True(vm.Training.CanResumeTraining);

        vm.Training.BeginTrainingRun();
        Assert.False(vm.Training.CanResumeTraining);

        vm.Training.CompleteTrainingRun(0);
        Assert.True(vm.Training.CanResumeTraining);
    }

    [Fact]
    public void NewConfigExport_ClearsResumeState_AndStoresOutputDir()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingCheckpoints(WithCheckpoints());
        Assert.True(vm.Training.CanResumeTraining);

        vm.Training.ApplyTrainingConfigExportResult(new TrainingConfigExportResult
        {
            Target = "axolotl_yaml",
            OutputPath = "C:/exports/p/training/config.yaml",
            TrainingOutputDirectory = "C:/exports/p/training/output",
        });

        Assert.False(vm.Training.CanResumeTraining); // stale resume argv cleared
        Assert.Equal("C:/exports/p/training/output", vm.Training.TrainingOutputDirectory);
        Assert.Equal("C:/exports/p/training/config.yaml", vm.Training.TrainingConfigPath);
    }
}
