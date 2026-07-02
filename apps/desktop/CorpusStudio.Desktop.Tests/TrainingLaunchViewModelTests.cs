using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class TrainingLaunchViewModelTests
{
    [Fact]
    public void ApplyTrainingConfig_SurfacesLaunchCommand_AndStoresIt()
    {
        var vm = new MainWindowViewModel();
        var result = new TrainingConfigExportResult
        {
            Target = "axolotl_yaml",
            OutputPath = "out/config.yaml",
            Launch = new TrainingLaunchPlan
            {
                Target = "axolotl_yaml",
                Command = "accelerate launch -m axolotl.cli.train \"out/config.yaml\"",
                ResumeCommand = "accelerate launch -m axolotl.cli.train \"out/config.yaml\" --resume_from_checkpoint=\"<checkpoint-dir>\"",
                ResumeSupported = true,
                Dependencies = ["axolotl", "accelerate"],
            },
        };

        vm.ApplyTrainingConfigExportResult(result);

        Assert.Equal(result.Launch.Command, vm.TrainingLaunchCommand);
        Assert.Contains("Launch command", vm.TrainingSummary);
        Assert.Contains("accelerate launch", vm.TrainingSummary);
        Assert.Contains("requires: axolotl, accelerate", vm.TrainingSummary);
    }

    [Fact]
    public void ApplyTrainingConfig_NoLaunch_ClearsCommand()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingConfigExportResult(
            new TrainingConfigExportResult { Target = "x", OutputPath = "y" }
        );
        Assert.Equal(string.Empty, vm.TrainingLaunchCommand);
    }
}
