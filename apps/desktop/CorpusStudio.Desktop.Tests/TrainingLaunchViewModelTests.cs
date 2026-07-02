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
    public void ApplyTrainingConfig_ShowsVramEstimateAndLoraSuggestion()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingConfigExportResult(new TrainingConfigExportResult
        {
            Target = "axolotl_yaml",
            OutputPath = "out/config.yaml",
            VramEstimate = new VramEstimate
            {
                ParameterCountBillions = 7.0,
                WeightsGbFp16 = 14.0,
                TotalGbFp16 = 17.2,
                TotalGbInt8 = 10.2,
                TotalGbInt4 = 6.7,
            },
            LoraRecommendation = new LoraRecommendation
            {
                RecommendedR = 16,
                RecommendedAlpha = 32,
                Warnings = ["lora_r=128 is unusually high for this model size"],
            },
        });

        Assert.Contains("VRAM (rough, 7B params)", vm.TrainingSummary);
        Assert.Contains("17.2 GB fp16", vm.TrainingSummary);
        Assert.Contains("LoRA suggestion: r=16, alpha=32", vm.TrainingSummary);
        Assert.Contains("unusually high", vm.TrainingSummary);
    }

    [Fact]
    public void ApplyTrainingConfig_UnknownModelSize_IsHonest()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyTrainingConfigExportResult(new TrainingConfigExportResult
        {
            Target = "axolotl_yaml",
            OutputPath = "out/config.yaml",
            VramEstimate = new VramEstimate { ParameterCountBillions = null },
        });

        Assert.Contains("no estimate", vm.TrainingSummary);
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
