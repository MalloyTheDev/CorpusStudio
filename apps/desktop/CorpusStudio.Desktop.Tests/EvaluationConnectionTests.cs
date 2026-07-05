using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Evaluation model-backend connection sub-view-model (backend-cluster slice 3, PR 3a). The
/// config defaults + model-list summary in isolation, and the Lab-settings flow through the shell,
/// which now applies to / reads from this child (ApplyLabSettings / BuildCurrentLabSettings).</summary>
public sealed class EvaluationConnectionTests
{
    [Fact]
    public void Defaults_AreOllamaQwenLocalhost()
    {
        var vm = new EvaluationConnectionViewModel();
        Assert.Equal("ollama", vm.EvaluationBackend);
        Assert.Equal("qwen2.5-coder:7b", vm.EvaluationModel);
        Assert.Equal("http://localhost:11434", vm.EvaluationBaseUrl);
        Assert.Equal("120", vm.EvaluationTimeoutSeconds);
        Assert.Empty(vm.EvaluationAvailableModels);
        Assert.Contains("Refresh models", vm.EvaluationModelListSummary);
    }

    [Fact]
    public void SetModelListSummary_UpdatesTheSummary()
    {
        var vm = new EvaluationConnectionViewModel();
        vm.SetModelListSummary("Refreshing models from ollama...");
        Assert.Equal("Refreshing models from ollama...", vm.EvaluationModelListSummary);
    }

    [Fact]
    public void ApplyLabSettings_PopulatesTheConnectionChild()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyLabSettings(new LabBackendSettings
        {
            Evaluation = new ModelBackendSettings
            {
                Backend = "lm-studio",
                Model = "phi-3",
                BaseUrl = "http://localhost:1234/v1",
                TimeoutSeconds = 90,
            },
        });

        Assert.Equal("lm-studio", vm.EvaluationConnection.EvaluationBackend);
        Assert.Equal("phi-3", vm.EvaluationConnection.EvaluationModel);
        Assert.Equal("http://localhost:1234/v1", vm.EvaluationConnection.EvaluationBaseUrl);
        Assert.Equal("90", vm.EvaluationConnection.EvaluationTimeoutSeconds);
    }

    [Fact]
    public void BuildCurrentLabSettings_ReadsTheConnectionChild()
    {
        var vm = new MainWindowViewModel();
        vm.EvaluationConnection.EvaluationBackend = "openai-compatible";
        vm.EvaluationConnection.EvaluationModel = "gpt-oss";
        vm.EvaluationConnection.EvaluationBaseUrl = "http://host:1234/v1";
        vm.EvaluationConnection.EvaluationTimeoutSeconds = "45";

        var built = vm.BuildCurrentLabSettings();

        Assert.Equal("openai-compatible", built.Evaluation.Backend);
        Assert.Equal("gpt-oss", built.Evaluation.Model);
        Assert.Equal("http://host:1234/v1", built.Evaluation.BaseUrl);
        Assert.Equal(45, built.Evaluation.TimeoutSeconds);
    }
}
