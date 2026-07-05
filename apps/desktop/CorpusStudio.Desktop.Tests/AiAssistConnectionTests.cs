using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>AI-Assist model-backend connection sub-view-model (backend-cluster slice 2, PR 2/3). The
/// config defaults + model-list summary in isolation, and the Lab-settings flow through the shell,
/// which now applies to / reads from this child (ApplyLabSettings / BuildCurrentLabSettings).</summary>
public sealed class AiAssistConnectionTests
{
    [Fact]
    public void Defaults_AreOllamaQwenLocalhost()
    {
        var vm = new AiAssistConnectionViewModel();
        Assert.Equal("ollama", vm.AiAssistBackend);
        Assert.Equal("qwen2.5-coder:7b", vm.AiAssistModel);
        Assert.Equal("http://localhost:11434", vm.AiAssistBaseUrl);
        Assert.Equal("120", vm.AiAssistTimeoutSeconds);
        Assert.Empty(vm.AiAssistAvailableModels);
        Assert.Contains("Refresh models", vm.AiAssistModelListSummary);
    }

    [Fact]
    public void SetModelListSummary_UpdatesTheSummary()
    {
        var vm = new AiAssistConnectionViewModel();
        vm.SetModelListSummary("Refreshing models from ollama...");
        Assert.Equal("Refreshing models from ollama...", vm.AiAssistModelListSummary);
    }

    [Fact]
    public void ApplyLabSettings_PopulatesTheConnectionChild()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyLabSettings(new LabBackendSettings
        {
            AiAssist = new ModelBackendSettings
            {
                Backend = "lm-studio",
                Model = "phi-3",
                BaseUrl = "http://localhost:1234/v1",
                TimeoutSeconds = 90,
            },
        });

        Assert.Equal("lm-studio", vm.AiAssistConnection.AiAssistBackend);
        Assert.Equal("phi-3", vm.AiAssistConnection.AiAssistModel);
        Assert.Equal("http://localhost:1234/v1", vm.AiAssistConnection.AiAssistBaseUrl);
        Assert.Equal("90", vm.AiAssistConnection.AiAssistTimeoutSeconds);
    }

    [Fact]
    public void BuildCurrentLabSettings_ReadsTheConnectionChild()
    {
        var vm = new MainWindowViewModel();
        vm.AiAssistConnection.AiAssistBackend = "openai-compatible";
        vm.AiAssistConnection.AiAssistModel = "gpt-oss";
        vm.AiAssistConnection.AiAssistBaseUrl = "http://host:1234/v1";
        vm.AiAssistConnection.AiAssistTimeoutSeconds = "45";

        var built = vm.BuildCurrentLabSettings();

        Assert.Equal("openai-compatible", built.AiAssist.Backend);
        Assert.Equal("gpt-oss", built.AiAssist.Model);
        Assert.Equal("http://host:1234/v1", built.AiAssist.BaseUrl);
        Assert.Equal(45, built.AiAssist.TimeoutSeconds);
    }
}
