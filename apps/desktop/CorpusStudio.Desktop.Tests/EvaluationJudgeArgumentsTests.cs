using System.Collections.Generic;
using System.Linq;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Issue #189: the desktop can now request the opt-in LLM-judge scorer. Guards that the eval-run
/// argument builder adds the judge flags only when a judge model is set, defaults the judge backend/base-url
/// to the eval run's own, and honours explicit overrides — and that the connection VM carries the field.</summary>
public sealed class EvaluationJudgeArgumentsTests
{
    private static List<string> Build(string? judgeModel, string? judgeBackend = null, string? judgeBaseUrl = null)
        => PythonEngineService.BuildEvaluationRunArguments(
            "examples.jsonl", "instruction", "ollama", "qwen2.5-coder:7b", "out.json",
            0.6, 120, "http://localhost:11434", 50, judgeModel, judgeBackend, judgeBaseUrl);

    [Fact]
    public void NoJudgeModel_KeepsKeywordOverlap_WithNoJudgeArgs()
    {
        var args = Build(null);
        Assert.DoesNotContain("--judge-model", args);
        Assert.DoesNotContain("--judge-backend", args);
        Assert.DoesNotContain("--judge-base-url", args);
        // The base eval run is unchanged.
        Assert.Contains("eval-run", args);
        Assert.Equal("qwen2.5-coder:7b", args[args.IndexOf("--model") + 1]);
    }

    [Fact]
    public void JudgeModel_AddsJudgeArgs_DefaultingBackendAndBaseUrlToTheEvalRun()
    {
        var args = Build("qwen2.5:14b");
        Assert.Equal("qwen2.5:14b", args[args.IndexOf("--judge-model") + 1]);
        Assert.Equal("ollama", args[args.IndexOf("--judge-backend") + 1]);
        Assert.Equal("http://localhost:11434", args[args.IndexOf("--judge-base-url") + 1]);
    }

    [Fact]
    public void JudgeModel_HonoursExplicitJudgeBackendAndBaseUrl()
    {
        var args = Build("gpt-4o-mini", judgeBackend: "openai-compatible", judgeBaseUrl: "http://host:1234/v1");
        Assert.Equal("openai-compatible", args[args.IndexOf("--judge-backend") + 1]);
        Assert.Equal("http://host:1234/v1", args[args.IndexOf("--judge-base-url") + 1]);
    }

    [Fact]
    public void BlankOrWhitespaceJudgeModel_IsTreatedAsNoJudge()
    {
        Assert.DoesNotContain("--judge-model", Build("   "));
    }

    [Fact]
    public void ConnectionViewModel_JudgeModel_DefaultsEmpty_AndIsSettable()
    {
        var vm = new EvaluationConnectionViewModel();
        Assert.Equal(string.Empty, vm.EvaluationJudgeModel);
        vm.EvaluationJudgeModel = "qwen2.5:14b";
        Assert.Equal("qwen2.5:14b", vm.EvaluationJudgeModel);
    }
}
