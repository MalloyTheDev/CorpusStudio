using System.Text.Json;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Issue #198: the desktop gate-threshold model must round-trip the engine's snake_case
/// gate_thresholds.json and default to the same values, and the Settings VM must load/report it.</summary>
public sealed class GateThresholdsTests
{
    [Fact]
    public void Deserializes_EngineSnakeCaseJson()
    {
        const string json = """
        {"max_exact_duplicates":5,"block_exact_duplicates":false,"max_normalized_duplicates":3,
         "block_normalized_duplicates":true,"max_low_information":2,"block_low_information":true,
         "warn_synthetic_pattern_issues":4,"block_on_high_severity_pii":false,"warn_on_medium_severity_pii":false,
         "min_eval_average_score":80.0,"min_eval_pass_rate":0.75,"max_regression_score_drop":3.5,
         "min_chat_turns":3,"max_chat_turns":10}
        """;

        var t = JsonSerializer.Deserialize<GateThresholds>(json)!;

        Assert.Equal(5, t.MaxExactDuplicates);
        Assert.False(t.BlockExactDuplicates);
        Assert.True(t.BlockNormalizedDuplicates);
        Assert.Equal(0.75, t.MinEvalPassRate);
        Assert.Equal(3, t.MinChatTurns);
        Assert.Equal(10, t.MaxChatTurns);
    }

    [Fact]
    public void Serializes_ToEngineSnakeCaseKeys()
    {
        var json = JsonSerializer.Serialize(new GateThresholds { MaxChatTurns = 12 });

        Assert.Contains("\"min_eval_pass_rate\"", json);
        Assert.Contains("\"max_chat_turns\":12", json);
        Assert.Contains("\"block_on_high_severity_pii\":true", json); // default
    }

    [Fact]
    public void Defaults_MatchTheEngine()
    {
        var t = new GateThresholds();

        Assert.True(t.BlockExactDuplicates);
        Assert.False(t.BlockNormalizedDuplicates);
        Assert.Equal(1, t.WarnSyntheticPatternIssues);
        Assert.Equal(70.0, t.MinEvalAverageScore);
        Assert.Equal(0.5, t.MinEvalPassRate);
        Assert.Equal(2.0, t.MaxRegressionScoreDrop);
        Assert.Equal(2, t.MinChatTurns);
        Assert.Equal(0, t.MaxChatTurns);
    }

    [Fact]
    public void ApplyGateThresholds_SetsInstanceAndSummary()
    {
        var vm = new SettingsViewModel();
        var t = new GateThresholds { MaxExactDuplicates = 7 };

        vm.ApplyGateThresholds(t);
        Assert.Same(t, vm.GateThresholds);
        Assert.Contains("effective gate thresholds", vm.GateThresholdsSummary);

        vm.SetGateThresholdsSaved();
        Assert.Contains("Saved", vm.GateThresholdsSummary);

        vm.SetGateThresholdsError("nope");
        Assert.Contains("could not be saved", vm.GateThresholdsSummary);
    }
}
