using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Per-project gate thresholds (issue #198), mirroring the engine's <c>GateThresholds</c>. Read
/// via <c>gate-thresholds</c> and written (validated) via <c>gate-thresholds-set</c>. Plain get/set so the
/// Settings form two-way-binds the fields directly; defaults match the engine so a fresh project shows the
/// effective values.</summary>
public sealed class GateThresholds
{
    [JsonPropertyName("max_exact_duplicates")]
    public int MaxExactDuplicates { get; set; }

    [JsonPropertyName("block_exact_duplicates")]
    public bool BlockExactDuplicates { get; set; } = true;

    [JsonPropertyName("max_normalized_duplicates")]
    public int MaxNormalizedDuplicates { get; set; }

    [JsonPropertyName("block_normalized_duplicates")]
    public bool BlockNormalizedDuplicates { get; set; }

    [JsonPropertyName("max_low_information")]
    public int MaxLowInformation { get; set; }

    [JsonPropertyName("block_low_information")]
    public bool BlockLowInformation { get; set; }

    [JsonPropertyName("warn_synthetic_pattern_issues")]
    public int WarnSyntheticPatternIssues { get; set; } = 1;

    [JsonPropertyName("block_on_high_severity_pii")]
    public bool BlockOnHighSeverityPii { get; set; } = true;

    [JsonPropertyName("warn_on_medium_severity_pii")]
    public bool WarnOnMediumSeverityPii { get; set; } = true;

    [JsonPropertyName("min_eval_average_score")]
    public double MinEvalAverageScore { get; set; } = 70.0;

    [JsonPropertyName("min_eval_pass_rate")]
    public double MinEvalPassRate { get; set; } = 0.5;

    [JsonPropertyName("max_regression_score_drop")]
    public double MaxRegressionScoreDrop { get; set; } = 2.0;

    [JsonPropertyName("min_chat_turns")]
    public int MinChatTurns { get; set; } = 2;

    [JsonPropertyName("max_chat_turns")]
    public int MaxChatTurns { get; set; }
}
