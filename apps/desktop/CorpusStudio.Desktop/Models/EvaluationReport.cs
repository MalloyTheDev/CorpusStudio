using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class EvaluationReport
{
    [JsonPropertyName("dataset")]
    public string Dataset { get; init; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("examples_tested")]
    public int ExamplesTested { get; init; }

    [JsonPropertyName("average_score")]
    public double AverageScore { get; init; }

    [JsonPropertyName("failed_examples")]
    public int FailedExamples { get; init; }

    [JsonPropertyName("weak_tags")]
    public List<string> WeakTags { get; init; } = [];

    [JsonPropertyName("tag_summary")]
    public List<EvaluationTagSummary> TagSummary { get; init; } = [];

    [JsonPropertyName("failure_reason_summary")]
    public List<EvaluationFailureReasonSummary> FailureReasonSummary { get; init; } = [];

    [JsonPropertyName("score_band_summary")]
    public List<EvaluationScoreBandSummary> ScoreBandSummary { get; init; } = [];

    [JsonPropertyName("manually_scored_examples")]
    public int ManuallyScoredExamples { get; init; }

    [JsonPropertyName("average_manual_score")]
    public double? AverageManualScore { get; init; }

    [JsonPropertyName("run_settings")]
    public EvaluationRunSettings? RunSettings { get; init; }

    [JsonPropertyName("results")]
    public List<EvaluationExampleResult> Results { get; init; } = [];
}

public sealed class EvaluationTagSummary
{
    [JsonPropertyName("tag")]
    public string Tag { get; init; } = string.Empty;

    [JsonPropertyName("examples")]
    public int Examples { get; init; }

    [JsonPropertyName("failed_examples")]
    public int FailedExamples { get; init; }

    [JsonPropertyName("average_score")]
    public double AverageScore { get; init; }
}

public sealed class EvaluationFailureReasonSummary
{
    [JsonPropertyName("reason")]
    public string Reason { get; init; } = string.Empty;

    [JsonPropertyName("failed_examples")]
    public int FailedExamples { get; init; }
}

public sealed class EvaluationScoreBandSummary
{
    [JsonPropertyName("band")]
    public string Band { get; init; } = string.Empty;

    [JsonPropertyName("examples")]
    public int Examples { get; init; }

    [JsonPropertyName("failed_examples")]
    public int FailedExamples { get; init; }

    [JsonPropertyName("average_score")]
    public double AverageScore { get; init; }
}

public sealed class EvaluationRunSettings
{
    [JsonPropertyName("dataset_path")]
    public string? DatasetPath { get; init; }

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("backend")]
    public string Backend { get; init; } = string.Empty;

    [JsonPropertyName("base_url")]
    public string? BaseUrl { get; init; }

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("limit")]
    public int? Limit { get; init; }

    [JsonPropertyName("score_threshold")]
    public double ScoreThreshold { get; init; } = 70.0;

    [JsonPropertyName("timeout_seconds")]
    public int TimeoutSeconds { get; init; } = 120;
}
