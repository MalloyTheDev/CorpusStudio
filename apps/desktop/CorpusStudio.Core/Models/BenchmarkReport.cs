using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Envelope emitted by the engine `benchmark` command.</summary>
public sealed class BenchmarkRunOutput
{
    [JsonPropertyName("benchmark")]
    public BenchmarkReport? Benchmark { get; init; }
}

public sealed class BenchmarkReport
{
    [JsonPropertyName("dataset")]
    public string Dataset { get; init; } = string.Empty;

    [JsonPropertyName("model_count")]
    public int ModelCount { get; init; }

    [JsonPropertyName("examples_tested")]
    public int ExamplesTested { get; init; }

    [JsonPropertyName("best_model")]
    public string BestModel { get; init; } = string.Empty;

    [JsonPropertyName("worst_model")]
    public string WorstModel { get; init; } = string.Empty;

    [JsonPropertyName("score_spread")]
    public double ScoreSpread { get; init; }

    [JsonPropertyName("models")]
    public IReadOnlyList<BenchmarkModelSummary> Models { get; init; } = [];

    [JsonPropertyName("commonly_failed_examples")]
    public IReadOnlyList<string> CommonlyFailedExamples { get; init; } = [];
}

public sealed class BenchmarkModelSummary
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("examples_tested")]
    public int ExamplesTested { get; init; }

    [JsonPropertyName("average_score")]
    public double AverageScore { get; init; }

    [JsonPropertyName("pass_rate")]
    public double PassRate { get; init; }

    [JsonPropertyName("failed_examples")]
    public int FailedExamples { get; init; }

    [JsonPropertyName("rank")]
    public int Rank { get; init; }

    [JsonPropertyName("score_delta_vs_best")]
    public double ScoreDeltaVsBest { get; init; }

    public string DisplayName
    {
        get
        {
            var delta = ScoreDeltaVsBest < 0
                ? $" ({ScoreDeltaVsBest:0.##} vs best)"
                : string.Empty;
            return $"#{Rank} {Model}: avg {AverageScore:0.##}, pass {PassRate:0.#}%, {FailedExamples} failed{delta}";
        }
    }
}
