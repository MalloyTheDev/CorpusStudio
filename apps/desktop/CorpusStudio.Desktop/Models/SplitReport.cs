using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class SplitReport
{
    [JsonPropertyName("train")]
    public int Train { get; init; }

    [JsonPropertyName("validation")]
    public int Validation { get; init; }

    [JsonPropertyName("test")]
    public int Test { get; init; }

    [JsonPropertyName("output_dir")]
    public string OutputDirectory { get; init; } = string.Empty;

    [JsonPropertyName("train_ratio")]
    public double TrainRatio { get; init; }

    [JsonPropertyName("validation_ratio")]
    public double ValidationRatio { get; init; }

    [JsonPropertyName("test_ratio")]
    public double TestRatio { get; init; }

    [JsonPropertyName("seed")]
    public int Seed { get; init; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}
