using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Result of the engine `training-compat` pre-check command.</summary>
public sealed class TrainingCompatibilityResult
{
    [JsonPropertyName("schema")]
    public string Schema { get; init; } = string.Empty;

    [JsonPropertyName("target")]
    public string Target { get; init; } = string.Empty;

    [JsonPropertyName("format")]
    public string Format { get; init; } = string.Empty;

    [JsonPropertyName("compatible")]
    public bool Compatible { get; init; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}
