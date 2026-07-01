using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class DatasetCardResult
{
    [JsonPropertyName("output_path")]
    public string? OutputPath { get; init; }

    [JsonPropertyName("markdown")]
    public string Markdown { get; init; } = string.Empty;

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}
