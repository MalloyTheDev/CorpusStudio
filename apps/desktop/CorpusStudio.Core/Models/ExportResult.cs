using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Result of the engine `export` command.</summary>
public sealed class ExportResult
{
    [JsonPropertyName("output_path")]
    public string OutputPath { get; init; } = string.Empty;

    [JsonPropertyName("cleaned")]
    public bool Cleaned { get; init; }

    [JsonPropertyName("input_rows")]
    public int InputRows { get; init; }

    [JsonPropertyName("output_rows")]
    public int OutputRows { get; init; }

    [JsonPropertyName("removed_rows")]
    public int RemovedRows { get; init; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}
