using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Result of the engine `preference-export` command.</summary>
public sealed class PreferenceExportResult
{
    [JsonPropertyName("format")]
    public string Format { get; init; } = string.Empty;

    [JsonPropertyName("input_rows")]
    public int InputRows { get; init; }

    [JsonPropertyName("output_rows")]
    public int OutputRows { get; init; }

    [JsonPropertyName("output_path")]
    public string OutputPath { get; init; } = string.Empty;
}
