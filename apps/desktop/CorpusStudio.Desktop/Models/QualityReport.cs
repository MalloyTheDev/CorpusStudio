using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class QualityReport
{
    [JsonPropertyName("example_count")]
    public int ExampleCount { get; init; }

    [JsonPropertyName("empty_row_count")]
    public int EmptyRowCount { get; init; }

    [JsonPropertyName("duplicate_exact_count")]
    public int DuplicateExactCount { get; init; }
}
