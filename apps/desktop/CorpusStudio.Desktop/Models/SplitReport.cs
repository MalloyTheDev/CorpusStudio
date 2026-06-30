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
}
