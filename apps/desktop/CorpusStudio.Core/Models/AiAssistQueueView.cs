using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class AiAssistQueueView
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("filter")]
    public string Filter { get; init; } = "All";

    [JsonPropertyName("search")]
    public string Search { get; init; } = string.Empty;

    [JsonPropertyName("sort")]
    public string Sort { get; init; } = "Newest";

    [JsonIgnore]
    public string DisplayName => Name;
}
