using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class BackendModelListReport
{
    [JsonPropertyName("provider_name")]
    public string ProviderName { get; init; } = string.Empty;

    [JsonPropertyName("base_url")]
    public string BaseUrl { get; init; } = string.Empty;

    [JsonPropertyName("reachable")]
    public bool Reachable { get; init; }

    [JsonPropertyName("models")]
    public IReadOnlyList<string> Models { get; init; } = [];

    [JsonPropertyName("error")]
    public string? Error { get; init; }
}
