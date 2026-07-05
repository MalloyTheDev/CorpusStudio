using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class BackendHealthReport
{
    [JsonPropertyName("provider_name")]
    public string ProviderName { get; init; } = string.Empty;

    [JsonPropertyName("base_url")]
    public string BaseUrl { get; init; } = string.Empty;

    [JsonPropertyName("model_name")]
    public string ModelName { get; init; } = string.Empty;

    [JsonPropertyName("reachable")]
    public bool Reachable { get; init; }

    [JsonPropertyName("model_available")]
    public bool ModelAvailable { get; init; }

    [JsonPropertyName("available_models")]
    public List<string> AvailableModels { get; init; } = [];

    [JsonPropertyName("error")]
    public string? Error { get; init; }
}
