using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Wrapper for the engine `provider-policy` list output.</summary>
public sealed class ProviderPolicyListResult
{
    [JsonPropertyName("providers")]
    public Dictionary<string, ProviderPolicyItem> Providers { get; init; } = new();
}

public sealed class ProviderPolicyItem
{
    [JsonPropertyName("provider_id")]
    public string ProviderId { get; init; } = string.Empty;

    [JsonPropertyName("display_name")]
    public string DisplayName { get; init; } = string.Empty;

    [JsonPropertyName("provider_kind")]
    public string ProviderKind { get; init; } = string.Empty;

    [JsonPropertyName("generation_allowed")]
    public bool GenerationAllowed { get; init; }

    [JsonPropertyName("evaluation_allowed")]
    public bool EvaluationAllowed { get; init; }

    [JsonPropertyName("user_approved_generation")]
    public bool UserApprovedGeneration { get; init; }
}
