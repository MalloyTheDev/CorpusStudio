using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class LabBackendSettings
{
    [JsonPropertyName("evaluation")]
    public ModelBackendSettings Evaluation { get; init; } = ModelBackendSettings.Default;

    [JsonPropertyName("ai_assist")]
    public ModelBackendSettings AiAssist { get; init; } = ModelBackendSettings.Default;

    public static LabBackendSettings Default => new();
}

public sealed class ModelBackendSettings
{
    [JsonPropertyName("backend")]
    public string Backend { get; init; } = "ollama";

    [JsonPropertyName("model")]
    public string Model { get; init; } = "qwen2.5-coder:7b";

    [JsonPropertyName("base_url")]
    public string BaseUrl { get; init; } = "http://localhost:11434";

    [JsonPropertyName("timeout_seconds")]
    public int TimeoutSeconds { get; init; } = 120;

    public static ModelBackendSettings Default => new();
}
