using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class AiAssistRunResult
{
    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("action")]
    public string Action { get; init; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("review_state")]
    public string ReviewState { get; init; } = "review_required";

    [JsonPropertyName("review_required")]
    public bool ReviewRequired { get; init; } = true;

    [JsonPropertyName("prompt_template_id")]
    public string PromptTemplateId { get; init; } = string.Empty;

    [JsonPropertyName("model_output")]
    public string ModelOutput { get; init; } = string.Empty;

    [JsonPropertyName("suggested_jsonl")]
    public string SuggestedJsonl { get; init; } = string.Empty;

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];

    [JsonPropertyName("validation_errors")]
    public List<string> ValidationErrors { get; init; } = [];
}
