using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class AiAssistReviewQueueItem
{
    [JsonPropertyName("review_id")]
    public string ReviewId { get; init; } = Guid.NewGuid().ToString("N");

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;

    [JsonPropertyName("decided_at")]
    public DateTime? DecidedAt { get; set; }

    [JsonPropertyName("review_state")]
    public string ReviewState { get; set; } = "review_required";

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("action")]
    public string Action { get; init; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("prompt_template_id")]
    public string PromptTemplateId { get; init; } = string.Empty;

    [JsonPropertyName("source_draft")]
    public string SourceDraft { get; init; } = string.Empty;

    [JsonPropertyName("model_output")]
    public string ModelOutput { get; init; } = string.Empty;

    [JsonPropertyName("suggested_jsonl")]
    public string SuggestedJsonl { get; init; } = string.Empty;

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];

    [JsonPropertyName("validation_errors")]
    public List<string> ValidationErrors { get; init; } = [];

    [JsonIgnore]
    public string DisplayName => $"{CreatedAt:yyyy-MM-dd HH:mm} | {Action} | {Model} | {ReviewState}";

    [JsonIgnore]
    public string DetailText
    {
        get
        {
            var lines = new List<string>
            {
                $"Review: {ReviewId}",
                $"State: {ReviewState}",
                $"Action: {Action}",
                $"Model: {Model}",
                "",
                "Model output:",
                ModelOutput,
            };

            if (!string.IsNullOrWhiteSpace(SuggestedJsonl))
            {
                lines.Add("");
                lines.Add("Suggested JSONL:");
                lines.Add(SuggestedJsonl.TrimEnd());
            }

            if (Warnings.Count > 0)
            {
                lines.Add("");
                lines.Add("Warnings:");
                lines.AddRange(Warnings.Select(warning => $"- {warning}"));
            }

            if (ValidationErrors.Count > 0)
            {
                lines.Add("");
                lines.Add("Suggested JSONL validation errors:");
                lines.AddRange(ValidationErrors.Select(error => $"- {error}"));
            }

            lines.Add("");
            lines.Add("Source draft:");
            lines.Add(SourceDraft.TrimEnd());

            return string.Join(Environment.NewLine, lines);
        }
    }

    public static AiAssistReviewQueueItem FromRunResult(
        string sourceDraft,
        AiAssistRunResult result
    )
    {
        return new AiAssistReviewQueueItem
        {
            ReviewState = result.ReviewState,
            SchemaId = result.SchemaId,
            Action = result.Action,
            Model = result.Model,
            PromptTemplateId = result.PromptTemplateId,
            SourceDraft = sourceDraft,
            ModelOutput = result.ModelOutput,
            SuggestedJsonl = result.SuggestedJsonl,
            Warnings = result.Warnings,
            ValidationErrors = result.ValidationErrors,
        };
    }
}
