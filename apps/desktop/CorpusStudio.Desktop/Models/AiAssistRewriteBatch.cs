using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class AiAssistRewriteBatch
{
    [JsonPropertyName("batch_id")]
    public string BatchId { get; init; } = Guid.NewGuid().ToString("N");

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("action")]
    public string Action { get; init; } = "rewrite-output";

    [JsonPropertyName("row_numbers")]
    public List<int> RowNumbers { get; init; } = [];

    [JsonPropertyName("issue_count")]
    public int IssueCount { get; init; }

    [JsonPropertyName("issue_summary")]
    public string IssueSummary { get; init; } = string.Empty;

    [JsonPropertyName("source_draft")]
    public string SourceDraft { get; init; } = string.Empty;

    [JsonPropertyName("instruction")]
    public string Instruction { get; init; } = string.Empty;

    [JsonIgnore]
    public string DisplayName =>
        $"{CreatedAt:yyyy-MM-dd HH:mm} | {RowNumbers.Count} row(s) | {IssueCount} issue(s)";
}
