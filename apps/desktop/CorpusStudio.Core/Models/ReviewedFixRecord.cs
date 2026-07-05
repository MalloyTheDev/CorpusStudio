using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>
/// Tracks a reviewed fix for a failed evaluation example: which row was edited,
/// which report flagged it, and whether a later re-test resolved the failure.
/// Each edit of the same example is stored as a new version so the amendment
/// history stays inspectable.
/// </summary>
public sealed class ReviewedFixRecord
{
    public const string StatusEdited = "edited";
    public const string StatusResolved = "resolved";
    public const string StatusStillFailing = "still-failing";

    [JsonPropertyName("fix_id")]
    public string FixId { get; init; } = Guid.NewGuid().ToString("N");

    [JsonPropertyName("example_id")]
    public string ExampleId { get; init; } = string.Empty;

    [JsonPropertyName("row_number")]
    public int RowNumber { get; init; }

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("version")]
    public int Version { get; init; } = 1;

    [JsonPropertyName("status")]
    public string Status { get; set; } = StatusEdited;

    [JsonPropertyName("original_score")]
    public double OriginalScore { get; init; }

    [JsonPropertyName("latest_score")]
    public double? LatestScore { get; set; }

    [JsonPropertyName("failure_reason")]
    public string FailureReason { get; init; } = string.Empty;

    [JsonPropertyName("source_report")]
    public string SourceReport { get; init; } = string.Empty;

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;

    [JsonPropertyName("updated_at")]
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    [JsonIgnore]
    public string StatusLabel => Status switch
    {
        StatusResolved => "resolved",
        StatusStillFailing => "still failing",
        _ => "edited (awaiting re-test)",
    };

    [JsonIgnore]
    public string DisplayName =>
        $"{CreatedAt:yyyy-MM-dd HH:mm} | {ExampleId} v{Version} | {StatusLabel}"
        + (LatestScore is null ? string.Empty : $" | score {LatestScore:0.##}");
}
