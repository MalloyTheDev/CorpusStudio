using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class ValidationReport
{
    [JsonPropertyName("valid")]
    public bool Valid { get; init; }

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("checked_rows")]
    public int CheckedRows { get; init; }

    [JsonPropertyName("errors")]
    public IReadOnlyList<ValidationIssue> Errors { get; init; } = [];

    [JsonPropertyName("warnings")]
    public IReadOnlyList<ValidationIssue> Warnings { get; init; } = [];
}

public sealed class ValidationIssue
{
    [JsonPropertyName("level")]
    public string Level { get; init; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; init; } = string.Empty;

    [JsonPropertyName("row_number")]
    public int? RowNumber { get; init; }

    [JsonPropertyName("field")]
    public string? Field { get; init; }
}
