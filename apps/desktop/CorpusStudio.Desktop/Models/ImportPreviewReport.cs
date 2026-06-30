using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class ImportPreviewReport
{
    [JsonPropertyName("valid")]
    public bool Valid { get; init; }

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("path")]
    public string Path { get; init; } = string.Empty;

    [JsonPropertyName("total_rows")]
    public int TotalRows { get; init; }

    [JsonPropertyName("accepted_rows")]
    public int AcceptedRows { get; init; }

    [JsonPropertyName("rejected_rows")]
    public int RejectedRows { get; init; }

    [JsonPropertyName("failed_rows")]
    public IReadOnlyList<ImportFailure> FailedRows { get; init; } = [];
}

public sealed class ImportFailure
{
    [JsonPropertyName("row_number")]
    public int RowNumber { get; init; }

    [JsonPropertyName("raw_preview")]
    public string RawPreview { get; init; } = string.Empty;

    [JsonPropertyName("errors")]
    public IReadOnlyList<ValidationIssue> Errors { get; init; } = [];
}
