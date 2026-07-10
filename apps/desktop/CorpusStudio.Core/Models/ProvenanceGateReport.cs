using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Deserialized result of the engine <c>provenance-gate</c> command: a per-row
/// dataset-provenance verdict. Each row's declared teacher (<c>meta.teacher</c>) is bucketed into
/// trainable / quarantined / unknown, and the overall status BLOCKs when any restricted-teacher
/// rows are present. Counts are row tallies, never a folded quality score — and the verdict is a
/// licensing judgment over the rows' DECLARED teachers, not a proof of origin.</summary>
public sealed class ProvenanceGateReport
{
    [JsonPropertyName("role")]
    public string Role { get; init; } = string.Empty;

    [JsonPropertyName("teacher_field")]
    public string TeacherField { get; init; } = string.Empty;

    [JsonPropertyName("target")]
    public string Target { get; init; } = string.Empty;

    [JsonPropertyName("total_rows")]
    public int TotalRows { get; init; }

    [JsonPropertyName("trainable_rows")]
    public int TrainableRows { get; init; }

    [JsonPropertyName("quarantined_rows")]
    public int QuarantinedRows { get; init; }

    [JsonPropertyName("unknown_rows")]
    public int UnknownRows { get; init; }

    [JsonPropertyName("strict")]
    public bool Strict { get; init; }

    [JsonPropertyName("overall_status")]
    public string OverallStatus { get; init; } = string.Empty;

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = string.Empty;

    [JsonPropertyName("buckets")]
    public IReadOnlyList<TeacherProvenanceBucket> Buckets { get; init; } = [];
}

/// <summary>One distinct teacher's roll-up: its resolved provider, provenance status, row count,
/// and licensing reason. Mirrors the engine's <c>TeacherProvenanceBucket</c>.</summary>
public sealed class TeacherProvenanceBucket
{
    [JsonPropertyName("teacher")]
    public string Teacher { get; init; } = string.Empty;

    [JsonPropertyName("provider_id")]
    public string ProviderId { get; init; } = string.Empty;

    /// <summary>pass | quarantined | unknown (the per-teacher provenance verdict).</summary>
    [JsonPropertyName("status")]
    public string Status { get; init; } = string.Empty;

    [JsonPropertyName("row_count")]
    public int RowCount { get; init; }

    [JsonPropertyName("note")]
    public string Note { get; init; } = string.Empty;
}
