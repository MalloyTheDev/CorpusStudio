using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>A durable dataset version record (v1.0). A lineage anchor: it stores the
/// dataset's identity (row count + a content fingerprint the engine computes) and
/// links to the runs/artifacts/evals that co-existed with it. Nothing derivable is
/// stored — the engine resolves scores/integrity live. The desktop never recomputes
/// the fingerprint itself; <see cref="CurrentIntegrity"/> is annotated by the engine
/// <c>dataset-version-list</c> command.</summary>
public sealed class DatasetVersionRecord
{
    [JsonPropertyName("version_id")]
    public string VersionId { get; set; } = string.Empty;

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = string.Empty;

    [JsonPropertyName("updated_at")]
    public string UpdatedAt { get; set; } = string.Empty;

    [JsonPropertyName("label")]
    public string Label { get; set; } = string.Empty;

    [JsonPropertyName("trigger")]
    public string Trigger { get; set; } = string.Empty;

    [JsonPropertyName("row_count")]
    public int RowCount { get; set; }

    [JsonPropertyName("content_fingerprint")]
    public string? ContentFingerprint { get; set; }

    [JsonPropertyName("fingerprint_algo")]
    public string FingerprintAlgo { get; set; } = string.Empty;

    [JsonPropertyName("row_signature_kind")]
    public string RowSignatureKind { get; set; } = string.Empty;

    [JsonPropertyName("source_run_ids")]
    public List<string> SourceRunIds { get; set; } = [];

    [JsonPropertyName("artifact_ids")]
    public List<string> ArtifactIds { get; set; } = [];

    [JsonPropertyName("eval_report_path")]
    public string? EvalReportPath { get; set; }

    [JsonPropertyName("gate_report_path")]
    public string? GateReportPath { get; set; }

    [JsonPropertyName("notes")]
    public string Notes { get; set; } = string.Empty;

    /// <summary>Live integrity of the version vs the current dataset, annotated by
    /// <c>dataset-version-list</c> (matches | drifted | unreadable). Null when this
    /// record came from a bare <c>dataset-version-create</c> response.</summary>
    [JsonPropertyName("current_integrity")]
    public string? CurrentIntegrity { get; set; }
}

/// <summary>Envelope for the <c>dataset-version-list</c> JSON payload.</summary>
public sealed class DatasetVersionListResult
{
    [JsonPropertyName("versions")]
    public List<DatasetVersionRecord> Versions { get; set; } = [];
}

/// <summary>A view row: a version record plus a display string with a live
/// integrity badge and its lineage link count.</summary>
public sealed class DatasetVersionDisplayItem
{
    public DatasetVersionDisplayItem(DatasetVersionRecord record)
    {
        Record = record;
    }

    public DatasetVersionRecord Record { get; }

    /// <summary>matches | drifted | unreadable. A missing annotation is treated as
    /// unreadable (we never claim "matches" without the engine confirming it).</summary>
    public string Integrity =>
        string.IsNullOrEmpty(Record.CurrentIntegrity) ? "unreadable" : Record.CurrentIntegrity;

    public int LinkCount => Record.SourceRunIds.Count + Record.ArtifactIds.Count;

    public string DisplayName
    {
        get
        {
            var badge = Integrity switch
            {
                "matches" => "✅ matches",
                "drifted" => "⚠ drifted",
                _ => "⛔ unreadable",
            };
            var label = string.IsNullOrWhiteSpace(Record.Label) ? "(no label)" : Record.Label;
            var links = LinkCount > 0 ? $" — {LinkCount} link{(LinkCount == 1 ? "" : "s")}" : "";
            return $"{badge} — {label} — {Record.RowCount} rows{links} — {Record.VersionId}";
        }
    }
}
