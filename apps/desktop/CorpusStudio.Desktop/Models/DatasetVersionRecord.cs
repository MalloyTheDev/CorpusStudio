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

    /// <summary>Whether this version stored its row bodies (=> diffable/restorable).
    /// The engine records a fingerprint-only version with <c>rows_stored=false</c> when
    /// the row store could not be written; such a version is NOT a safe undo point.</summary>
    [JsonPropertyName("rows_stored")]
    public bool RowsStored { get; set; }

    [JsonPropertyName("stored_row_count")]
    public int StoredRowCount { get; set; }

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

/// <summary>Result of an engine <c>dataset-version-restore</c> (mirrors the engine
/// RestoreResult). The engine reconstructs to a temp file, verified against the
/// recorded fingerprint; the desktop performs the atomic in-place swap.</summary>
public sealed class RestoreResult
{
    [JsonPropertyName("version_id")]
    public string VersionId { get; set; } = string.Empty;

    [JsonPropertyName("rows_written")]
    public int RowsWritten { get; set; }

    [JsonPropertyName("verified")]
    public bool Verified { get; set; }

    [JsonPropertyName("verify_skipped")]
    public bool VerifySkipped { get; set; }

    [JsonPropertyName("output_path")]
    public string OutputPath { get; set; } = string.Empty;
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

    /// <summary>Normalized to exactly matches | drifted | unreadable — the single
    /// source of truth for both the badge and the summary counts. Anything else
    /// (null, empty, or an unrecognized engine value) is treated as unreadable, so
    /// we never claim "matches" without the engine confirming it and a row can never
    /// show a badge that no summary bucket counts.</summary>
    public string Integrity => Record.CurrentIntegrity switch
    {
        "matches" => "matches",
        "drifted" => "drifted",
        _ => "unreadable",
    };

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
