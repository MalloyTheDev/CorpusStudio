using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Result of a dataset-version row-store garbage collection (engine
/// <c>dataset-version-gc</c>). Safe by construction: only rows no version manifest references are
/// pruned, and an unreadable manifest aborts rather than risk deleting referenced rows.</summary>
public sealed class RowStoreGcResult
{
    /// <summary>Unique row-ids referenced by all version manifests (the live set that is kept).</summary>
    [JsonPropertyName("referenced_row_ids")]
    public int ReferencedRowIds { get; set; }

    /// <summary>Row-store lines with a valid row-id that were scanned.</summary>
    [JsonPropertyName("scanned_rows")]
    public int ScannedRows { get; set; }

    /// <summary>Referenced rows kept.</summary>
    [JsonPropertyName("kept_rows")]
    public int KeptRows { get; set; }

    /// <summary>Unreferenced rows pruned (0 for a dry run).</summary>
    [JsonPropertyName("pruned_rows")]
    public int PrunedRows { get; set; }

    /// <summary>True when this was a report-only dry run (nothing rewritten).</summary>
    [JsonPropertyName("dry_run")]
    public bool DryRun { get; set; }
}
