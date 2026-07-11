using System;
using System.Globalization;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>A durable model artifact record (v0.9). Stores only non-derivable
/// fields — base model and eval scores are resolved live through <see cref="RunId"/>.
/// Mutable because status/updated_at/fingerprint change across its lifecycle.</summary>
public sealed class ModelArtifactRecord
{
    [JsonPropertyName("artifact_id")]
    public string ArtifactId { get; set; } = string.Empty;

    [JsonPropertyName("run_id")]
    public string RunId { get; set; } = string.Empty;

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = string.Empty;

    [JsonPropertyName("updated_at")]
    public string UpdatedAt { get; set; } = string.Empty;

    [JsonPropertyName("path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("kind")]
    public string Kind { get; set; } = "adapter";

    [JsonPropertyName("status")]
    public string Status { get; set; } = "candidate";

    [JsonPropertyName("fingerprint")]
    public string? Fingerprint { get; set; }

    [JsonPropertyName("notes")]
    public string Notes { get; set; } = string.Empty;
}

/// <summary>A view row: a record plus its computed integrity and the base model
/// resolved through the source run (never stored on the artifact).</summary>
public sealed class ArtifactDisplayItem
{
    public ArtifactDisplayItem(ModelArtifactRecord record, string integrity, string resolvedBaseModel)
    {
        Record = record;
        Integrity = integrity;
        ResolvedBaseModel = resolvedBaseModel;
    }

    public ModelArtifactRecord Record { get; }

    public string Integrity { get; }

    public string ResolvedBaseModel { get; }

    public string DisplayName
    {
        get
        {
            var badge = Integrity switch
            {
                "missing" => "⛔ missing",
                "modified" => "⚠ modified",
                _ => "✅ ok",
            };
            var model = string.IsNullOrWhiteSpace(ResolvedBaseModel) ? "(base unknown)" : ResolvedBaseModel;
            return $"[{Record.Status}] {Record.Kind} — {badge} — {model} — {Record.ArtifactId}";
        }
    }

    // --- Nocturne card display helpers (production re-skin) -------------------
    // Derived purely from the record + the computed integrity signal — no new state,
    // no fabricated fields. The integrity states below are the REAL provenance verdict
    // (fingerprint match / mismatch / missing path), surfaced truthfully on the card.

    /// <summary>Card headline: the resolved base model this artifact adapts (never stored on the
    /// artifact; resolved live through the run). Falls back to an explicit "(base unknown)".</summary>
    public string PrimaryName =>
        string.IsNullOrWhiteSpace(ResolvedBaseModel) ? "(base unknown)" : ResolvedBaseModel;

    /// <summary>Integrity is intact (fingerprint matches, or none was stored).</summary>
    public bool IsOk => Integrity == "ok";

    /// <summary>Weights changed since the artifact was registered/evaluated (fingerprint mismatch).</summary>
    public bool IsModified => Integrity == "modified";

    /// <summary>The recorded weight path no longer exists.</summary>
    public bool IsMissing => Integrity == "missing";

    /// <summary>Any non-intact integrity state — the promote gate fails closed on these.</summary>
    public bool IsFlagged => Integrity != "ok";

    /// <summary>The integrity chip caption, phrased as the design's "integrity: present / modified".</summary>
    public string IntegrityLabel => Integrity switch
    {
        "ok" => "integrity: present",
        "modified" => "integrity: modified",
        "missing" => "integrity: missing",
        _ => $"integrity: {Integrity}",
    };

    /// <summary>Honest promote-block reason derived from the real integrity signal: a modified
    /// fingerprint or a missing path both fail the promote gate closed, so the card can state why
    /// before the user even runs Keep. Empty when integrity is intact.</summary>
    public string IntegrityBlockMessage => Integrity switch
    {
        "modified" => "Promote gate blocks — weights changed since eval",
        "missing" => "Promote gate blocks — weights missing at the recorded path",
        _ => string.Empty,
    };

    /// <summary>A compact, honest created timestamp. Parses the stored ISO-8601 UTC string; if it is
    /// not a parseable timestamp (e.g. legacy/test data) the raw value is shown unchanged.</summary>
    public string CreatedDisplay =>
        DateTimeOffset.TryParse(
            Record.CreatedAt,
            CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
            out var created)
            ? created.ToString("yyyy-MM-dd HH:mm 'UTC'", CultureInfo.InvariantCulture)
            : Record.CreatedAt;
}
