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
}
