using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Deserialized result of the engine `hf-inspect` command: a public Hugging Face
/// dataset's configs/splits, sample columns, and license (read-only, no auth).</summary>
public sealed class HfDatasetInspection
{
    [JsonPropertyName("dataset_id")]
    public string DatasetId { get; init; } = string.Empty;

    [JsonPropertyName("viewable")]
    public bool Viewable { get; init; }

    [JsonPropertyName("gated")]
    public bool Gated { get; init; }

    [JsonPropertyName("license")]
    public string? License { get; init; }

    [JsonPropertyName("license_note")]
    public string LicenseNote { get; init; } = string.Empty;

    [JsonPropertyName("configs_splits")]
    public IReadOnlyList<HfConfigSplit> ConfigsSplits { get; init; } = [];

    [JsonPropertyName("sample_columns")]
    public IReadOnlyList<string> SampleColumns { get; init; } = [];
}

public sealed class HfConfigSplit
{
    [JsonPropertyName("config")]
    public string Config { get; init; } = string.Empty;

    [JsonPropertyName("split")]
    public string Split { get; init; } = string.Empty;

    /// <summary>"config / split" for display in a picker.</summary>
    public string Display => $"{Config} / {Split}";
}

/// <summary>Deserialized result of the engine `hf-import` command: what was staged, the
/// column mapping used, and the license note (surfaced before the rows are committed).</summary>
public sealed class HfImportResult
{
    [JsonPropertyName("dataset_id")]
    public string DatasetId { get; init; } = string.Empty;

    [JsonPropertyName("config")]
    public string Config { get; init; } = string.Empty;

    [JsonPropertyName("split")]
    public string Split { get; init; } = string.Empty;

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("fetched_rows")]
    public int FetchedRows { get; init; }

    [JsonPropertyName("mapping")]
    public Dictionary<string, string> Mapping { get; init; } = new();

    [JsonPropertyName("unmapped_schema_fields")]
    public IReadOnlyList<string> UnmappedSchemaFields { get; init; } = [];

    [JsonPropertyName("unused_columns")]
    public IReadOnlyList<string> UnusedColumns { get; init; } = [];

    [JsonPropertyName("license")]
    public string? License { get; init; }

    [JsonPropertyName("license_note")]
    public string LicenseNote { get; init; } = string.Empty;

    [JsonPropertyName("out_path")]
    public string? OutPath { get; init; }
}
