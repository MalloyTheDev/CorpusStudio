using System;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>The `.corpus/project.json` manifest that identifies a Corpus Studio
/// workspace (v1.2.2 Workspace System). The manifest is the primary way to open a
/// workspace; it points at the authoritative dataset files (e.g. <see cref="ExamplesFile"/>)
/// but never replaces them — the engine/desktop persistence under the workspace root
/// stays the source of truth for dataset content.</summary>
public sealed class WorkspaceProjectManifest
{
    /// <summary>Marker value distinguishing a Corpus Studio manifest from arbitrary JSON.</summary>
    public const string ExpectedFormat = "corpus_studio_project";

    /// <summary>Current manifest schema version. Readers tolerate a higher value
    /// (forward-compatible) rather than refusing to open a newer workspace.</summary>
    public const int CurrentFormatVersion = 1;

    /// <summary>Relative directory that holds workspace metadata.</summary>
    public const string MetadataDirectoryName = ".corpus";

    /// <summary>Manifest file name inside <see cref="MetadataDirectoryName"/>.</summary>
    public const string ManifestFileName = "project.json";

    [JsonPropertyName("format")]
    public string Format { get; set; } = ExpectedFormat;

    [JsonPropertyName("format_version")]
    public int FormatVersion { get; set; } = CurrentFormatVersion;

    [JsonPropertyName("project_id")]
    public string ProjectId { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; set; } = string.Empty;

    [JsonPropertyName("template_id")]
    public string? TemplateId { get; set; }

    [JsonPropertyName("created_at")]
    public string? CreatedAt { get; set; }

    [JsonPropertyName("last_opened_at")]
    public string? LastOpenedAt { get; set; }

    /// <summary>Relative path (from the workspace root) to the dataset rows file.
    /// Defaults to <c>examples.jsonl</c>; the desktop remains its single writer.</summary>
    [JsonPropertyName("examples_file")]
    public string ExamplesFile { get; set; } = "examples.jsonl";

    /// <summary>Relative path (from the workspace root) to the asset directory.</summary>
    [JsonPropertyName("asset_root")]
    public string AssetRoot { get; set; } = "assets";

    [JsonPropertyName("notes")]
    public string? Notes { get; set; }

    /// <summary>True when the marker format matches — i.e. this is recognizably a
    /// Corpus Studio manifest and not some unrelated <c>project.json</c>.</summary>
    [JsonIgnore]
    public bool IsRecognized =>
        string.Equals(Format, ExpectedFormat, StringComparison.Ordinal);

    /// <summary>True when the manifest was written by a newer format version than this
    /// build knows. It is still openable (forward-compatible), but callers may warn.</summary>
    [JsonIgnore]
    public bool IsFutureVersion => FormatVersion > CurrentFormatVersion;
}
