using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>One indexed project row returned by the engine `project-list` command.</summary>
public sealed class ProjectIndexEntryDto
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("schema_id")]
    public string SchemaId { get; init; } = string.Empty;

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; init; } = string.Empty;

    [JsonPropertyName("updated_at")]
    public string UpdatedAt { get; init; } = string.Empty;

    [JsonPropertyName("example_count")]
    public int ExampleCount { get; init; }

    [JsonPropertyName("path")]
    public string Path { get; init; } = string.Empty;
}

/// <summary>Result of the engine `project-list` command.</summary>
public sealed class ProjectIndexListReport
{
    [JsonPropertyName("projects_root")]
    public string ProjectsRoot { get; init; } = string.Empty;

    [JsonPropertyName("index_path")]
    public string IndexPath { get; init; } = string.Empty;

    [JsonPropertyName("count")]
    public int Count { get; init; }

    [JsonPropertyName("projects")]
    public List<ProjectIndexEntryDto> Projects { get; init; } = [];
}

/// <summary>Result of the engine `project-index-rebuild` command.</summary>
public sealed class ProjectIndexRebuildResult
{
    [JsonPropertyName("projects_root")]
    public string ProjectsRoot { get; init; } = string.Empty;

    [JsonPropertyName("index_path")]
    public string IndexPath { get; init; } = string.Empty;

    [JsonPropertyName("indexed")]
    public int Indexed { get; init; }
}
