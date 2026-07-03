using System;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>One entry in the user-local Recent Workspaces registry (v1.2.2 Workspace
/// System). Stored outside the repo under the user's local app-data. A missing path is
/// kept (with <see cref="MissingPath"/> set) rather than silently dropped, so the user
/// can still see and un-pin/remove it.</summary>
public sealed class RecentWorkspaceRecord
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    /// <summary>Absolute path to the workspace root folder.</summary>
    [JsonPropertyName("path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("schema_id")]
    public string? SchemaId { get; set; }

    [JsonPropertyName("last_opened_at")]
    public string? LastOpenedAt { get; set; }

    [JsonPropertyName("is_pinned")]
    public bool IsPinned { get; set; }

    /// <summary>Live-computed: the workspace folder no longer exists on disk. Not
    /// persisted — refreshed on load so a stale registry never crashes startup and the
    /// entry surfaces a clear "missing" badge instead.</summary>
    [JsonIgnore]
    public bool MissingPath { get; set; }

    [JsonIgnore]
    public string DisplayName => string.IsNullOrWhiteSpace(Name)
        ? (string.IsNullOrWhiteSpace(Path) ? "(unnamed)" : System.IO.Path.GetFileName(Path.TrimEnd('/', '\\')))
        : Name;
}
