using System;
using System.IO;
using System.Text.Json;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>Outcome of reading a manifest. A missing or malformed manifest is a
/// non-crashing <see cref="Manifest"/>=null result carrying a human-readable
/// <see cref="Error"/>, never an exception.</summary>
public sealed class WorkspaceManifestReadResult
{
    public WorkspaceProjectManifest? Manifest { get; init; }
    public string? Error { get; init; }
    public bool Ok => Manifest is not null;
}

/// <summary>Reads and writes the `.corpus/project.json` workspace manifest. Reading is
/// tolerant (malformed/absent → error result, not a crash) and forward-compatible (a
/// newer <c>format_version</c> still opens). Writing is atomic (temp + move) and creates
/// the `.corpus` directory. The manifest identifies a workspace; it does not own dataset
/// content.</summary>
public sealed class WorkspaceManifestService
{
    private static readonly JsonSerializerOptions WriteOptions = new()
    {
        WriteIndented = true,
    };

    private static readonly JsonSerializerOptions ReadOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true,
    };

    public string MetadataDirectory(string workspaceRoot) =>
        Path.Combine(workspaceRoot, WorkspaceProjectManifest.MetadataDirectoryName);

    public string ManifestPath(string workspaceRoot) =>
        Path.Combine(MetadataDirectory(workspaceRoot), WorkspaceProjectManifest.ManifestFileName);

    /// <summary>True when a manifest file is present at the workspace root.</summary>
    public bool HasManifest(string workspaceRoot)
    {
        try
        {
            return File.Exists(ManifestPath(workspaceRoot));
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>Read the manifest at the workspace root. Returns an error result (never
    /// throws) when the file is missing, unreadable, not JSON, or not a JSON object. A
    /// parsed-but-unrecognized manifest (wrong <c>format</c> marker) is returned so the
    /// caller can decide — check <see cref="WorkspaceProjectManifest.IsRecognized"/>.</summary>
    public WorkspaceManifestReadResult Read(string workspaceRoot)
    {
        if (string.IsNullOrWhiteSpace(workspaceRoot))
        {
            return new WorkspaceManifestReadResult { Error = "Workspace root was empty." };
        }

        string path;
        try
        {
            path = ManifestPath(workspaceRoot);
        }
        catch (ArgumentException ex)
        {
            return new WorkspaceManifestReadResult { Error = $"Invalid workspace path: {ex.Message}" };
        }

        if (!File.Exists(path))
        {
            return new WorkspaceManifestReadResult { Error = "No .corpus/project.json manifest found." };
        }

        string text;
        try
        {
            text = File.ReadAllText(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new WorkspaceManifestReadResult { Error = $"Could not read manifest: {ex.Message}" };
        }

        try
        {
            var manifest = JsonSerializer.Deserialize<WorkspaceProjectManifest>(text, ReadOptions);
            if (manifest is null)
            {
                return new WorkspaceManifestReadResult { Error = "Manifest was empty or null." };
            }

            return new WorkspaceManifestReadResult { Manifest = manifest };
        }
        catch (JsonException ex)
        {
            return new WorkspaceManifestReadResult { Error = $"Manifest is not valid JSON: {ex.Message}" };
        }
    }

    /// <summary>Write the manifest atomically (temp file + move) under the workspace root,
    /// creating `.corpus` if needed. Returns an error string on failure (never throws for
    /// expected I/O problems); null on success.</summary>
    public string? Write(string workspaceRoot, WorkspaceProjectManifest manifest)
    {
        if (string.IsNullOrWhiteSpace(workspaceRoot))
        {
            return "Workspace root was empty.";
        }

        if (manifest is null)
        {
            return "Manifest was null.";
        }

        string directory;
        string destination;
        try
        {
            directory = MetadataDirectory(workspaceRoot);
            destination = ManifestPath(workspaceRoot);
        }
        catch (ArgumentException ex)
        {
            return $"Invalid workspace path: {ex.Message}";
        }

        var temp = destination + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            Directory.CreateDirectory(directory);
            var json = JsonSerializer.Serialize(manifest, WriteOptions);
            File.WriteAllText(temp, json);
            // Atomic replace on the same volume; falls back to a plain move when absent.
            File.Move(temp, destination, overwrite: true);
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            TryDeleteTemp(temp);
            return $"Could not write manifest: {ex.Message}";
        }
    }

    private static void TryDeleteTemp(string temp)
    {
        try
        {
            if (File.Exists(temp))
            {
                File.Delete(temp);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Best-effort cleanup; a leftover temp is harmless.
        }
    }
}
