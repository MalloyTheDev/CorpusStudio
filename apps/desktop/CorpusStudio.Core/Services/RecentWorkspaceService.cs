using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>User-local Recent Workspaces registry (v1.2.2 Workspace System). Stored
/// outside the repo under local app-data (the first user-local persistence in the app).
/// A missing or corrupt registry recovers to an empty list without crashing startup;
/// missing workspace folders are kept and flagged (<see cref="RecentWorkspaceRecord.MissingPath"/>),
/// not silently dropped. List mutations are pure/static so they are trivially testable.</summary>
public sealed class RecentWorkspaceService
{
    /// <summary>Upper bound on stored entries. Pinned entries are always kept; unpinned
    /// entries fill the remaining slots most-recent-first.</summary>
    public const int MaxEntries = 50;

    private const string FileName = "recent_workspaces.json";

    private static readonly JsonSerializerOptions WriteOptions = new() { WriteIndented = true };
    private static readonly JsonSerializerOptions ReadOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        AllowTrailingCommas = true,
    };

    private readonly string _storageDirectory;
    private readonly Func<string, bool> _directoryExists;

    /// <param name="storageDirectory">Registry directory. Defaults to
    /// <c>%LOCALAPPDATA%/CorpusStudio</c>. Injectable so tests use a temp dir.</param>
    /// <param name="directoryExists">Workspace-existence probe (for missing-path
    /// detection). Injectable for tests; defaults to <see cref="Directory.Exists(string)"/>.</param>
    public RecentWorkspaceService(string? storageDirectory = null, Func<string, bool>? directoryExists = null)
    {
        _storageDirectory = string.IsNullOrWhiteSpace(storageDirectory)
            ? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "CorpusStudio")
            : storageDirectory;
        _directoryExists = directoryExists ?? Directory.Exists;
    }

    public string RegistryPath => Path.Combine(_storageDirectory, FileName);

    /// <summary>Load the registry, tolerating a missing/corrupt file (→ empty list) and
    /// refreshing each entry's live <see cref="RecentWorkspaceRecord.MissingPath"/>.</summary>
    public List<RecentWorkspaceRecord> Load()
    {
        var records = ReadRaw();
        foreach (var record in records)
        {
            record.MissingPath = string.IsNullOrWhiteSpace(record.Path) || !_directoryExists(record.Path);
        }

        return records;
    }

    private List<RecentWorkspaceRecord> ReadRaw()
    {
        string path = RegistryPath;
        if (!File.Exists(path))
        {
            return new List<RecentWorkspaceRecord>();
        }

        try
        {
            var text = File.ReadAllText(path);
            var records = JsonSerializer.Deserialize<List<RecentWorkspaceRecord>>(text, ReadOptions);
            // Drop null/blank-path entries defensively; recover to empty on any parse error.
            return records?.Where(r => r is not null && !string.IsNullOrWhiteSpace(r.Path)).ToList()
                   ?? new List<RecentWorkspaceRecord>();
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            // Corrupt/unreadable registry must never crash startup — recover empty.
            return new List<RecentWorkspaceRecord>();
        }
    }

    /// <summary>Persist the registry atomically (capped via <see cref="ApplyCap"/>).
    /// Returns an error string on failure (never throws for expected I/O), null on success.</summary>
    public string? Save(IEnumerable<RecentWorkspaceRecord> records)
    {
        var capped = ApplyCap(records, MaxEntries);
        var temp = RegistryPath + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            Directory.CreateDirectory(_storageDirectory);
            var json = JsonSerializer.Serialize(capped, WriteOptions);
            File.WriteAllText(temp, json);
            File.Move(temp, RegistryPath, overwrite: true);
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            try
            {
                if (File.Exists(temp))
                {
                    File.Delete(temp);
                }
            }
            catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException)
            {
                // Best-effort; a leftover temp is harmless.
            }

            return $"Could not save recent workspaces: {ex.Message}";
        }
    }

    // ---- Pure list operations (testable without I/O) -----------------------------

    /// <summary>Insert or update <paramref name="record"/> by normalized path (case-
    /// insensitive on Windows), moving it to the front (most-recent). Pin state carries
    /// over from an existing entry unless the incoming record pins it.</summary>
    public static List<RecentWorkspaceRecord> AddOrUpdate(
        IEnumerable<RecentWorkspaceRecord> existing,
        RecentWorkspaceRecord record)
    {
        var list = existing?.ToList() ?? new List<RecentWorkspaceRecord>();
        if (record is null || string.IsNullOrWhiteSpace(record.Path))
        {
            return list;
        }

        var prior = list.FirstOrDefault(r => SamePath(r.Path, record.Path));
        if (prior is not null)
        {
            list.Remove(prior);
            record.IsPinned = record.IsPinned || prior.IsPinned;
        }

        list.Insert(0, record);
        return list;
    }

    /// <summary>Set the pin state of the entry matching <paramref name="path"/>.</summary>
    public static List<RecentWorkspaceRecord> SetPinned(
        IEnumerable<RecentWorkspaceRecord> existing, string path, bool pinned)
    {
        var list = existing?.ToList() ?? new List<RecentWorkspaceRecord>();
        var match = list.FirstOrDefault(r => SamePath(r.Path, path));
        if (match is not null)
        {
            match.IsPinned = pinned;
        }

        return list;
    }

    /// <summary>Remove the entry matching <paramref name="path"/>.</summary>
    public static List<RecentWorkspaceRecord> Remove(
        IEnumerable<RecentWorkspaceRecord> existing, string path)
    {
        var list = existing?.ToList() ?? new List<RecentWorkspaceRecord>();
        list.RemoveAll(r => SamePath(r.Path, path));
        return list;
    }

    /// <summary>Cap to <paramref name="max"/> entries: all pinned entries are kept (in
    /// order); unpinned entries fill the remaining slots most-recent-first.</summary>
    public static List<RecentWorkspaceRecord> ApplyCap(IEnumerable<RecentWorkspaceRecord> existing, int max)
    {
        var list = existing?.ToList() ?? new List<RecentWorkspaceRecord>();
        if (max <= 0 || list.Count <= max)
        {
            return list;
        }

        var pinned = list.Where(r => r.IsPinned).ToList();
        var unpinned = list.Where(r => !r.IsPinned).ToList();
        var slotsForUnpinned = Math.Max(0, max - pinned.Count);

        // Preserve original relative order while keeping all pins.
        var kept = new HashSet<RecentWorkspaceRecord>(pinned);
        foreach (var record in unpinned.Take(slotsForUnpinned))
        {
            kept.Add(record);
        }

        return list.Where(kept.Contains).ToList();
    }

    private static bool SamePath(string? left, string? right)
    {
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
        {
            return false;
        }

        var comparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;
        return string.Equals(
            Path.TrimEndingDirectorySeparator(left),
            Path.TrimEndingDirectorySeparator(right),
            comparison);
    }
}
