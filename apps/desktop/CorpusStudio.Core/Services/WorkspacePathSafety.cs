using System;
using System.IO;
using System.Linq;

namespace CorpusStudio.Desktop.Services;

/// <summary>Pure path-safety guarantees for the Workspace System. Every workspace file
/// operation must resolve through here so nothing can read/write outside the active
/// workspace root (no traversal, no absolute escapes). No I/O — string/path logic only.</summary>
public static class WorkspacePathSafety
{
    private static StringComparison PathComparison =>
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    /// <summary>Normalize a root to a full path with any trailing separator trimmed.
    /// Throws <see cref="ArgumentException"/> for an empty root.</summary>
    public static string NormalizeRoot(string root)
    {
        if (string.IsNullOrWhiteSpace(root))
        {
            throw new ArgumentException("Workspace root must not be empty.", nameof(root));
        }

        return Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
    }

    /// <summary>True when <paramref name="candidateFullPath"/> is the root itself or a
    /// descendant of it. Uses a trailing-separator guard so "C:\a" is not treated as a
    /// parent of "C:\ab".</summary>
    public static bool IsWithinRoot(string root, string candidateFullPath)
    {
        if (string.IsNullOrWhiteSpace(root) || string.IsNullOrWhiteSpace(candidateFullPath))
        {
            return false;
        }

        string normRoot;
        string normCandidate;
        try
        {
            normRoot = NormalizeRoot(root);
            normCandidate = Path.TrimEndingDirectorySeparator(Path.GetFullPath(candidateFullPath));
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }

        if (string.Equals(normRoot, normCandidate, PathComparison))
        {
            return true;
        }

        var rootWithSeparator = normRoot + Path.DirectorySeparatorChar;
        return normCandidate.StartsWith(rootWithSeparator, PathComparison);
    }

    /// <summary>Resolve a workspace-relative path against the root, allowing ONLY
    /// descent from the root. Rejects absolute/rooted inputs and any traversal that
    /// escapes the root. Returns false (never throws) on any unsafe or malformed input.</summary>
    public static bool TryResolveWithinRoot(string root, string relativePath, out string resolvedFullPath)
    {
        resolvedFullPath = string.Empty;
        if (string.IsNullOrWhiteSpace(root) || string.IsNullOrWhiteSpace(relativePath))
        {
            return false;
        }

        // An absolute/rooted child (e.g. "C:\x", "/etc", "\\server\share") is never a
        // safe relative descent — reject before combining.
        if (Path.IsPathRooted(relativePath))
        {
            return false;
        }

        string normRoot;
        string candidate;
        try
        {
            normRoot = NormalizeRoot(root);
            candidate = Path.TrimEndingDirectorySeparator(Path.GetFullPath(Path.Combine(normRoot, relativePath)));
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }

        if (!IsWithinRoot(normRoot, candidate))
        {
            return false;
        }

        resolvedFullPath = candidate;
        return true;
    }

    /// <summary>Sanitize a single path segment (project/file/folder name) into something
    /// safe to create: invalid filename characters and separators become '_', surrounding
    /// whitespace/dots are trimmed. Returns empty when nothing usable remains (e.g. "" or
    /// "..") so callers can refuse rather than create a dangerous name.</summary>
    public static string SanitizeSegmentName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return string.Empty;
        }

        var invalid = Path.GetInvalidFileNameChars();
        var cleaned = new string(name
            .Select(c => invalid.Contains(c) || c == '/' || c == '\\' ? '_' : c)
            .ToArray())
            .Trim()
            .Trim('.')
            .Trim();

        // Reject reserved/degenerate results outright.
        return cleaned is "" or "." or ".." ? string.Empty : cleaned;
    }
}
