using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>Result of a create-file/folder action inside a workspace.</summary>
public sealed class ExplorerCreateResult
{
    public bool Ok => Error is null;
    public string? Error { get; init; }
    public string FullPath { get; init; } = string.Empty;
    public string RelativePath { get; init; } = string.Empty;
    public WorkspaceFileKind FileKind { get; init; }
}

/// <summary>Builds the Universal Workspace Explorer tree and performs guarded create
/// operations (v1.2.3 Workspace System, slice 4). The tree is deterministic (folders
/// first, then files, each alphabetical / case-insensitive), root-bounded, skips VCS /
/// build junk, and guards against symlink loops (reparse points are not followed). No
/// deletes — this slice never removes files.</summary>
public sealed class WorkspaceExplorerService
{
    public int MaxDepth { get; init; } = 32;

    /// <summary>Build the tree rooted at the workspace. Never throws for expected I/O:
    /// unreadable directories are simply skipped.</summary>
    public WorkspaceTreeNode BuildTree(string workspaceRoot)
    {
        var root = WorkspacePathSafety.NormalizeRoot(workspaceRoot);
        var comparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        var visited = new HashSet<string>(comparer);

        var node = new WorkspaceTreeNode
        {
            Name = Path.GetFileName(root.TrimEnd(Path.DirectorySeparatorChar)),
            FullPath = root,
            RelativePath = string.Empty,
            IsDirectory = true,
            FileKind = WorkspaceFileKind.Folder,
            IsExpanded = true,
        };
        Populate(node, root, root, 0, visited);
        return node;
    }

    private void Populate(WorkspaceTreeNode parent, string root, string dirFull, int depth, HashSet<string> visited)
    {
        if (depth >= MaxDepth) return;

        string real;
        try { real = Path.GetFullPath(dirFull); }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException) { return; }
        if (!visited.Add(real)) return; // symlink / junction loop guard

        IEnumerable<string> entries;
        try { entries = Directory.EnumerateFileSystemEntries(dirFull); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return; }

        var children = new List<WorkspaceTreeNode>();
        foreach (var entry in entries)
        {
            bool isDir;
            try
            {
                var attr = File.GetAttributes(entry);
                if (attr.HasFlag(FileAttributes.ReparsePoint)) continue; // do not follow symlinks/junctions
                isDir = attr.HasFlag(FileAttributes.Directory);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { continue; }

            var name = Path.GetFileName(entry.TrimEnd(Path.DirectorySeparatorChar));
            if (isDir && WorkspaceLayout.IsIgnoredDirectory(name)) continue;
            if (!WorkspacePathSafety.IsWithinRoot(root, entry)) continue;

            var rel = WorkspaceLayout.Normalize(Path.GetRelativePath(root, entry));
            var child = new WorkspaceTreeNode
            {
                Name = name,
                FullPath = entry,
                RelativePath = rel,
                IsDirectory = isDir,
                Extension = isDir ? string.Empty : Path.GetExtension(entry).ToLowerInvariant(),
                FileKind = WorkspaceFileKinds.Classify(entry, isDir),
                IsGeneratedArtifact = WorkspaceLayout.IsGeneratedArtifact(rel),
                IsDatasetCoreFile = WorkspaceLayout.IsDatasetCoreFile(rel),
            };
            if (isDir) Populate(child, root, entry, depth + 1, visited);
            children.Add(child);
        }

        foreach (var c in Sort(children)) parent.Children.Add(c);
    }

    /// <summary>Deterministic order: directories first, then files; each group sorted
    /// case-insensitively, then by ordinal as a stable tie-breaker.</summary>
    public static IEnumerable<WorkspaceTreeNode> Sort(IEnumerable<WorkspaceTreeNode> nodes) =>
        nodes.OrderByDescending(n => n.IsDirectory)
             .ThenBy(n => n.Name, StringComparer.OrdinalIgnoreCase)
             .ThenBy(n => n.Name, StringComparer.Ordinal);

    public ExplorerCreateResult CreateFolder(string workspaceRoot, string relativePath)
        => Create(workspaceRoot, relativePath, isDirectory: true);

    public ExplorerCreateResult CreateFile(string workspaceRoot, string relativePath)
        => Create(workspaceRoot, relativePath, isDirectory: false);

    private ExplorerCreateResult Create(string workspaceRoot, string relativePath, bool isDirectory)
    {
        if (!WorkspacePathSafety.TryResolveWithinRoot(workspaceRoot, relativePath, out var full))
            return new ExplorerCreateResult { Error = "Path is empty, absolute, or escapes the workspace root." };

        try
        {
            if (isDirectory)
            {
                if (Directory.Exists(full)) return new ExplorerCreateResult { Error = "A folder with that name already exists." };
                Directory.CreateDirectory(full);
            }
            else
            {
                if (File.Exists(full)) return new ExplorerCreateResult { Error = "A file with that name already exists." };
                Directory.CreateDirectory(Path.GetDirectoryName(full)!);
                using (File.Create(full)) { }
            }

            var root = WorkspacePathSafety.NormalizeRoot(workspaceRoot);
            var rel = WorkspaceLayout.Normalize(Path.GetRelativePath(root, full));
            return new ExplorerCreateResult
            {
                FullPath = full,
                RelativePath = rel,
                FileKind = WorkspaceFileKinds.Classify(full, isDirectory),
            };
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new ExplorerCreateResult { Error = ex.Message };
        }
    }

    /// <summary>Rename a file/folder in-place within the workspace. <paramref name="newName"/> must be a
    /// single safe filename (no separators / traversal / invalid chars); the target must not already exist
    /// and the source must exist and not be the workspace root.</summary>
    public ExplorerCreateResult RenamePath(string workspaceRoot, string fullPath, string newName)
    {
        var root = WorkspacePathSafety.NormalizeRoot(workspaceRoot);
        var source = Path.GetFullPath(fullPath);

        if (string.Equals(source, root, StringComparison.OrdinalIgnoreCase))
            return new ExplorerCreateResult { Error = "The workspace root cannot be renamed." };
        if (!IsWithinRoot(root, source))
            return new ExplorerCreateResult { Error = "Path escapes the workspace root." };

        var isDirectory = Directory.Exists(source);
        if (!isDirectory && !File.Exists(source))
            return new ExplorerCreateResult { Error = "That file or folder no longer exists." };

        var nameError = ValidateSingleName(newName);
        if (nameError is not null)
            return new ExplorerCreateResult { Error = nameError };

        var target = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(source)!, newName));
        if (string.Equals(source, target, StringComparison.OrdinalIgnoreCase))
            return new ExplorerCreateResult { Error = "The new name is the same as the current name." };
        if (Directory.Exists(target) || File.Exists(target))
            return new ExplorerCreateResult { Error = "A file or folder with that name already exists." };

        try
        {
            if (isDirectory)
                Directory.Move(source, target);
            else
                File.Move(source, target);

            return new ExplorerCreateResult
            {
                FullPath = target,
                RelativePath = WorkspaceLayout.Normalize(Path.GetRelativePath(root, target)),
                FileKind = WorkspaceFileKinds.Classify(target, isDirectory),
            };
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new ExplorerCreateResult { Error = ex.Message };
        }
    }

    /// <summary>Permanently delete a file/folder within the workspace. The source must exist, stay within
    /// the workspace, and not be the workspace root. Folders are removed recursively.</summary>
    public ExplorerCreateResult DeletePath(string workspaceRoot, string fullPath)
    {
        var root = WorkspacePathSafety.NormalizeRoot(workspaceRoot);
        var source = Path.GetFullPath(fullPath);

        if (string.Equals(source, root, StringComparison.OrdinalIgnoreCase))
            return new ExplorerCreateResult { Error = "The workspace root cannot be deleted." };
        if (!IsWithinRoot(root, source))
            return new ExplorerCreateResult { Error = "Path escapes the workspace root." };

        var isDirectory = Directory.Exists(source);
        if (!isDirectory && !File.Exists(source))
            return new ExplorerCreateResult { Error = "That file or folder no longer exists." };

        try
        {
            if (isDirectory)
                Directory.Delete(source, recursive: true);
            else
                File.Delete(source);

            return new ExplorerCreateResult
            {
                FullPath = source,
                RelativePath = WorkspaceLayout.Normalize(Path.GetRelativePath(root, source)),
            };
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new ExplorerCreateResult { Error = ex.Message };
        }
    }

    private static bool IsWithinRoot(string normalizedRoot, string fullPath)
    {
        var rooted = normalizedRoot.EndsWith(Path.DirectorySeparatorChar) ? normalizedRoot
            : normalizedRoot + Path.DirectorySeparatorChar;
        return fullPath.StartsWith(rooted, StringComparison.OrdinalIgnoreCase);
    }

    private static string? ValidateSingleName(string newName)
    {
        var name = (newName ?? string.Empty).Trim();
        if (name.Length == 0)
            return "Enter a name.";
        if (name is "." or "..")
            return "That name is not allowed.";
        if (name.IndexOf(Path.DirectorySeparatorChar) >= 0 || name.IndexOf(Path.AltDirectorySeparatorChar) >= 0)
            return "The name cannot contain a path separator.";
        if (name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            return "The name contains invalid characters.";
        return null;
    }
}
