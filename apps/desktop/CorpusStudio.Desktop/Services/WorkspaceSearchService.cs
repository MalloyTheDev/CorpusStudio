using System;
using System.Collections.Generic;
using System.IO;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>Workspace content search ("find in files"): walks the same root-bounded,
/// junk-skipping, symlink-safe tree the Universal Explorer builds, and reports
/// case-insensitive (by default) substring matches per line in text-editable files.
/// Pure and deterministic over the filesystem, so it is unit-testable; the view-model
/// runs it off the UI thread. Read-only — it never opens binaries, huge files, or
/// mutates anything.</summary>
public sealed class WorkspaceSearchService
{
    private readonly WorkspaceExplorerService _explorer;

    public WorkspaceSearchService(WorkspaceExplorerService? explorer = null)
    {
        _explorer = explorer ?? new WorkspaceExplorerService();
    }

    /// <summary>Maximum matches returned; hitting it sets <see cref="WorkspaceSearchResult.Truncated"/>.</summary>
    public int MaxResults { get; init; } = 500;

    /// <summary>Files larger than this (bytes) are skipped — content search is for source/data,
    /// not multi-MB blobs.</summary>
    public long MaxFileBytes { get; init; } = 5_000_000;

    /// <summary>Matching lines longer than this are truncated in the result (the whole line is
    /// still searched).</summary>
    public int MaxLineLength { get; init; } = 240;

    public WorkspaceSearchResult Search(string? workspaceRoot, string? query, bool caseSensitive = false)
    {
        if (string.IsNullOrEmpty(query) || string.IsNullOrWhiteSpace(workspaceRoot))
        {
            return WorkspaceSearchResult.Empty;
        }

        WorkspaceTreeNode tree;
        try
        {
            tree = _explorer.BuildTree(workspaceRoot);
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException)
        {
            return WorkspaceSearchResult.Empty;
        }

        var files = new List<WorkspaceTreeNode>();
        CollectTextFiles(tree, files);

        var comparison = caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
        var matches = new List<WorkspaceSearchMatch>();
        var filesScanned = 0;
        var filesMatched = 0;
        var truncated = false;

        foreach (var file in files)
        {
            if (matches.Count >= MaxResults)
            {
                truncated = true;
                break;
            }

            long length;
            try
            {
                length = new FileInfo(file.FullPath).Length;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                continue;
            }
            if (length > MaxFileBytes)
            {
                continue;
            }

            filesScanned++;
            var fileHadMatch = false;
            var lineNumber = 0;
            try
            {
                foreach (var line in File.ReadLines(file.FullPath))
                {
                    lineNumber++;
                    if (line.IndexOf(query, comparison) < 0)
                    {
                        continue;
                    }

                    fileHadMatch = true;
                    matches.Add(new WorkspaceSearchMatch
                    {
                        RelativePath = file.RelativePath,
                        FullPath = file.FullPath,
                        LineNumber = lineNumber,
                        LineText = Trim(line),
                    });
                    if (matches.Count >= MaxResults)
                    {
                        truncated = true;
                        break;
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                continue; // unreadable file — skip, don't fail the whole search
            }

            if (fileHadMatch)
            {
                filesMatched++;
            }
        }

        return new WorkspaceSearchResult
        {
            Matches = matches,
            FilesScanned = filesScanned,
            FilesMatched = filesMatched,
            Truncated = truncated,
        };
    }

    private string Trim(string line)
    {
        var trimmed = line.Trim();
        return trimmed.Length > MaxLineLength ? trimmed[..MaxLineLength] + "…" : trimmed;
    }

    private static void CollectTextFiles(WorkspaceTreeNode node, List<WorkspaceTreeNode> into)
    {
        if (node.IsDirectory)
        {
            foreach (var child in node.Children)
            {
                CollectTextFiles(child, into);
            }
        }
        else if (WorkspaceFileKinds.IsTextEditable(node.FileKind))
        {
            into.Add(node);
        }
    }
}
