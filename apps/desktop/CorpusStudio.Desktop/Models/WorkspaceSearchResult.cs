namespace CorpusStudio.Desktop.Models;

/// <summary>One line in one workspace file that matched a content search.</summary>
public sealed class WorkspaceSearchMatch
{
    public string RelativePath { get; init; } = string.Empty;
    public string FullPath { get; init; } = string.Empty;

    /// <summary>1-based line number of the match within the file.</summary>
    public int LineNumber { get; init; }

    /// <summary>The matching line, trimmed and length-capped for display.</summary>
    public string LineText { get; init; } = string.Empty;

    // The display line split around the (first) match so the view can bold/highlight it.
    // BeforeMatch + MatchText + AfterMatch == LineText; MatchText is empty when the match
    // fell outside the truncated display window.
    public string BeforeMatch { get; init; } = string.Empty;
    public string MatchText { get; init; } = string.Empty;
    public string AfterMatch { get; init; } = string.Empty;

    /// <summary>Compact list label, e.g. <c>reports/q.json:12</c>.</summary>
    public string Location => $"{RelativePath}:{LineNumber}";
}

/// <summary>Result of a workspace content search: the matches plus scan/limit metadata.</summary>
public sealed class WorkspaceSearchResult
{
    public IReadOnlyList<WorkspaceSearchMatch> Matches { get; init; } = System.Array.Empty<WorkspaceSearchMatch>();

    /// <summary>Text files read (binary/oversize/ignored-dir files are not counted).</summary>
    public int FilesScanned { get; init; }

    /// <summary>Distinct files that contained at least one match.</summary>
    public int FilesMatched { get; init; }

    /// <summary>True when the match cap was hit and more matches likely exist.</summary>
    public bool Truncated { get; init; }

    public static WorkspaceSearchResult Empty { get; } = new();
}
