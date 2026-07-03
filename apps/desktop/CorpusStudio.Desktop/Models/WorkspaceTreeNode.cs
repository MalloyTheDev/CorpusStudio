using System.Collections.Generic;

namespace CorpusStudio.Desktop.Models;

/// <summary>One node in the Universal Workspace Explorer tree (v1.2.3 Workspace System,
/// slice 4). The file <em>kind</em> (<see cref="FileKind"/>) selects the viewer; the tree
/// is always built root-bounded and deterministically ordered (folders first, then files,
/// each alphabetical and case-insensitive).</summary>
public sealed class WorkspaceTreeNode
{
    public string Name { get; init; } = string.Empty;

    /// <summary>Absolute path on disk.</summary>
    public string FullPath { get; init; } = string.Empty;

    /// <summary>Path relative to the workspace root (forward-slash separated). Empty for the root.</summary>
    public string RelativePath { get; init; } = string.Empty;

    public bool IsDirectory { get; init; }

    /// <summary>Lowercased extension including the dot (e.g. <c>.jsonl</c>); empty for
    /// directories or extensionless files.</summary>
    public string Extension { get; init; } = string.Empty;

    public WorkspaceFileKind FileKind { get; init; }

    public List<WorkspaceTreeNode> Children { get; } = new();

    /// <summary>Expansion state, owned by the explorer view-model (not persisted here).</summary>
    public bool IsExpanded { get; set; }

    /// <summary>True when this node is under a generated-artifact directory (reports,
    /// training_runs, model_artifacts, dataset_versions, evaluation_reports, arena_reports).
    /// Such files open read-only by default.</summary>
    public bool IsGeneratedArtifact { get; init; }

    /// <summary>True for the dataset's core files (examples.jsonl and the manifest), so the
    /// single-writer rule can be surfaced distinctly in the UI.</summary>
    public bool IsDatasetCoreFile { get; init; }
}
