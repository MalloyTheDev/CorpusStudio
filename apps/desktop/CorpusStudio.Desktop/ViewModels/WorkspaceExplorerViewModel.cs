using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>Backs the Universal Workspace Explorer (v1.2.4 Workspace System, slice 3b):
/// the file tree, open document tabs, and viewers over the active workspace root. Wraps
/// the slice-2 services; the desktop never mutates dataset content except through an
/// explicit document save. Kept separate from MainWindowViewModel per the design.</summary>
public sealed class WorkspaceExplorerViewModel : INotifyPropertyChanged
{
    private readonly WorkspaceExplorerService _explorer;
    private readonly WorkspaceDocumentService _documents;

    public WorkspaceExplorerViewModel(
        WorkspaceExplorerService? explorer = null,
        WorkspaceDocumentService? documents = null)
    {
        _explorer = explorer ?? new WorkspaceExplorerService();
        _documents = documents ?? new WorkspaceDocumentService();
    }

    private string? _workspaceRoot;
    private string _workspaceName = string.Empty;
    private WorkspaceTreeNode? _rootNode;
    private OpenWorkspaceDocument? _activeDocument;
    private string _explorerStatus = "Open or create a project to browse its files.";

    public string WorkspaceName
    {
        get => _workspaceName;
        private set => SetField(ref _workspaceName, value);
    }

    /// <summary>Tree root; the view binds a TreeView to <c>RootNode.Children</c>.</summary>
    public WorkspaceTreeNode? RootNode
    {
        get => _rootNode;
        private set => SetField(ref _rootNode, value);
    }

    public bool HasWorkspace => !string.IsNullOrWhiteSpace(_workspaceRoot);

    public string ExplorerStatus
    {
        get => _explorerStatus;
        private set => SetField(ref _explorerStatus, value);
    }

    public ObservableCollection<OpenWorkspaceDocument> OpenDocuments { get; } = new();

    /// <summary>Whether any open document has unsaved edits — used to warn before a project
    /// switch or app close discards them.</summary>
    public bool HasDirtyDocuments => OpenDocuments.Any(document => document.IsDirty);

    public OpenWorkspaceDocument? ActiveDocument
    {
        get => _activeDocument;
        set
        {
            var previous = _activeDocument;
            if (SetField(ref _activeDocument, value))
            {
                // Exactly one tab is active: clear the old, set the new.
                if (previous is not null)
                {
                    previous.IsActive = false;
                }

                if (value is not null)
                {
                    value.IsActive = true;
                }

                RebuildMetadataRows();
                OnChanged(nameof(IsNoDocument));
                OnChanged(nameof(IsTextDocument));
                OnChanged(nameof(IsImageDocument));
                OnChanged(nameof(IsBinaryDocument));
                OnChanged(nameof(IsExamplesFile));
                OnChanged(nameof(ActiveRelPath));
            }
        }
    }

    // ---- Viewer-state helpers (partition the four viewers) -----------------------

    public bool IsNoDocument => _activeDocument is null;

    public bool IsTextDocument =>
        _activeDocument is { } d && WorkspaceFileKinds.IsTextEditable(d.FileKind);

    public bool IsImageDocument =>
        _activeDocument is { FileKind: WorkspaceFileKind.Image };

    /// <summary>Any opened non-text, non-image document (audio/video/binary/unknown) —
    /// the "no text preview" panel.</summary>
    public bool IsBinaryDocument =>
        _activeDocument is not null && !IsTextDocument && !IsImageDocument;

    public string ActiveRelPath => _activeDocument?.RelativePath ?? string.Empty;

    /// <summary>True when the active document is the dataset's <c>examples.jsonl</c> — the
    /// caution banner reminds the user the desktop is its single writer.</summary>
    public bool IsExamplesFile =>
        _activeDocument is { } d
        && d.FileKind == WorkspaceFileKind.Jsonl
        && WorkspaceLayout.IsDatasetCoreFile(d.RelativePath);

    public ObservableCollection<WorkspaceMetadataRow> MetadataRows { get; } = new();

    // ---- Workspace root ----------------------------------------------------------

    /// <summary>Point the explorer at a workspace root and build its tree. No-ops (keeps
    /// the tree and open tabs) when the root is unchanged, so toggling Studio ↔ Files is
    /// cheap. A null/blank root clears the explorer.</summary>
    public void SetWorkspaceRoot(string? root, string name)
    {
        if (string.IsNullOrWhiteSpace(root))
        {
            Reset();
            return;
        }

        if (SamePath(_workspaceRoot, root) && _rootNode is not null)
        {
            return;
        }

        _workspaceRoot = root;
        WorkspaceName = string.IsNullOrWhiteSpace(name) ? "Workspace" : name;
        OpenDocuments.Clear();
        ActiveDocument = null;
        RefreshTree();
        OnChanged(nameof(HasWorkspace));
    }

    /// <summary>Clear all explorer state (e.g. on project switch) so a stale tree/tabs
    /// can't linger.</summary>
    public void Reset()
    {
        _workspaceRoot = null;
        WorkspaceName = string.Empty;
        OpenDocuments.Clear();
        ActiveDocument = null;
        RootNode = null;
        ExplorerStatus = "Open or create a project to browse its files.";
        OnChanged(nameof(HasWorkspace));
    }

    public void RefreshTree()
    {
        if (!HasWorkspace)
        {
            RootNode = null;
            return;
        }

        try
        {
            var tree = _explorer.BuildTree(_workspaceRoot!);
            RootNode = tree;
            var count = tree.Children.Count;
            ExplorerStatus = count == 0 ? "Empty workspace." : $"{count} item{(count == 1 ? "" : "s")} at the root.";
        }
        catch (Exception ex) when (ex is ArgumentException or System.IO.IOException or UnauthorizedAccessException)
        {
            RootNode = null;
            ExplorerStatus = "Could not read the workspace folder.";
        }
    }

    /// <summary>Rebuilding a fresh tree resets every node's expansion — the pragmatic
    /// "collapse all" (the tree nodes are not change-notifying).</summary>
    public void CollapseAll() => RefreshTree();

    // ---- Documents ---------------------------------------------------------------

    /// <summary>Open a file node in a document tab (re-activating an already-open tab).
    /// Directories and a null node are ignored. The file read happens off the UI thread.</summary>
    public async Task OpenNodeAsync(WorkspaceTreeNode? node)
    {
        if (node is null || node.IsDirectory || !HasWorkspace)
        {
            return;
        }

        await OpenByRelativePathAsync(node.RelativePath);
    }

    private async Task OpenByRelativePathAsync(string relativePath)
    {
        var existing = OpenDocuments.FirstOrDefault(d => SamePath(d.RelativePath, relativePath));
        if (existing is not null)
        {
            ActiveDocument = existing;
            return;
        }

        // Read/classify/metadata run on the thread pool; the continuation resumes on the
        // caller's context (the UI thread in the app) to mutate the bound collections.
        var result = await _documents.OpenAsync(_workspaceRoot!, relativePath);
        if (!result.Ok || result.Document is null)
        {
            ExplorerStatus = result.Error ?? "Could not open the file.";
            return;
        }

        OpenDocuments.Add(result.Document);
        ActiveDocument = result.Document;
    }

    /// <summary>Create a file inside the workspace and open it. Returns an error string
    /// (never throws for expected problems); null on success.</summary>
    public async Task<string?> CreateFileAsync(string relativePath)
    {
        if (!HasWorkspace)
        {
            return "No workspace is open.";
        }

        var result = _explorer.CreateFile(_workspaceRoot!, relativePath);
        if (!result.Ok)
        {
            return result.Error;
        }

        RefreshTree();
        await OpenByRelativePathAsync(result.RelativePath);
        return null;
    }

    public string? CreateFolder(string relativePath)
    {
        if (!HasWorkspace)
        {
            return "No workspace is open.";
        }

        var result = _explorer.CreateFolder(_workspaceRoot!, relativePath);
        if (!result.Ok)
        {
            return result.Error;
        }

        RefreshTree();
        return null;
    }

    /// <summary>Explicitly save the active document (refused for read-only docs by the
    /// service). Returns an error string or null on success.</summary>
    public string? SaveActiveDocument()
    {
        if (_activeDocument is null)
        {
            return "No document is open.";
        }

        return _documents.Save(_activeDocument);
    }

    /// <summary>Whether the active document is the given project's examples.jsonl. Saving it
    /// through the editor is a dataset change, so the caller must invalidate the debt grade and
    /// re-check version integrity (otherwise those badges keep asserting a stale verdict).</summary>
    public bool ActiveDocumentIsDatasetFile(string? projectPath)
    {
        if (_activeDocument is null
            || string.IsNullOrWhiteSpace(_activeDocument.FullPath)
            || string.IsNullOrWhiteSpace(projectPath))
        {
            return false;
        }

        return SamePath(_activeDocument.FullPath, System.IO.Path.Combine(projectPath, "examples.jsonl"));
    }

    /// <summary>Close a document tab, re-activating a neighbour when the active tab closed.
    /// The caller is responsible for confirming unsaved changes first.</summary>
    public void CloseDocument(OpenWorkspaceDocument document)
    {
        var index = OpenDocuments.IndexOf(document);
        if (index < 0)
        {
            return;
        }

        OpenDocuments.Remove(document);
        if (ReferenceEquals(_activeDocument, document))
        {
            ActiveDocument = OpenDocuments.Count == 0
                ? null
                : OpenDocuments[Math.Min(index, OpenDocuments.Count - 1)];
        }
    }

    private void RebuildMetadataRows()
    {
        MetadataRows.Clear();
        if (_activeDocument?.Metadata is not { } meta)
        {
            return;
        }

        MetadataRows.Add(new WorkspaceMetadataRow("Path", meta.RelativePath, mono: true));
        MetadataRows.Add(new WorkspaceMetadataRow("Kind", meta.FileKind.ToString(), mono: false));
        if (meta.Exists)
        {
            MetadataRows.Add(new WorkspaceMetadataRow("Size", meta.HumanSize, mono: false));
            if (meta.LastModifiedUtc is { } modified)
            {
                MetadataRows.Add(new WorkspaceMetadataRow(
                    "Modified",
                    modified.LocalDateTime.ToString("yyyy-MM-dd HH:mm", System.Globalization.CultureInfo.InvariantCulture),
                    mono: false));
            }
        }

        if (meta.IsGeneratedArtifact)
        {
            MetadataRows.Add(new WorkspaceMetadataRow("Note", "Generated artifact — opened read-only.", mono: false));
        }
        else if (meta.IsDatasetCoreFile)
        {
            MetadataRows.Add(new WorkspaceMetadataRow("Note", "Dataset core file — the desktop is its single writer.", mono: false));
        }
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
            left.Replace('\\', '/').TrimEnd('/'),
            right.Replace('\\', '/').TrimEnd('/'),
            comparison);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        OnChanged(name);
        return true;
    }

    private void OnChanged(string? name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>One row in the Explorer's selected-file metadata panel.</summary>
public sealed class WorkspaceMetadataRow
{
    public WorkspaceMetadataRow(string key, string value, bool mono)
    {
        Key = key;
        Value = value;
        IsMono = mono;
    }

    public string Key { get; }
    public string Value { get; }
    public bool IsMono { get; }
}
