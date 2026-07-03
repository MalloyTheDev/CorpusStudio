using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace CorpusStudio.Desktop.Models;

/// <summary>An open document tab in the Workspace editor (v1.2.3 Workspace System, slice
/// 5). Dirty state is derived by comparing <see cref="TextContent"/> against the content
/// last loaded or saved. Read-only documents (generated artifacts, too-large previews,
/// images, binaries) never become dirty.</summary>
public sealed class OpenWorkspaceDocument : INotifyPropertyChanged
{
    public string DisplayName { get; init; } = string.Empty;
    public string FullPath { get; init; } = string.Empty;
    public string RelativePath { get; init; } = string.Empty;
    public WorkspaceFileKind FileKind { get; init; }
    public bool IsReadOnly { get; init; }

    /// <summary>Set for image documents — the absolute path the view binds a preview to.
    /// Null for non-image documents.</summary>
    public string? ImagePreviewPath { get; init; }

    public WorkspaceFileMetadata? Metadata { get; init; }

    private string _statusMessage = string.Empty;
    public string StatusMessage
    {
        get => _statusMessage;
        set { if (_statusMessage != value) { _statusMessage = value; OnChanged(); } }
    }

    private string _originalContent = string.Empty;
    private string _textContent = string.Empty;

    /// <summary>Editable text (empty for image/binary docs). Setting it recomputes
    /// <see cref="IsDirty"/>.</summary>
    public string TextContent
    {
        get => _textContent;
        set
        {
            if (_textContent == value) return;
            _textContent = value;
            OnChanged();
            OnChanged(nameof(IsDirty));
        }
    }

    /// <summary>True when unsaved edits exist. Always false for read-only documents.</summary>
    public bool IsDirty => !IsReadOnly && _textContent != _originalContent;

    /// <summary>Called by the document service after a successful load/save so dirty
    /// tracking has a clean baseline.</summary>
    public void MarkClean(string content)
    {
        _originalContent = content;
        _textContent = content;
        OnChanged(nameof(TextContent));
        OnChanged(nameof(IsDirty));
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
