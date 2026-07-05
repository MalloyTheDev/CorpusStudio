using System.Threading.Tasks;

namespace CorpusStudio.Desktop.Services;

/// <summary>A portable file-type filter. The WPF adapter renders it to a "Name (*.ext)|*.ext"
/// string; an Avalonia adapter renders a <c>FilePickerFileType</c>. Extensions omit the dot
/// ("jsonl"); "*" means all files.</summary>
public sealed record FilePickerFilter(string Name, params string[] Extensions);

/// <summary>Head-agnostic file/folder picker seam so shared logic (and either UI head) can prompt
/// without a hard WPF dependency. Async because non-WPF pickers (Avalonia's StorageProvider) are
/// async. Returns the chosen path, or <c>null</c> when the user cancels. See
/// docs/AVALONIA_MIGRATION_PLAN.md (Phase 0).</summary>
public interface IFilePickerService
{
    /// <summary>Prompt for a folder; returns its path, or null if cancelled.</summary>
    Task<string?> PickFolderAsync(string title);

    /// <summary>Prompt for a single existing file; returns its path, or null if cancelled.</summary>
    Task<string?> PickFileAsync(string title, params FilePickerFilter[] filters);
}
