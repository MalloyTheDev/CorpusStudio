using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform.Storage;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Avalonia.Services;

/// <summary>Avalonia adapter for <see cref="IFilePickerService"/> (issue #185) using the cross-platform
/// <c>StorageProvider</c>. Owner resolved at call time from the desktop lifetime (no window in the ctor).
/// Returns the chosen local path, or null on cancel / a non-local (e.g. cloud) pick.</summary>
public sealed class AvaloniaFilePickerService : IFilePickerService
{
    private static Window? Owner =>
        (Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.MainWindow;

    public async Task<string?> PickFolderAsync(string title)
    {
        var owner = Owner;
        if (owner is null)
        {
            return null;
        }

        var folders = await owner.StorageProvider.OpenFolderPickerAsync(
            new FolderPickerOpenOptions { Title = title, AllowMultiple = false });
        return folders.Count > 0 ? folders[0].TryGetLocalPath() : null;
    }

    public async Task<string?> PickFileAsync(string title, params FilePickerFilter[] filters)
    {
        var owner = Owner;
        if (owner is null)
        {
            return null;
        }

        var options = new FilePickerOpenOptions { Title = title, AllowMultiple = false };
        if (filters.Length > 0)
        {
            options.FileTypeFilter = filters
                .Select(filter => new FilePickerFileType(filter.Name)
                {
                    Patterns = filter.Extensions.Select(ext => "*." + ext.TrimStart('.')).ToList(),
                })
                .ToList<FilePickerFileType>();
        }

        var files = await owner.StorageProvider.OpenFilePickerAsync(options);
        return files.Count > 0 ? files[0].TryGetLocalPath() : null;
    }
}
