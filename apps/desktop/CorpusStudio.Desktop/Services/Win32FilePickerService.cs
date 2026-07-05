using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Win32;

namespace CorpusStudio.Desktop.Services;

/// <summary>WPF implementation of <see cref="IFilePickerService"/> over Microsoft.Win32's
/// OpenFolderDialog/OpenFileDialog. Synchronous under the hood, wrapped in completed tasks; an
/// Avalonia head supplies a <c>StorageProvider</c>-based implementation of the same interface.
/// Uses the app's main window as owner. The filter rendering is pure/public for unit tests.</summary>
public sealed class Win32FilePickerService : IFilePickerService
{
    public Task<string?> PickFolderAsync(string title)
    {
        var dialog = new OpenFolderDialog { Title = title };
        return Task.FromResult(ShowDialog(dialog) == true ? dialog.FolderName : null);
    }

    public Task<string?> PickFileAsync(string title, params FilePickerFilter[] filters)
    {
        var dialog = new OpenFileDialog
        {
            Title = title,
            Filter = ToWpfFilter(filters),
            CheckFileExists = true,
            Multiselect = false,
        };
        return Task.FromResult(ShowDialog(dialog) == true ? dialog.FileName : null);
    }

    private static bool? ShowDialog(CommonDialog dialog)
    {
        var owner = Application.Current?.MainWindow;
        return owner is null ? dialog.ShowDialog() : dialog.ShowDialog(owner);
    }

    /// <summary>Render portable filters to a WPF "Name (*.ext)|*.ext|..." filter string. An empty
    /// set falls back to all-files.</summary>
    public static string ToWpfFilter(IReadOnlyList<FilePickerFilter> filters)
    {
        if (filters is null || filters.Count == 0)
        {
            return "All files (*.*)|*.*";
        }

        return string.Join("|", filters.Select(filter =>
        {
            var patterns = string.Join(";", filter.Extensions.Select(ext => ext == "*" ? "*.*" : $"*.{ext}"));
            return $"{filter.Name} ({patterns})|{patterns}";
        }));
    }
}
