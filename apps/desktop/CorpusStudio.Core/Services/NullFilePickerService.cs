using System.Threading.Tasks;

namespace CorpusStudio.Desktop.Services;

/// <summary>No-op <see cref="IFilePickerService"/> for design-time / headless view-model construction
/// (no UI available). Both pickers return null (as if the user cancelled). The real heads inject
/// <c>Win32FilePickerService</c> (WPF) or <c>AvaloniaFilePickerService</c> via DI; this is only the
/// fallback for the parameterless design-time constructor so existing <c>new MainWindowViewModel()</c>
/// call sites keep working. Mirrors <see cref="NullDialogService"/>.</summary>
public sealed class NullFilePickerService : IFilePickerService
{
    public Task<string?> PickFolderAsync(string title) => Task.FromResult<string?>(null);

    public Task<string?> PickFileAsync(string title, params FilePickerFilter[] filters) => Task.FromResult<string?>(null);
}
