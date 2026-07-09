using System.Threading.Tasks;

namespace CorpusStudio.Desktop.Services;

/// <summary>No-op <see cref="IDialogService"/> for design-time / headless view-model construction
/// (no UI available). Confirms return the caller's safe default (<paramref name="defaultAffirmative"/>)
/// and Show is a no-op. The real heads inject <c>MessageBoxDialogService</c> (WPF) or
/// <c>AvaloniaDialogService</c> via DI; this is only the fallback for the parameterless design-time
/// constructor so existing <c>new MainWindowViewModel()</c> call sites keep working.</summary>
public sealed class NullDialogService : IDialogService
{
    public Task<bool> ConfirmAsync(
        string message,
        string title,
        DialogButtons buttons = DialogButtons.YesNo,
        DialogSeverity severity = DialogSeverity.Question,
        bool defaultAffirmative = false)
        => Task.FromResult(defaultAffirmative);

    public Task ShowAsync(string message, string title, DialogSeverity severity = DialogSeverity.Information)
        => Task.CompletedTask;
}
