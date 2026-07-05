using System.Threading.Tasks;
using System.Windows;

namespace CorpusStudio.Desktop.Services;

/// <summary>WPF implementation of <see cref="IDialogService"/> over <see cref="MessageBox"/>.
/// Synchronous under the hood (WPF's MessageBox blocks the UI thread), wrapped in completed tasks;
/// an Avalonia head will supply a genuinely-async implementation of the same interface. Uses the
/// app's main window as the owner. The enum→WPF mapping is pure and public so it is unit-testable
/// without a UI thread.</summary>
public sealed class MessageBoxDialogService : IDialogService
{
    public Task<bool> ConfirmAsync(
        string message, string title, DialogButtons buttons, DialogSeverity severity, bool defaultAffirmative)
    {
        var button = ToButton(buttons);
        var result = Show(message, title, button, ToImage(severity), DefaultResult(button, defaultAffirmative));
        return Task.FromResult(IsAffirmative(result));
    }

    public Task ShowAsync(string message, string title, DialogSeverity severity)
    {
        Show(message, title, MessageBoxButton.OK, ToImage(severity), MessageBoxResult.OK);
        return Task.CompletedTask;
    }

    private static MessageBoxResult Show(
        string message, string title, MessageBoxButton button, MessageBoxImage image, MessageBoxResult defaultResult)
    {
        var owner = Application.Current?.MainWindow;
        return owner is null
            ? MessageBox.Show(message, title, button, image, defaultResult)
            : MessageBox.Show(owner, message, title, button, image, defaultResult);
    }

    // --- pure, testable mapping ------------------------------------------------

    public static MessageBoxButton ToButton(DialogButtons buttons) => buttons switch
    {
        DialogButtons.OkCancel => MessageBoxButton.OKCancel,
        _ => MessageBoxButton.YesNo,
    };

    public static MessageBoxImage ToImage(DialogSeverity severity) => severity switch
    {
        DialogSeverity.Warning => MessageBoxImage.Warning,
        DialogSeverity.Error => MessageBoxImage.Error,
        DialogSeverity.Question => MessageBoxImage.Question,
        _ => MessageBoxImage.Information,
    };

    public static MessageBoxResult DefaultResult(MessageBoxButton button, bool affirmative)
    {
        if (affirmative)
        {
            return button == MessageBoxButton.OKCancel ? MessageBoxResult.OK : MessageBoxResult.Yes;
        }

        return button == MessageBoxButton.OKCancel ? MessageBoxResult.Cancel : MessageBoxResult.No;
    }

    public static bool IsAffirmative(MessageBoxResult result) =>
        result is MessageBoxResult.Yes or MessageBoxResult.OK;
}
