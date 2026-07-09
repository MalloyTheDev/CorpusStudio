using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

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

    public Task<string?> PromptAsync(string title, string message, string defaultValue = "")
    {
        // WPF's MessageBox has no text input, so build a minimal modal input dialog in code.
        var input = new TextBox { Text = defaultValue, MinWidth = 320, Margin = new Thickness(0, 8, 0, 12) };
        var ok = new Button { Content = "OK", IsDefault = true, MinWidth = 74, Margin = new Thickness(0, 0, 8, 0) };
        var cancel = new Button { Content = "Cancel", IsCancel = true, MinWidth = 74 };
        var buttons = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
        };
        buttons.Children.Add(ok);
        buttons.Children.Add(cancel);
        var body = new StackPanel { Margin = new Thickness(16), MaxWidth = 420 };
        body.Children.Add(new TextBlock { Text = message, TextWrapping = TextWrapping.Wrap });
        body.Children.Add(input);
        body.Children.Add(buttons);

        var owner = Application.Current?.MainWindow;
        var window = new Window
        {
            Title = title,
            Content = body,
            SizeToContent = SizeToContent.WidthAndHeight,
            ResizeMode = ResizeMode.NoResize,
            ShowInTaskbar = false,
            WindowStartupLocation = owner is null ? WindowStartupLocation.CenterScreen : WindowStartupLocation.CenterOwner,
            Owner = owner,
        };
        ok.Click += (_, _) => { window.DialogResult = true; };
        input.Loaded += (_, _) => { input.SelectAll(); input.Focus(); };

        var confirmed = window.ShowDialog() == true;
        return Task.FromResult<string?>(confirmed ? input.Text : null);
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
