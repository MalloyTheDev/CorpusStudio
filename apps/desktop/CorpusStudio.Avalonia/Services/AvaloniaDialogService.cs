using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Layout;
using Avalonia.Media;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Avalonia.Services;

/// <summary>Avalonia adapter for <see cref="IDialogService"/> (issue #185), mirroring the WPF
/// MessageBox adapter with a modal Avalonia <see cref="Window"/> (Avalonia has no MessageBox). The
/// owner window is resolved at call time from the desktop lifetime, so this composes in DI before the
/// main window exists and needs no ctor dependency on it. If there is no window (headless), confirms
/// return the safe/negative default and messages no-op.</summary>
public sealed class AvaloniaDialogService : IDialogService
{
    private static Window? Owner =>
        (Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.MainWindow;

    public async Task<bool> ConfirmAsync(
        string message,
        string title,
        DialogButtons buttons = DialogButtons.YesNo,
        DialogSeverity severity = DialogSeverity.Question,
        bool defaultAffirmative = false)
    {
        var owner = Owner;
        if (owner is null)
        {
            return false;
        }

        var (affirmText, negativeText) = buttons == DialogButtons.OkCancel ? ("OK", "Cancel") : ("Yes", "No");
        var result = false;
        var dialog = NewDialog(title);
        var affirm = new Button { Content = affirmText, IsDefault = defaultAffirmative, MinWidth = 84, Margin = new Thickness(0, 0, 8, 0) };
        var negative = new Button { Content = negativeText, IsCancel = true, MinWidth = 84 };
        affirm.Click += (_, _) => { result = true; dialog.Close(); };
        negative.Click += (_, _) => { result = false; dialog.Close(); };
        dialog.Content = BuildBody(message, affirm, negative);
        await dialog.ShowDialog(owner);
        return result;
    }

    public async Task ShowAsync(string message, string title, DialogSeverity severity = DialogSeverity.Information)
    {
        var owner = Owner;
        if (owner is null)
        {
            return;
        }

        var dialog = NewDialog(title);
        var ok = new Button { Content = "OK", IsDefault = true, MinWidth = 84 };
        ok.Click += (_, _) => dialog.Close();
        dialog.Content = BuildBody(message, ok);
        await dialog.ShowDialog(owner);
    }

    private static Window NewDialog(string title) => new()
    {
        Title = title,
        SizeToContent = SizeToContent.WidthAndHeight,
        WindowStartupLocation = WindowStartupLocation.CenterOwner,
        CanResize = false,
        ShowInTaskbar = false,
    };

    private static Control BuildBody(string message, params Button[] buttons)
    {
        var buttonRow = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(0, 14, 0, 0),
        };
        foreach (var button in buttons)
        {
            buttonRow.Children.Add(button);
        }

        var panel = new StackPanel { Margin = new Thickness(20), MaxWidth = 420 };
        panel.Children.Add(new TextBlock { Text = message, TextWrapping = TextWrapping.Wrap });
        panel.Children.Add(buttonRow);
        return panel;
    }
}
