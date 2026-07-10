using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Styling;
using CorpusStudio.Avalonia.Services;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Avalonia.Views;

// The activity bar binds Command="{Binding X}" to the shell VM's ICommands (RelayCommand), so this
// head needs no code-behind for navigation. The platform seams (dialogs / file picker) are set from
// DI in App.axaml.cs and default to the Avalonia adapters (issue #185), mirroring the WPF head — so
// per-tab operations can route through them when their views are ported (issue #186). The one
// handler here is the theme toggle (#187/#201): light/dark is an Application-level View concern.
public partial class MainWindow : Window
{
    public IDialogService Dialogs { get; set; } = new AvaloniaDialogService();
    public IFilePickerService FilePicker { get; set; } = new AvaloniaFilePickerService();

    public MainWindow() => InitializeComponent();

    // Cycle the app theme: System (follow OS) → Light → Dark → System. FluentTheme re-styles the
    // built-in controls and the ThemeDictionaries re-resolve the app's own colors on the change.
    private void CycleTheme_Click(object? sender, RoutedEventArgs e)
    {
        if (Application.Current is not { } app)
        {
            return;
        }

        app.RequestedThemeVariant =
            app.RequestedThemeVariant == ThemeVariant.Light ? ThemeVariant.Dark
            : app.RequestedThemeVariant == ThemeVariant.Dark ? ThemeVariant.Default
            : ThemeVariant.Light;
    }
}
