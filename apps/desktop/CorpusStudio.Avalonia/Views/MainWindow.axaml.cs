using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Styling;
using CorpusStudio.Avalonia.Services;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;

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

    public MainWindow()
    {
        InitializeComponent();
        // The Settings → Appearance segmented toggle reflects the ACTUAL applied variant (which is
        // always Light or Dark, even when the app follows the OS via Default). Keep it in sync whenever
        // the resolved variant changes; also seed it once the visual tree is up.
        ActualThemeVariantChanged += (_, _) => SyncThemeSegments();
        Loaded += (_, _) => SyncThemeSegments();
    }

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

    // Settings → Appearance segmented toggle. Choosing a segment pins the app variant explicitly
    // (Dark/Light); ActualThemeVariantChanged then re-syncs which segment reads as selected.
    private void SelectDarkTheme_Click(object? sender, RoutedEventArgs e)
    {
        if (Application.Current is { } app)
        {
            app.RequestedThemeVariant = ThemeVariant.Dark;
        }
    }

    private void SelectLightTheme_Click(object? sender, RoutedEventArgs e)
    {
        if (Application.Current is { } app)
        {
            app.RequestedThemeVariant = ThemeVariant.Light;
        }
    }

    // Per-item "Repair" on an Import & Quarantine reject card. The shell exposes the retry as a public
    // method (RetrySelectedImportQuarantineItem), not an ICommand, and the card list is an ItemsControl
    // with no selection — so this view handler mirrors the WPF head's RetryQuarantineItemButton_Click:
    // select the clicked row, load its raw text into the Writing Studio draft (the shell tracks the
    // pending row so a successful save clears it from quarantine), then reveal Writing Studio for the
    // edit. MainWindowViewModel is left untouched — no command is added there.
    private void RepairQuarantineItem_Click(object? sender, RoutedEventArgs e)
    {
        if (sender is Button { DataContext: ImportQuarantineItem item }
            && DataContext is MainWindowViewModel vm)
        {
            vm.Quarantine.SelectedImportQuarantineItem = item;
            vm.RetrySelectedImportQuarantineItem();
            vm.SelectStudioTabCommand.Execute("WritingStudio");
        }
    }

    // Mark the segment matching the resolved variant as selected (accent-soft pill), the other neutral.
    private void SyncThemeSegments()
    {
        if (this.FindControl<Button>("ThemeDarkButton") is not { } dark
            || this.FindControl<Button>("ThemeLightButton") is not { } light)
        {
            return;
        }

        var isDark = ActualThemeVariant == ThemeVariant.Dark;
        dark.Classes.Set("selected", isDark);
        light.Classes.Set("selected", !isDark);
    }
}
