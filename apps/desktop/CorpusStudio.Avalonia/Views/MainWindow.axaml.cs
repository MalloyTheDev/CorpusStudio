using Avalonia.Controls;
using CorpusStudio.Avalonia.Services;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Avalonia.Views;

// The activity bar binds Command="{Binding X}" to the shell VM's ICommands (RelayCommand), so this
// head needs no code-behind for navigation. The platform seams (dialogs / file picker) are set from
// DI in App.axaml.cs and default to the Avalonia adapters (issue #185), mirroring the WPF head — so
// per-tab operations can route through them when their views are ported (issue #186).
public partial class MainWindow : Window
{
    public IDialogService Dialogs { get; set; } = new AvaloniaDialogService();
    public IFilePickerService FilePicker { get; set; } = new AvaloniaFilePickerService();

    public MainWindow() => InitializeComponent();
}
