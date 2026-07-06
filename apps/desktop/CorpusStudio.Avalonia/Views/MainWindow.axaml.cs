using Avalonia.Controls;

namespace CorpusStudio.Avalonia.Views;

// The activity bar binds Command="{Binding X}" to the shell VM's ICommands (RelayCommand), so this
// head needs no code-behind — the whole window is data-bound to the shared CorpusStudio.Core VMs.
public partial class MainWindow : Window
{
    public MainWindow() => InitializeComponent();
}
