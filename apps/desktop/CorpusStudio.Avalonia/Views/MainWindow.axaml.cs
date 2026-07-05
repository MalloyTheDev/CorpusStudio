using Avalonia.Controls;
using Avalonia.Interactivity;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Avalonia.Views;

public partial class MainWindow : Window
{
    public MainWindow() => InitializeComponent();

    // Activity-bar handlers. The shell has no ICommand yet (same as the WPF head, which uses
    // code-behind _Click), so these call the shell view-model directly to switch mode / toggle panels.
    private MainWindowViewModel? Vm => DataContext as MainWindowViewModel;

    private void ActivityHome_Click(object? sender, RoutedEventArgs e) => Vm?.ShowStartCenter();
    private void ActivityFiles_Click(object? sender, RoutedEventArgs e) => Vm?.ShowFiles();
    private void ActivityStudio_Click(object? sender, RoutedEventArgs e) => Vm?.ShowStudio();
    private void ToggleProblems_Click(object? sender, RoutedEventArgs e) => Vm?.ToggleProblemsPanel();
    private void ToggleOutput_Click(object? sender, RoutedEventArgs e) => Vm?.ToggleOutputPanel();
}
