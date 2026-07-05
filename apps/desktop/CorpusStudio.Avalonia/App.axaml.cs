using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using CorpusStudio.Avalonia.Views;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Microsoft.Extensions.DependencyInjection;

namespace CorpusStudio.Avalonia;

public partial class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            // Mirror the WPF head's DI composition (App.xaml.cs) EXACTLY — same per-tab VMs behind
            // the same interfaces, same MainWindowViewModel — to prove the shared graph composes
            // and binds identically on Avalonia.
            var services = new ServiceCollection();
            services.AddTransient<IDebtViewModel, DebtViewModel>();
            services.AddTransient<IArenaViewModel, ArenaViewModel>();
            services.AddTransient<ISettingsViewModel, SettingsViewModel>();
            services.AddTransient<IVersionsViewModel, VersionsViewModel>();
            services.AddTransient<IArtifactsViewModel, ArtifactsViewModel>();
            services.AddTransient<ISuitesViewModel, SuitesViewModel>();
            services.AddTransient<ISplitsViewModel, SplitsViewModel>();
            services.AddTransient<MainWindowViewModel>();
            var provider = services.BuildServiceProvider();

            desktop.MainWindow = new MainWindow
            {
                DataContext = provider.GetRequiredService<MainWindowViewModel>(),
            };
        }

        base.OnFrameworkInitializationCompleted();
    }
}
