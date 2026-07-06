using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using CorpusStudio.Desktop.Views;
using Microsoft.Extensions.DependencyInjection;

namespace CorpusStudio.Desktop;

public partial class App : Application
{
    /// <summary>Install global exception handlers, then compose the app from a DI container
    /// (backlog #4) and show the main window with its injected view-model.</summary>
    protected override void OnStartup(StartupEventArgs e)
    {
        DispatcherUnhandledException += OnDispatcherUnhandledException;
        AppDomain.CurrentDomain.UnhandledException += OnDomainUnhandledException;
        TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;
        base.OnStartup(e);

        var services = new ServiceCollection();
        ConfigureServices(services);
        var provider = services.BuildServiceProvider();

        var window = new MainWindow
        {
            DataContext = provider.GetRequiredService<MainWindowViewModel>(),
            Dialogs = provider.GetRequiredService<IDialogService>(),
            FilePicker = provider.GetRequiredService<IFilePickerService>(),
        };
        window.Show();
    }

    /// <summary>Compose the view-model graph. Per-tab view-models are registered behind their
    /// interfaces so the shell (and tests) depend on the contract, not the concrete class.</summary>
    private static void ConfigureServices(IServiceCollection services)
    {
        services.AddSingleton<IDialogService, MessageBoxDialogService>();
        services.AddSingleton<IFilePickerService, Win32FilePickerService>();
        services.AddTransient<IDebtViewModel, DebtViewModel>();
        services.AddTransient<IArenaViewModel, ArenaViewModel>();
        services.AddTransient<ISettingsViewModel, SettingsViewModel>();
        services.AddTransient<IVersionsViewModel, VersionsViewModel>();
        services.AddTransient<IArtifactsViewModel, ArtifactsViewModel>();
        services.AddTransient<ISuitesViewModel, SuitesViewModel>();
        services.AddTransient<ISplitsViewModel, SplitsViewModel>();
        services.AddTransient<IPreferenceReviewViewModel, PreferenceReviewViewModel>();
        services.AddTransient<IQuarantineViewModel, QuarantineViewModel>();
        services.AddTransient<IExamplesViewModel, ExamplesViewModel>();
        services.AddTransient<IWritingStudioViewModel, WritingStudioViewModel>();
        services.AddTransient<IAiAssistRewriteBatchesViewModel, AiAssistRewriteBatchesViewModel>();
        services.AddTransient<IAiAssistConnectionViewModel, AiAssistConnectionViewModel>();
        services.AddTransient<IEvaluationConnectionViewModel, EvaluationConnectionViewModel>();
        services.AddTransient<IQualityViewModel, QualityViewModel>();
        services.AddSingleton<CorpusStudio.Desktop.Services.IEngineService, CorpusStudio.Desktop.Services.PythonEngineService>();
            services.AddTransient<MainWindowViewModel>();
    }

    private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        // UI-thread errors are usually recoverable — report and keep the app alive.
        ReportCrash("A UI error occurred", e.Exception);
        e.Handled = true;
    }

    private void OnDomainUnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        ReportCrash("A fatal error occurred", e.ExceptionObject as Exception);
    }

    private void OnUnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        ReportCrash("A background error occurred", e.Exception);
        e.SetObserved();
    }

    private static void ReportCrash(string title, Exception? exception)
    {
        var logPath = WriteCrashLog(title, exception);
        var message = exception?.Message ?? "An unknown error occurred.";
        try
        {
            MessageBox.Show(
                $"{message}\n\nDetails were written to:\n{logPath}",
                $"Corpus Studio — {title}",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
        }
        catch
        {
            // Never let the crash reporter itself throw.
        }
    }

    private static string WriteCrashLog(string title, Exception? exception)
    {
        try
        {
            var directory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "CorpusStudio");
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, "crash.log");

            var entry = new StringBuilder()
                .AppendLine($"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {title}: "
                    + $"{exception?.GetType().Name}: {exception?.Message}")
                .AppendLine(exception?.ToString() ?? "(no exception object)")
                .AppendLine(new string('-', 72))
                .ToString();
            File.AppendAllText(path, entry, Encoding.UTF8);
            return path;
        }
        catch
        {
            return "(could not write a crash log)";
        }
    }
}
