using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

namespace CorpusStudio.Desktop;

public partial class App : Application
{
    /// <summary>Install global exception handlers so an unhandled error surfaces as a dialog +
    /// a crash log instead of a silent process death (the app previously had none).</summary>
    protected override void OnStartup(StartupEventArgs e)
    {
        DispatcherUnhandledException += OnDispatcherUnhandledException;
        AppDomain.CurrentDomain.UnhandledException += OnDomainUnhandledException;
        TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;
        base.OnStartup(e);
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
