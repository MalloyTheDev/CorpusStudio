using System;
using Avalonia;

namespace CorpusStudio.Avalonia;

internal static class Program
{
    // Avalonia entry point. Kept minimal — the spike's purpose is to prove the shared VMs bind.
    [STAThread]
    public static void Main(string[] args) =>
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .LogToTrace();
}
