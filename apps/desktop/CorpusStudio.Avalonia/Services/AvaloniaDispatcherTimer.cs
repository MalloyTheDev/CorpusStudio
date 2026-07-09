using System;
using Avalonia.Threading;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Avalonia.Services;

/// <summary>Avalonia adapter for <see cref="IDispatcherTimer"/> over
/// <see cref="Avalonia.Threading.DispatcherTimer"/> (ticks on the UI thread), mirroring the WPF
/// <c>WpfDispatcherTimer</c> so the shared view-model's run timers work on both heads.</summary>
public sealed class AvaloniaDispatcherTimer : IDispatcherTimer
{
    private readonly DispatcherTimer _timer = new(DispatcherPriority.Background);

    public AvaloniaDispatcherTimer()
    {
        _timer.Tick += (_, _) => Tick?.Invoke(this, EventArgs.Empty);
    }

    public TimeSpan Interval
    {
        get => _timer.Interval;
        set => _timer.Interval = value;
    }

    public bool IsRunning => _timer.IsEnabled;

    public event EventHandler? Tick;

    public void Start() => _timer.Start();

    public void Stop() => _timer.Stop();
}

/// <summary>Creates Avalonia dispatcher timers.</summary>
public sealed class AvaloniaDispatcherTimerFactory : IDispatcherTimerFactory
{
    public IDispatcherTimer Create() => new AvaloniaDispatcherTimer();
}
