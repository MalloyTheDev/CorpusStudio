using System;
using System.Windows.Threading;

namespace CorpusStudio.Desktop.Services;

/// <summary>WPF adapter for <see cref="IDispatcherTimer"/> over
/// <see cref="System.Windows.Threading.DispatcherTimer"/> (ticks on the UI thread at Background
/// priority — matching the training-run timers this replaces).</summary>
public sealed class WpfDispatcherTimer : IDispatcherTimer
{
    private readonly DispatcherTimer _timer = new(DispatcherPriority.Background);

    public WpfDispatcherTimer()
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

/// <summary>Creates WPF dispatcher timers.</summary>
public sealed class WpfDispatcherTimerFactory : IDispatcherTimerFactory
{
    public IDispatcherTimer Create() => new WpfDispatcherTimer();
}
