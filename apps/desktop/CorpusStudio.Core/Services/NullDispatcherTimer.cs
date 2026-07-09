using System;

namespace CorpusStudio.Desktop.Services;

/// <summary>No-op <see cref="IDispatcherTimer"/> that never ticks (tracks Start/Stop only) — the
/// default for the parameterless design-time constructor so existing <c>new MainWindowViewModel()</c>
/// call sites keep working, and the fallback for any head without a real dispatcher timer.</summary>
public sealed class NullDispatcherTimer : IDispatcherTimer
{
    public TimeSpan Interval { get; set; }
    public bool IsRunning { get; private set; }
#pragma warning disable CS0067 // Tick is never raised by the no-op timer (by design).
    public event EventHandler? Tick;
#pragma warning restore CS0067
    public void Start() => IsRunning = true;
    public void Stop() => IsRunning = false;
}

/// <summary>Creates <see cref="NullDispatcherTimer"/> instances (the design-time / no-op default).</summary>
public sealed class NullDispatcherTimerFactory : IDispatcherTimerFactory
{
    public IDispatcherTimer Create() => new NullDispatcherTimer();
}
