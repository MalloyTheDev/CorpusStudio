using System;

namespace CorpusStudio.Desktop.Services;

/// <summary>Head-agnostic periodic timer whose <see cref="Tick"/> fires on the UI thread — the
/// seam the shared view-model uses to drive the training-run log flush + checkpoint poll without
/// depending on <c>System.Windows.Threading.DispatcherTimer</c>. WPF/Avalonia adapters wrap their
/// own dispatcher timer; a fake ticks on demand in tests (mirrors <see cref="IProcessRunner"/>).</summary>
public interface IDispatcherTimer
{
    /// <summary>How often <see cref="Tick"/> fires while the timer is running.</summary>
    TimeSpan Interval { get; set; }

    /// <summary>True between <see cref="Start"/> and <see cref="Stop"/>.</summary>
    bool IsRunning { get; }

    /// <summary>Raised on the UI thread every <see cref="Interval"/> while running.</summary>
    event EventHandler Tick;

    void Start();

    void Stop();
}

/// <summary>Creates <see cref="IDispatcherTimer"/> instances — injected so the shared view-model can
/// own its run timers without knowing which head (WPF/Avalonia) supplies them.</summary>
public interface IDispatcherTimerFactory
{
    IDispatcherTimer Create();
}
