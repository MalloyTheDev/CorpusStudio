using System;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The head-agnostic dispatcher-timer seam (#246): the training-run launch drives its
/// log-flush + checkpoint-poll timers through this instead of a View-owned DispatcherTimer.</summary>
public sealed class DispatcherTimerTests
{
    [Fact]
    public void NullTimer_TracksStartStop_AndNeverTicks()
    {
        var ticks = 0;
        var timer = new NullDispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        timer.Tick += (_, _) => ticks++;

        Assert.False(timer.IsRunning);
        timer.Start();
        Assert.True(timer.IsRunning);
        timer.Stop();
        Assert.False(timer.IsRunning);
        Assert.Equal(0, ticks); // the no-op timer never fires (used at design time)
    }

    [Fact]
    public void NullFactory_CreatesDistinctTimers()
    {
        var factory = new NullDispatcherTimerFactory();
        var a = factory.Create();
        var b = factory.Create();
        Assert.NotSame(a, b);
        Assert.IsType<NullDispatcherTimer>(a);
    }
}
