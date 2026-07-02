using System.Collections.Concurrent;
using System.Threading;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class TrainingProcessRunnerTests
{
    private static IReadOnlyList<string> Cmd(string command) => new[] { "cmd", "/c", command };

    [Fact]
    public async Task RunAsync_StreamsOutput_AndReturnsExitCode()
    {
        var runner = new TrainingProcessRunner();
        var lines = new ConcurrentQueue<string>();
        var exit = await runner.RunAsync(Cmd("echo streamed-line"), null, lines.Enqueue, CancellationToken.None);
        Assert.Equal(0, exit);
        Assert.Contains(lines, line => line.Contains("streamed-line"));
    }

    [Fact]
    public async Task RunAsync_ReturnsNonZeroExitCode()
    {
        var runner = new TrainingProcessRunner();
        var exit = await runner.RunAsync(Cmd("exit 3"), null, _ => { }, CancellationToken.None);
        Assert.Equal(3, exit);
    }

    [Fact]
    public async Task RunAsync_Cancellation_KillsProcessAndThrows()
    {
        var runner = new TrainingProcessRunner();
        using var cts = new CancellationTokenSource();
        cts.CancelAfter(TimeSpan.FromMilliseconds(300));

        // A ~30s process that cancellation must kill well before it finishes.
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => runner.RunAsync(Cmd("ping -n 30 127.0.0.1 >nul"), null, _ => { }, cts.Token)
        );
    }

    [Fact]
    public async Task RunAsync_EmptyArgv_Throws()
    {
        var runner = new TrainingProcessRunner();
        await Assert.ThrowsAsync<ArgumentException>(
            () => runner.RunAsync(Array.Empty<string>(), null, _ => { }, CancellationToken.None)
        );
    }

    [Fact]
    public async Task RunAsync_MissingExecutable_ThrowsInvalidOperation()
    {
        var runner = new TrainingProcessRunner();
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => runner.RunAsync(new[] { "corpus-studio-no-such-exe-xyz" }, null, _ => { }, CancellationToken.None)
        );
    }
}
