using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class TrainingRunClassifierTests
{
    [Fact]
    public void CleanZeroExit_IsSucceeded()
    {
        var outcome = TrainingRunClassifier.Classify(exitCode: 0, cancelRequested: false, error: null);
        Assert.Equal(TrainingRunOutcome.Succeeded, outcome.Status);
        Assert.Equal(0, outcome.ExitCode);
        Assert.Null(outcome.Note);
    }

    [Fact]
    public void CleanNonZeroExit_IsFailed_KeepingTheCode()
    {
        var outcome = TrainingRunClassifier.Classify(exitCode: 1, cancelRequested: false, error: null);
        Assert.Equal(TrainingRunOutcome.Failed, outcome.Status);
        Assert.Equal(1, outcome.ExitCode);
        Assert.Null(outcome.Note);
    }

    [Fact]
    public void UserCancelWithCleanExit_IsCancelled_KeepingTheCode()
    {
        var outcome = TrainingRunClassifier.Classify(exitCode: 3, cancelRequested: true, error: null);
        Assert.Equal(TrainingRunOutcome.Cancelled, outcome.Status);
        Assert.Equal(3, outcome.ExitCode);
    }

    [Fact]
    public void OperationCanceled_IsCancelled_WithNoExitCode()
    {
        var outcome = TrainingRunClassifier.Classify(exitCode: null, cancelRequested: false, error: new OperationCanceledException());
        Assert.Equal(TrainingRunOutcome.Cancelled, outcome.Status);
        Assert.Null(outcome.ExitCode);
        Assert.Null(outcome.Note);
    }

    [Fact]
    public void OtherException_IsFailed_CarryingTheMessage()
    {
        var outcome = TrainingRunClassifier.Classify(exitCode: null, cancelRequested: false, error: new InvalidOperationException("boom"));
        Assert.Equal(TrainingRunOutcome.Failed, outcome.Status);
        Assert.Null(outcome.ExitCode);
        Assert.Equal("boom", outcome.Note);
    }

    [Fact]
    public void CancellationException_TakesPrecedenceOverCancelFlag()
    {
        var outcome = TrainingRunClassifier.Classify(exitCode: null, cancelRequested: true, error: new OperationCanceledException());
        Assert.Equal(TrainingRunOutcome.Cancelled, outcome.Status);
    }

    [Fact]
    public void NonCancelException_WinsOverCancelFlag_MatchingTheLaunchFlow()
    {
        // A non-cancellation error is a failure even if a cancel had been requested —
        // mirrors the original catch(Exception) branch, which never checked the flag.
        var outcome = TrainingRunClassifier.Classify(exitCode: null, cancelRequested: true, error: new InvalidOperationException("io"));
        Assert.Equal(TrainingRunOutcome.Failed, outcome.Status);
        Assert.Equal("io", outcome.Note);
    }

    // --- The IProcessRunner seam is drivable by a fake (the shape 2b's orchestration
    //     tests will use — no real process spawn). ------------------------------------

    private sealed class FakeProcessRunner : IProcessRunner
    {
        private readonly int _exitCode;
        private readonly IReadOnlyList<string> _lines;
        public IReadOnlyList<string>? LastArgv { get; private set; }
        public bool KillCalled { get; private set; }

        public FakeProcessRunner(int exitCode, params string[] lines)
        {
            _exitCode = exitCode;
            _lines = lines;
        }

        public Task<int> RunAsync(
            IReadOnlyList<string> argv,
            string? workingDirectory,
            Action<string> onOutputLine,
            CancellationToken cancellationToken,
            Action<int, DateTime?>? onStarted = null)
        {
            LastArgv = argv;
            onStarted?.Invoke(4242, null);
            foreach (var line in _lines)
            {
                onOutputLine(line);
            }
            return Task.FromResult(_exitCode);
        }

        public void TryKillCurrent() => KillCalled = true;
    }

    [Fact]
    public async Task FakeProcessRunner_DrivesOutputStartedAndExit()
    {
        var runner = new FakeProcessRunner(0, "epoch 1", "epoch 2");
        var captured = new List<string>();
        var startedPid = 0;

        var exit = await runner.RunAsync(
            new[] { "trainer", "config.yaml" },
            workingDirectory: null,
            onOutputLine: captured.Add,
            cancellationToken: CancellationToken.None,
            onStarted: (pid, _) => startedPid = pid);

        Assert.Equal(0, exit);
        Assert.Equal(new[] { "epoch 1", "epoch 2" }, captured);
        Assert.Equal(4242, startedPid);
        Assert.Equal(new[] { "trainer", "config.yaml" }, runner.LastArgv);
    }
}
