namespace CorpusStudio.Desktop.Services;

/// <summary>The terminal outcome of a training run: the status recorded on the run
/// record, the exit code (null when the process never cleanly exited), and a note
/// (the error message when the run failed with an exception).</summary>
public readonly record struct TrainingRunOutcome(string Status, int? ExitCode, string? Note)
{
    public const string Succeeded = "succeeded";
    public const string Failed = "failed";
    public const string Cancelled = "cancelled";
}

/// <summary>Pure classification of a training run's result — extracted from the
/// launch code-behind so the "run → terminal status" decision is unit-testable
/// (independent of the live process, timers, and UI marshaling).</summary>
public static class TrainingRunClassifier
{
    /// <summary>Map a completed run to its terminal outcome.
    ///
    /// Precedence mirrors the launch flow exactly:
    /// a cancellation (<see cref="System.OperationCanceledException"/>) → cancelled;
    /// any other error → failed, carrying the message as the note; an explicit
    /// user cancel request that still returned an exit code → cancelled (keeping the
    /// code); otherwise exit 0 → succeeded, non-zero → failed.</summary>
    public static TrainingRunOutcome Classify(int? exitCode, bool cancelRequested, System.Exception? error)
    {
        if (error is System.OperationCanceledException)
        {
            return new TrainingRunOutcome(TrainingRunOutcome.Cancelled, null, null);
        }

        if (error is not null)
        {
            return new TrainingRunOutcome(TrainingRunOutcome.Failed, null, error.Message);
        }

        if (cancelRequested)
        {
            return new TrainingRunOutcome(TrainingRunOutcome.Cancelled, exitCode, null);
        }

        var status = exitCode == 0 ? TrainingRunOutcome.Succeeded : TrainingRunOutcome.Failed;
        return new TrainingRunOutcome(status, exitCode, null);
    }
}
