using System.Threading;

namespace CorpusStudio.Desktop.Services;

/// <summary>Head-agnostic seam for spawning an external process from a structured
/// argv and streaming its output line by line — the trainer-launch abstraction the
/// desktop code-behind uses so the launch orchestration can be driven by a fake in
/// tests (mirrors <see cref="IEngineService"/> / <see cref="IDialogService"/> /
/// <see cref="IFilePickerService"/>). The engine never runs this; it only produces
/// the argv, and the user confirms the command before it is spawned.</summary>
public interface IProcessRunner
{
    /// <summary>Run <paramref name="argv"/> to completion, invoking
    /// <paramref name="onOutputLine"/> for each stdout/stderr line (on a background
    /// thread — the caller marshals to the UI). Returns the exit code; throws
    /// <see cref="System.OperationCanceledException"/> after killing the process tree
    /// if cancelled. <paramref name="onStarted"/> reports the pid and start time.</summary>
    Task<int> RunAsync(
        IReadOnlyList<string> argv,
        string? workingDirectory,
        Action<string> onOutputLine,
        CancellationToken cancellationToken,
        Action<int, DateTime?>? onStarted = null
    );

    /// <summary>Best-effort synchronous kill of the currently running process tree,
    /// if any (used to avoid orphaning a trainer when the app is closing).</summary>
    void TryKillCurrent();
}
