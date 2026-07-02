using System.Diagnostics;
using System.Text;
using System.Threading;

namespace CorpusStudio.Desktop.Services;

/// <summary>
/// Spawns an external trainer process from a structured argv (no shell) and
/// streams its stdout/stderr line by line. Cancellation kills the process tree.
/// The engine never runs this; it only produces the argv, and the user confirms
/// the command before it is spawned.
/// </summary>
public sealed class TrainingProcessRunner
{
    private static readonly Encoding Utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

    /// <summary>Run <paramref name="argv"/> to completion, invoking
    /// <paramref name="onOutputLine"/> for each stdout/stderr line (on a
    /// background thread — the caller marshals to the UI). Returns the exit code;
    /// throws <see cref="OperationCanceledException"/> after killing the process
    /// tree if cancelled.</summary>
    public async Task<int> RunAsync(
        IReadOnlyList<string> argv,
        string? workingDirectory,
        Action<string> onOutputLine,
        CancellationToken cancellationToken
    )
    {
        if (argv is null || argv.Count == 0)
        {
            throw new ArgumentException("A command (argv) is required.", nameof(argv));
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = argv[0],
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Utf8NoBom,
            StandardErrorEncoding = Utf8NoBom,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        for (var index = 1; index < argv.Count; index++)
        {
            startInfo.ArgumentList.Add(argv[index]);
        }

        if (!string.IsNullOrWhiteSpace(workingDirectory))
        {
            startInfo.WorkingDirectory = workingDirectory;
        }

        // Trainers are usually Python; force UTF-8 so non-ASCII logs don't corrupt.
        startInfo.Environment["PYTHONUTF8"] = "1";
        startInfo.Environment["PYTHONIOENCODING"] = "utf-8";

        using var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };

        var outputComplete = new TaskCompletionSource();
        var errorComplete = new TaskCompletionSource();
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is null)
            {
                outputComplete.TrySetResult();
            }
            else
            {
                onOutputLine(e.Data);
            }
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is null)
            {
                errorComplete.TrySetResult();
            }
            else
            {
                onOutputLine(e.Data);
            }
        };

        try
        {
            process.Start();
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to start '{argv[0]}'. Is it installed and on PATH? ({ex.Message})",
                ex
            );
        }

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        try
        {
            await process.WaitForExitAsync(cancellationToken);
        }
        catch (OperationCanceledException)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // The process may have exited between the check and the kill.
            }

            throw;
        }

        // Wait for the async readers to drain any buffered output before returning.
        await Task.WhenAll(outputComplete.Task, errorComplete.Task);
        return process.ExitCode;
    }
}
