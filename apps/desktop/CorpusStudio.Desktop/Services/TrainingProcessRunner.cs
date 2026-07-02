using System.Diagnostics;
using System.IO;
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

    // Drain buffered pipe output within a bound so a surviving grandchild that
    // still holds the stdout/stderr handle cannot hang the run forever.
    private static readonly TimeSpan DrainTimeout = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan KillWaitTimeout = TimeSpan.FromSeconds(5);

    private volatile Process? _currentProcess;

    /// <summary>Run <paramref name="argv"/> to completion, invoking
    /// <paramref name="onOutputLine"/> for each stdout/stderr line (on a
    /// background thread — the caller marshals to the UI). Returns the exit code;
    /// throws <see cref="OperationCanceledException"/> after killing the process
    /// tree if cancelled.</summary>
    public async Task<int> RunAsync(
        IReadOnlyList<string> argv,
        string? workingDirectory,
        Action<string> onOutputLine,
        CancellationToken cancellationToken,
        Action<int>? onStarted = null
    )
    {
        if (argv is null || argv.Count == 0)
        {
            throw new ArgumentException("A command (argv) is required.", nameof(argv));
        }

        var startInfo = new ProcessStartInfo
        {
            // Resolve argv[0] to a full path found on PATH (not the working
            // directory) so a binary planted in the export dir can't be run
            // instead of the trusted trainer.
            FileName = ResolveExecutable(argv[0]),
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

        _currentProcess = process;
        try
        {
            try
            {
                onStarted?.Invoke(process.Id);
            }
            catch
            {
                // A recording callback must never break the run.
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

                // Let the tree actually terminate and flush its tail, bounded so a
                // lingering grandchild can't wedge cancellation.
                await WaitForExitBoundedAsync(process, KillWaitTimeout);
                await DrainBoundedAsync(outputComplete.Task, errorComplete.Task);
                throw;
            }

            // Normal exit: drain buffered output, but never block indefinitely if a
            // surviving grandchild still holds the pipe handle.
            await DrainBoundedAsync(outputComplete.Task, errorComplete.Task);
            return process.ExitCode;
        }
        finally
        {
            _currentProcess = null;
        }
    }

    /// <summary>Synchronously kill the currently running process tree, if any.
    /// Best-effort; used to avoid orphaning a trainer when the app is closing.</summary>
    public void TryKillCurrent()
    {
        var process = _currentProcess;
        if (process is null)
        {
            return;
        }

        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }
        }
        catch
        {
            // Race with normal exit/dispose — nothing to do.
        }
    }

    private static async Task DrainBoundedAsync(Task outputEof, Task errorEof)
    {
        var drain = Task.WhenAll(outputEof, errorEof);
        await Task.WhenAny(drain, Task.Delay(DrainTimeout));
    }

    private static async Task WaitForExitBoundedAsync(Process process, TimeSpan timeout)
    {
        try
        {
            await process.WaitForExitAsync(CancellationToken.None).WaitAsync(timeout);
        }
        catch
        {
            // Timed out or already exited; the kill has been issued regardless.
        }
    }

    private static string ResolveExecutable(string name)
    {
        // A caller-specified path (contains a separator) is used verbatim.
        if (name.Contains(Path.DirectorySeparatorChar) || name.Contains(Path.AltDirectorySeparatorChar))
        {
            return name;
        }

        var pathExt = (Environment.GetEnvironmentVariable("PATHEXT") ?? ".COM;.EXE;.BAT;.CMD")
            .Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var alreadyHasExtension = pathExt.Any(ext => name.EndsWith(ext, StringComparison.OrdinalIgnoreCase));
        var searchDirs = (Environment.GetEnvironmentVariable("PATH") ?? string.Empty)
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        foreach (var dir in searchDirs)
        {
            string basePath;
            try
            {
                basePath = Path.Combine(dir, name);
            }
            catch
            {
                continue; // Malformed PATH entry.
            }

            if (alreadyHasExtension)
            {
                if (File.Exists(basePath))
                {
                    return basePath;
                }

                continue;
            }

            foreach (var ext in pathExt)
            {
                var candidate = basePath + ext;
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        // Not found on PATH: keep the bare name so Process.Start surfaces a clear
        // "not installed / not on PATH" error rather than searching the CWD.
        return name;
    }
}
