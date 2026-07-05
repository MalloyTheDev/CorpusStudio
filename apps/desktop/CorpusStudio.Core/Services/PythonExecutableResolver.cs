using System;
using System.IO;

namespace CorpusStudio.Desktop.Services;

/// <summary>Locates the Python interpreter for the engine venv.
///
/// A venv created on Windows keeps the interpreter at <c>.venv/Scripts/python.exe</c>; on
/// macOS/Linux it is <c>.venv/bin/python</c>. The pure overload takes the platform and a
/// file-exists probe so cross-platform resolution is unit-testable without a real venv or OS —
/// a step toward the cross-platform port (see docs/AVALONIA_MIGRATION_PLAN.md, Phase 0).</summary>
public static class PythonExecutableResolver
{
    /// <summary>Resolve the engine interpreter path. Prefers the current platform's venv layout,
    /// accepts the other too (a repo synced across OSes can carry either shape), and falls back to
    /// <c>python</c> on PATH when no venv is present.</summary>
    public static string Resolve(string engineDirectory, bool isWindows, Func<string, bool> fileExists)
    {
        var windowsVenv = Path.Combine(engineDirectory, ".venv", "Scripts", "python.exe");
        var posixVenv = Path.Combine(engineDirectory, ".venv", "bin", "python");
        var candidates = isWindows
            ? new[] { windowsVenv, posixVenv }
            : new[] { posixVenv, windowsVenv };

        foreach (var candidate in candidates)
        {
            if (fileExists(candidate))
            {
                return candidate;
            }
        }

        return "python";
    }

    /// <summary>Resolve against the real OS + filesystem.</summary>
    public static string Resolve(string engineDirectory) =>
        Resolve(engineDirectory, OperatingSystem.IsWindows(), File.Exists);
}
