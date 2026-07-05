using System.IO;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Cross-platform venv interpreter resolution (Phase 0 of the Avalonia migration): the old
/// code only checked the Windows venv path, so a macOS/Linux venv was silently bypassed.</summary>
public sealed class PythonExecutableResolverTests
{
    private static string WinVenv(string dir) => Path.Combine(dir, ".venv", "Scripts", "python.exe");
    private static string PosixVenv(string dir) => Path.Combine(dir, ".venv", "bin", "python");

    [Fact]
    public void Posix_ResolvesBinPython()
    {
        var dir = "engine";
        var posix = PosixVenv(dir);
        var result = PythonExecutableResolver.Resolve(dir, isWindows: false, fileExists: p => p == posix);
        Assert.Equal(posix, result);
    }

    [Fact]
    public void Windows_ResolvesScriptsPythonExe()
    {
        var dir = "engine";
        var win = WinVenv(dir);
        var result = PythonExecutableResolver.Resolve(dir, isWindows: true, fileExists: p => p == win);
        Assert.Equal(win, result);
    }

    [Fact]
    public void FallsBackToPath_WhenNoVenvPresent()
    {
        var result = PythonExecutableResolver.Resolve("engine", isWindows: false, fileExists: _ => false);
        Assert.Equal("python", result); // bare PATH lookup, not a bogus venv path
    }

    [Fact]
    public void AcceptsTheOtherPlatformsVenvShape()
    {
        // On POSIX but the repo carries a Windows-shaped venv (or vice versa): still found, so a
        // synced-across-OSes checkout isn't silently bypassed for a system python.
        var dir = "engine";
        var win = WinVenv(dir);
        var result = PythonExecutableResolver.Resolve(dir, isWindows: false, fileExists: p => p == win);
        Assert.Equal(win, result);
    }
}
