using System;
using System.Diagnostics;
using System.IO;

namespace CorpusStudio.Desktop.Services;

/// <summary>Builds the <see cref="ProcessStartInfo"/> to reveal-and-select a path in Windows File
/// Explorer, safely (issue #208). The path is carried as an escaped <see cref="ProcessStartInfo.ArgumentList"/>
/// entry — never interpolated into a shell-parsed argument string — with <c>UseShellExecute = false</c>, and
/// the target must exist on disk. That closes the argument-quoting / injection hole in the old
/// <c>explorer.exe /select,"{path}"</c> + <c>UseShellExecute = true</c> pattern, and also stops the app
/// launching Explorer against a path that was deleted since it was opened.</summary>
public static class RevealInFileExplorer
{
    /// <summary>Returns start info that selects <paramref name="fullPath"/> in Explorer, or <c>null</c>
    /// when the path is empty, malformed, or no longer exists (the caller should message instead).</summary>
    public static ProcessStartInfo? BuildStartInfo(string? fullPath)
    {
        if (string.IsNullOrWhiteSpace(fullPath))
        {
            return null;
        }

        string normalized;
        try
        {
            normalized = Path.GetFullPath(fullPath);
        }
        catch (Exception)
        {
            // Invalid characters, path too long, etc. — refuse rather than pass it through.
            return null;
        }

        if (!File.Exists(normalized) && !Directory.Exists(normalized))
        {
            return null;
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = "explorer.exe",
            UseShellExecute = false,
        };
        // One escaped argument: .NET quotes/escapes it per the Windows rules, so a path containing
        // spaces, '&', ',', quotes, etc. is passed literally and can't inject extra Explorer arguments.
        startInfo.ArgumentList.Add($"/select,{normalized}");
        return startInfo;
    }
}
