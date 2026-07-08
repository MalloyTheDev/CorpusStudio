using System;
using System.IO;
using System.Linq;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Guards the issue #208 hardening: the reveal-in-Explorer command must pass the path as an
/// escaped argument with UseShellExecute=false, and must refuse a missing/empty/malformed path rather
/// than launching Explorer against it.</summary>
public sealed class RevealInFileExplorerTests
{
    [Fact]
    public void BuildStartInfo_ReturnsNull_ForEmptyOrMissingPath()
    {
        Assert.Null(RevealInFileExplorer.BuildStartInfo(null));
        Assert.Null(RevealInFileExplorer.BuildStartInfo(""));
        Assert.Null(RevealInFileExplorer.BuildStartInfo("   "));

        var missing = Path.Combine(Path.GetTempPath(), "corpus-missing-" + Guid.NewGuid().ToString("N") + ".txt");
        Assert.Null(RevealInFileExplorer.BuildStartInfo(missing));
    }

    [Fact]
    public void BuildStartInfo_PassesPathAsEscapedArgument_NotShellParsed()
    {
        // A real, existing file whose name contains shell-significant (but path-legal) characters:
        // spaces, '&', ',', "'", '(' ')'. Under the old interpolated string these would be a quoting
        // hazard; here they must ride through as one literal argument.
        var dir = Path.Combine(Path.GetTempPath(), "corpus_reveal_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var file = Path.Combine(dir, "weird & name, (v2) 'x'.txt");
        File.WriteAllText(file, "x");
        try
        {
            var startInfo = RevealInFileExplorer.BuildStartInfo(file);

            Assert.NotNull(startInfo);
            Assert.False(startInfo!.UseShellExecute); // not shell-parsed
            Assert.Equal("explorer.exe", startInfo.FileName);
            // The path rides as a single discrete argument (with the /select, prefix), verbatim — the
            // caller never interpolates it into a shell string.
            Assert.Equal($"/select,{Path.GetFullPath(file)}", startInfo.ArgumentList.Single());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
