using System.IO;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Covers the distributability guard (v1.2.15): the service never throws from its
/// constructor and can be re-pointed at an engine folder at runtime.</summary>
public sealed class PythonEngineServiceTests
{
    [Fact]
    public void Constructor_DoesNotThrow_AndReportsAvailability()
    {
        // The point of the change: constructing must never throw even if the engine is missing.
        var service = new PythonEngineService();
        // Availability is a bool either way; when unavailable there must be a reason.
        if (!service.IsEngineAvailable)
        {
            Assert.False(string.IsNullOrWhiteSpace(service.EngineUnavailableReason));
        }
    }

    [Fact]
    public void TryLocateEngine_RejectsNonEngineFolder()
    {
        var service = new PythonEngineService();
        var temp = Directory.CreateTempSubdirectory().FullName;
        try
        {
            Assert.False(service.TryLocateEngine(temp));
            Assert.False(service.TryLocateEngine(Path.Combine(temp, "does-not-exist")));
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public void TryLocateEngine_AcceptsFolderContainingEngineCli()
    {
        var service = new PythonEngineService();
        var temp = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var pkg = Path.Combine(temp, "corpus_studio");
            Directory.CreateDirectory(pkg);
            File.WriteAllText(Path.Combine(pkg, "cli.py"), "# stub");

            Assert.True(service.TryLocateEngine(temp));
            Assert.True(service.IsEngineAvailable);
            Assert.Null(service.EngineUnavailableReason);
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public void TryLocateEngine_AcceptsRepoRootWithEngineSubfolder()
    {
        var service = new PythonEngineService();
        var temp = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var pkg = Path.Combine(temp, "engine", "corpus_studio");
            Directory.CreateDirectory(pkg);
            File.WriteAllText(Path.Combine(pkg, "cli.py"), "# stub");

            Assert.True(service.TryLocateEngine(temp));
            Assert.True(service.IsEngineAvailable);
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }
}
