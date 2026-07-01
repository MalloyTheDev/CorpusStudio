using System.IO;

namespace CorpusStudio.Desktop.Tests;

/// <summary>
/// Creates a unique temporary project directory for a test and deletes it on
/// dispose. The persistence methods under test write project-local JSON files,
/// so each test gets an isolated directory.
/// </summary>
public sealed class TempProjectDirectory : IDisposable
{
    public string Path { get; }

    public TempProjectDirectory()
    {
        Path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            "corpusstudio-tests",
            Guid.NewGuid().ToString("N")
        );
        Directory.CreateDirectory(Path);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
        catch (IOException)
        {
            // Best-effort cleanup; a leaked temp dir must never fail a test.
        }
    }
}
