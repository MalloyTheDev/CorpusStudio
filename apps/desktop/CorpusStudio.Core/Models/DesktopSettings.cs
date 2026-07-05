namespace CorpusStudio.Desktop.Models;

public sealed record DesktopSettings(
    string RepositoryRoot,
    string EngineDirectory,
    string PythonExecutable,
    string ProjectDirectory,
    string ExportDirectory
);
