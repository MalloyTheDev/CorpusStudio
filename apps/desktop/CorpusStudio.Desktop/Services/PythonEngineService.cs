using System.Diagnostics;
using System.IO;
using System.Text.Json;

using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

public sealed class PythonEngineService
{
    private sealed record EngineProcessResult(int ExitCode, string Output, string Error);

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private readonly string _repositoryRoot;
    private readonly string _engineDirectory;
    private readonly string _pythonExecutable;
    private readonly IReadOnlyDictionary<string, string> _localEnvironment;

    public PythonEngineService()
    {
        _repositoryRoot = FindRepositoryRoot();
        _localEnvironment = LoadLocalEnvironment(_repositoryRoot);
        _engineDirectory = ResolveEngineDirectory(_repositoryRoot, _localEnvironment);
        _pythonExecutable = ResolvePythonExecutable(_engineDirectory);
    }

    public async Task<IReadOnlyList<DatasetSchema>> GetSchemasAsync()
    {
        var output = await RunEngineCommandAsync("schemas");
        return JsonSerializer.Deserialize<List<DatasetSchema>>(output, JsonOptions) ?? [];
    }

    public Task<string> CreateProjectAsync(string projectId, string name, string schemaId)
    {
        return RunEngineCommandAsync("new-project", projectId, name, schemaId);
    }

    public IReadOnlyList<DatasetProject> LoadProjects()
    {
        var projectRoot = ResolveProjectRoot();
        if (!Directory.Exists(projectRoot))
        {
            return [];
        }

        var projects = new List<DatasetProject>();
        foreach (var projectFile in Directory.EnumerateFiles(projectRoot, "project.json", SearchOption.AllDirectories))
        {
            try
            {
                var json = File.ReadAllText(projectFile);
                var project = JsonSerializer.Deserialize<DatasetProject>(json, JsonOptions);
                if (project is not null)
                {
                    projects.Add(project);
                }
            }
            catch (JsonException)
            {
                continue;
            }
        }

        return projects
            .OrderBy(project => project.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public DesktopSettings GetSettings()
    {
        return new DesktopSettings(
            _repositoryRoot,
            _engineDirectory,
            _pythonExecutable,
            ResolveProjectRoot(),
            ResolveExportRoot()
        );
    }

    public Task<string> ValidateAsync(string datasetPath, string schemaId)
    {
        return RunEngineCommandAsync("validate", datasetPath, schemaId);
    }

    public async Task<ValidationReport> ValidateDraftAsync(string draftText, string schemaId)
    {
        var tempPath = WriteDraftToTempJsonl(draftText);

        try
        {
            var result = await RunEngineProcessAsync(_engineDirectory, "validate", tempPath, schemaId);
            var payload = string.IsNullOrWhiteSpace(result.Output) ? result.Error : result.Output;

            if (string.IsNullOrWhiteSpace(payload))
            {
                throw new InvalidOperationException("The Python engine returned an empty validation report.");
            }

            return JsonSerializer.Deserialize<ValidationReport>(payload, JsonOptions)
                ?? throw new InvalidOperationException("The Python engine returned an invalid validation report.");
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    public int AppendDraftToProjectExamples(string projectPath, string draftText)
    {
        var jsonl = NormalizeDraftToJsonl(draftText);
        var rowCount = jsonl.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries).Length;

        if (rowCount == 0)
        {
            return 0;
        }

        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        Directory.CreateDirectory(projectPath);
        File.AppendAllText(examplesPath, jsonl, encoding: System.Text.Encoding.UTF8);
        return rowCount;
    }

    public async Task<string> ExportProjectExamplesAsync(string projectPath, string schemaId)
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var outputPath = Path.Combine(ResolveExportRoot(), projectId, "export.jsonl");
        await RunEngineCommandAsync("export", examplesPath, outputPath, schemaId);
        return outputPath;
    }

    public async Task<string> ValidateAsync(string engineDirectory, string datasetPath, string schemaId)
    {
        return await RunEngineCommandInDirectoryAsync(engineDirectory, "validate", datasetPath, schemaId);
    }

    private Task<string> RunEngineCommandAsync(params string[] arguments)
    {
        return RunEngineCommandInDirectoryAsync(_engineDirectory, arguments);
    }

    private async Task<string> RunEngineCommandInDirectoryAsync(
        string engineDirectory,
        params string[] arguments
    )
    {
        var result = await RunEngineProcessAsync(engineDirectory, arguments);

        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException(
                string.IsNullOrWhiteSpace(result.Error) ? result.Output : result.Error
            );
        }

        return result.Output;
    }

    private async Task<EngineProcessResult> RunEngineProcessAsync(
        string engineDirectory,
        params string[] arguments
    )
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = _pythonExecutable,
            WorkingDirectory = engineDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        startInfo.ArgumentList.Add("-m");
        startInfo.ArgumentList.Add("corpus_studio.cli");

        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        foreach (var (key, value) in _localEnvironment)
        {
            startInfo.Environment[key] = value;
        }

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start Python engine process.");

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        return new EngineProcessResult(process.ExitCode, output, error);
    }

    private static string WriteDraftToTempJsonl(string draftText)
    {
        var directory = Path.Combine(Path.GetTempPath(), "CorpusStudio");
        Directory.CreateDirectory(directory);

        var path = Path.Combine(directory, $"{Guid.NewGuid():N}.jsonl");
        File.WriteAllText(path, NormalizeDraftToJsonl(draftText), encoding: System.Text.Encoding.UTF8);
        return path;
    }

    private static string NormalizeDraftToJsonl(string draftText)
    {
        var trimmed = draftText.Trim();
        if (trimmed.Length == 0)
        {
            return string.Empty;
        }

        try
        {
            using var document = JsonDocument.Parse(trimmed);
            if (document.RootElement.ValueKind == JsonValueKind.Array)
            {
                var rows = document.RootElement.EnumerateArray()
                    .Select(element => JsonSerializer.Serialize(element));
                return string.Join(Environment.NewLine, rows) + Environment.NewLine;
            }

            return JsonSerializer.Serialize(document.RootElement) + Environment.NewLine;
        }
        catch (JsonException)
        {
            return draftText.EndsWith(Environment.NewLine, StringComparison.Ordinal)
                ? draftText
                : draftText + Environment.NewLine;
        }
    }

    private static string FindRepositoryRoot()
    {
        foreach (var startPath in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            var root = FindRepositoryRootFrom(startPath);
            if (root is not null)
            {
                return root;
            }
        }

        throw new DirectoryNotFoundException("Could not find the Corpus Studio repository root.");
    }

    private static string? FindRepositoryRootFrom(string startPath)
    {
        var directory = new DirectoryInfo(startPath);
        while (directory is not null)
        {
            var engineCli = Path.Combine(directory.FullName, "engine", "corpus_studio", "cli.py");
            if (File.Exists(engineCli))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static string ResolveEngineDirectory(
        string repositoryRoot,
        IReadOnlyDictionary<string, string> localEnvironment
    )
    {
        var configured = Environment.GetEnvironmentVariable("CORPUS_STUDIO_ENGINE_DIR");
        if (string.IsNullOrWhiteSpace(configured))
        {
            localEnvironment.TryGetValue("CORPUS_STUDIO_ENGINE_DIR", out configured);
        }

        var engineDirectory = string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(repositoryRoot, "engine")
            : ResolvePath(repositoryRoot, configured);

        if (!Directory.Exists(engineDirectory))
        {
            throw new DirectoryNotFoundException($"Python engine directory not found: {engineDirectory}");
        }

        return engineDirectory;
    }

    private static string ResolvePythonExecutable(string engineDirectory)
    {
        var venvPython = Path.Combine(engineDirectory, ".venv", "Scripts", "python.exe");
        return File.Exists(venvPython) ? venvPython : "python";
    }

    private string ResolveProjectRoot()
    {
        var configured = Environment.GetEnvironmentVariable("CORPUS_STUDIO_DATA_DIR");
        if (string.IsNullOrWhiteSpace(configured))
        {
            _localEnvironment.TryGetValue("CORPUS_STUDIO_DATA_DIR", out configured);
        }

        return string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(_repositoryRoot, "data", "projects")
            : ResolvePath(_repositoryRoot, configured);
    }

    private string ResolveExportRoot()
    {
        var configured = Environment.GetEnvironmentVariable("CORPUS_STUDIO_EXPORT_DIR");
        if (string.IsNullOrWhiteSpace(configured))
        {
            _localEnvironment.TryGetValue("CORPUS_STUDIO_EXPORT_DIR", out configured);
        }

        return string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(_repositoryRoot, "exports")
            : ResolvePath(_repositoryRoot, configured);
    }

    private static IReadOnlyDictionary<string, string> LoadLocalEnvironment(string repositoryRoot)
    {
        var path = Path.Combine(repositoryRoot, ".env");
        if (!File.Exists(path))
        {
            return new Dictionary<string, string>();
        }

        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var rawLine in File.ReadLines(path))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }

            var separatorIndex = line.IndexOf('=');
            if (separatorIndex <= 0)
            {
                continue;
            }

            var key = line[..separatorIndex].Trim();
            var value = line[(separatorIndex + 1)..].Trim().Trim('"');
            values[key] = value;
        }

        return values;
    }

    private static string ResolvePath(string repositoryRoot, string path)
    {
        return Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(repositoryRoot, path));
    }
}
