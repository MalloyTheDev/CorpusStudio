using System.Diagnostics;

namespace CorpusStudio.Desktop.Services;

public sealed class PythonEngineService
{
    public async Task<string> ValidateAsync(string engineDirectory, string datasetPath, string schemaId)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = $"-m corpus_studio.cli validate \"{datasetPath}\" {schemaId}",
            WorkingDirectory = engineDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start Python engine process.");

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(error);
        }

        return output;
    }
}
