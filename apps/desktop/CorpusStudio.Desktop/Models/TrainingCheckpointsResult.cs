using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Result of the engine `training-checkpoints` command.</summary>
public sealed class TrainingCheckpointsResult
{
    [JsonPropertyName("output_dir")]
    public string OutputDirectory { get; init; } = string.Empty;

    [JsonPropertyName("checkpoints")]
    public List<string> Checkpoints { get; init; } = [];

    [JsonPropertyName("latest_checkpoint")]
    public string? LatestCheckpoint { get; init; }

    [JsonPropertyName("resume_command")]
    public string? ResumeCommand { get; init; }

    [JsonPropertyName("resume_argv")]
    public List<string>? ResumeArgv { get; init; }

    [JsonPropertyName("resume_supported")]
    public bool? ResumeSupported { get; init; }
}
