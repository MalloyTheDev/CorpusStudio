using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class TrainingConfigExportResult
{
    [JsonPropertyName("target")]
    public string Target { get; init; } = string.Empty;

    [JsonPropertyName("output_path")]
    public string OutputPath { get; init; } = string.Empty;

    [JsonPropertyName("training_launcher_implemented")]
    public bool TrainingLauncherImplemented { get; init; }

    [JsonPropertyName("config_text")]
    public string ConfigText { get; init; } = string.Empty;

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}
