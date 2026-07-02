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

    [JsonPropertyName("token_budget")]
    public TokenBudgetEstimate? TokenBudget { get; init; }

    [JsonPropertyName("launch")]
    public TrainingLaunchPlan? Launch { get; init; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}

public sealed class TrainingLaunchPlan
{
    [JsonPropertyName("target")]
    public string Target { get; init; } = string.Empty;

    [JsonPropertyName("command")]
    public string Command { get; init; } = string.Empty;

    [JsonPropertyName("resume_command")]
    public string ResumeCommand { get; init; } = string.Empty;

    [JsonPropertyName("resume_supported")]
    public bool ResumeSupported { get; init; }

    [JsonPropertyName("dependencies")]
    public List<string> Dependencies { get; init; } = [];

    [JsonPropertyName("notes")]
    public List<string> Notes { get; init; } = [];
}

public sealed class TokenBudgetEstimate
{
    [JsonPropertyName("example_count")]
    public int ExampleCount { get; init; }

    [JsonPropertyName("estimated_tokens")]
    public long EstimatedTokens { get; init; }

    [JsonPropertyName("method")]
    public string Method { get; init; } = string.Empty;

    [JsonPropertyName("sequence_len")]
    public int SequenceLen { get; init; }

    [JsonPropertyName("mean_tokens_per_example")]
    public double MeanTokensPerExample { get; init; }

    [JsonPropertyName("max_tokens_in_example")]
    public int MaxTokensInExample { get; init; }

    [JsonPropertyName("examples_over_sequence_len")]
    public int ExamplesOverSequenceLen { get; init; }

    [JsonPropertyName("tokens_per_epoch")]
    public long TokensPerEpoch { get; init; }
}
