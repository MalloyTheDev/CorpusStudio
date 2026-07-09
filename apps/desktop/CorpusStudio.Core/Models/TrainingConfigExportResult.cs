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

    [JsonPropertyName("training_output_dir")]
    public string TrainingOutputDirectory { get; init; } = string.Empty;

    [JsonPropertyName("vram_estimate")]
    public VramEstimate? VramEstimate { get; init; }

    [JsonPropertyName("lora_recommendation")]
    public LoraRecommendation? LoraRecommendation { get; init; }

    [JsonPropertyName("preflight")]
    public TrainingPreflightReport? Preflight { get; init; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}

/// <summary>Cheap fail-fast checks run at config-generation time (mirrors the engine's
/// <c>training/preflight.py</c>): trainer on PATH, config/data present, dataset size, truncation.
/// A pre-flight, not a guarantee — a green result means "nothing obviously wrong", not "this run
/// will succeed". <see cref="CanLaunch"/> is false only when a check blocks (missing config/data,
/// empty dataset).</summary>
public sealed class TrainingPreflightReport
{
    [JsonPropertyName("status")]
    public string Status { get; init; } = "pass"; // pass | warn | block

    [JsonPropertyName("can_launch")]
    public bool CanLaunch { get; init; } = true;

    [JsonPropertyName("trainer_command")]
    public string TrainerCommand { get; init; } = string.Empty;

    [JsonPropertyName("trainer_found")]
    public bool TrainerFound { get; init; }

    [JsonPropertyName("checks")]
    public List<TrainingPreflightCheck> Checks { get; init; } = [];
}

public sealed class TrainingPreflightCheck
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; init; } = string.Empty; // pass | warn | block

    [JsonPropertyName("message")]
    public string Message { get; init; } = string.Empty;
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

    [JsonPropertyName("argv")]
    public List<string> Argv { get; init; } = [];

    [JsonPropertyName("resume_argv")]
    public List<string> ResumeArgv { get; init; } = [];

    [JsonPropertyName("dependencies")]
    public List<string> Dependencies { get; init; } = [];

    [JsonPropertyName("notes")]
    public List<string> Notes { get; init; } = [];
}

public sealed class VramEstimate
{
    [JsonPropertyName("parameter_count_billions")]
    public double? ParameterCountBillions { get; init; }

    [JsonPropertyName("weights_gb_fp16")]
    public double? WeightsGbFp16 { get; init; }

    [JsonPropertyName("total_gb_fp16")]
    public double? TotalGbFp16 { get; init; }

    [JsonPropertyName("total_gb_int8")]
    public double? TotalGbInt8 { get; init; }

    [JsonPropertyName("total_gb_int4")]
    public double? TotalGbInt4 { get; init; }

    [JsonPropertyName("note")]
    public string Note { get; init; } = string.Empty;
}

public sealed class LoraRecommendation
{
    [JsonPropertyName("recommended_r")]
    public int RecommendedR { get; init; }

    [JsonPropertyName("recommended_alpha")]
    public int RecommendedAlpha { get; init; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
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
