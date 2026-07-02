using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Durable training run record. The desktop owns the process, so it
/// writes these directly (no subprocess on the crash path); the engine owns the
/// schema and headless listing. Mutable because the launcher updates it across
/// the run lifecycle (prepared → running → terminal).</summary>
public sealed class TrainingRunRecord
{
    [JsonPropertyName("run_id")]
    public string RunId { get; set; } = string.Empty;

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = string.Empty;

    [JsonPropertyName("updated_at")]
    public string UpdatedAt { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = "prepared";

    [JsonPropertyName("target")]
    public string Target { get; set; } = string.Empty;

    [JsonPropertyName("base_model")]
    public string BaseModel { get; set; } = string.Empty;

    [JsonPropertyName("config_path")]
    public string ConfigPath { get; set; } = string.Empty;

    [JsonPropertyName("output_dir")]
    public string OutputDir { get; set; } = string.Empty;

    [JsonPropertyName("argv")]
    public List<string> Argv { get; set; } = [];

    [JsonPropertyName("pid")]
    public int? Pid { get; set; }

    [JsonPropertyName("exit_code")]
    public int? ExitCode { get; set; }

    [JsonPropertyName("checkpoints")]
    public List<string> Checkpoints { get; set; } = [];

    [JsonPropertyName("before_eval_path")]
    public string? BeforeEvalPath { get; set; }

    [JsonPropertyName("after_eval_path")]
    public string? AfterEvalPath { get; set; }

    [JsonPropertyName("after_eval_model")]
    public string? AfterEvalModel { get; set; }

    [JsonPropertyName("notes")]
    public string Notes { get; set; } = string.Empty;
}
