using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Deserialized result of the engine `gate-run` command.</summary>
public sealed class GateReport
{
    [JsonPropertyName("scope")]
    public string Scope { get; init; } = string.Empty;

    [JsonPropertyName("target")]
    public string Target { get; init; } = string.Empty;

    [JsonPropertyName("overall_status")]
    public string OverallStatus { get; init; } = string.Empty;

    [JsonPropertyName("pass_count")]
    public int PassCount { get; init; }

    [JsonPropertyName("warn_count")]
    public int WarnCount { get; init; }

    [JsonPropertyName("block_count")]
    public int BlockCount { get; init; }

    [JsonPropertyName("results")]
    public IReadOnlyList<GateResult> Results { get; init; } = [];
}

public sealed class GateResult
{
    [JsonPropertyName("gate_id")]
    public string GateId { get; init; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; init; } = string.Empty;

    [JsonPropertyName("observed")]
    public string Observed { get; init; } = string.Empty;

    [JsonPropertyName("expected")]
    public string Expected { get; init; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; init; } = string.Empty;

    [JsonPropertyName("repair")]
    public string? Repair { get; init; }
}
