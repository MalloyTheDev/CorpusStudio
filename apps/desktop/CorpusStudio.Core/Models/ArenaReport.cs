using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Deserialized result of the engine `arena-run` command.</summary>
public sealed class ArenaReport
{
    [JsonPropertyName("prompt_count")]
    public int PromptCount { get; init; }

    [JsonPropertyName("models")]
    public IReadOnlyList<string> Models { get; init; } = [];

    [JsonPropertyName("prompts")]
    public IReadOnlyList<ArenaPromptItem> Prompts { get; init; } = [];

    [JsonPropertyName("responses")]
    public IReadOnlyList<ArenaResponse> Responses { get; init; } = [];

    [JsonPropertyName("model_summaries")]
    public IReadOnlyList<ArenaModelSummary> ModelSummaries { get; init; } = [];

    [JsonPropertyName("judge_model")]
    public string? JudgeModel { get; init; }

    [JsonPropertyName("judgments")]
    public IReadOnlyList<ArenaJudgment> Judgments { get; init; } = [];
}

public sealed class ArenaPromptItem
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("prompt")]
    public string Prompt { get; init; } = string.Empty;

    [JsonPropertyName("system")]
    public string? System { get; init; }
}

public sealed class ArenaResponse
{
    [JsonPropertyName("prompt_id")]
    public string PromptId { get; init; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("text")]
    public string Text { get; init; } = string.Empty;

    /// <summary>Set when this prompt/model call failed after retries; the batch
    /// records the failure instead of aborting.</summary>
    [JsonPropertyName("error")]
    public string? Error { get; init; }
}

public sealed class ArenaModelSummary
{
    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("response_count")]
    public int ResponseCount { get; init; }

    [JsonPropertyName("empty_response_count")]
    public int EmptyResponseCount { get; init; }

    /// <summary>Responses that failed with a backend error (distinct from a model
    /// that legitimately returned empty text).</summary>
    [JsonPropertyName("error_count")]
    public int ErrorCount { get; init; }

    [JsonPropertyName("win_count")]
    public int WinCount { get; init; }

    [JsonPropertyName("average_judge_score")]
    public double? AverageJudgeScore { get; init; }
}

public sealed class ArenaJudgment
{
    [JsonPropertyName("prompt_id")]
    public string PromptId { get; init; } = string.Empty;

    [JsonPropertyName("winner")]
    public string Winner { get; init; } = string.Empty;

    [JsonPropertyName("scores")]
    public Dictionary<string, double> Scores { get; init; } = new();

    [JsonPropertyName("rationale")]
    public string Rationale { get; init; } = string.Empty;

    [JsonPropertyName("parsed")]
    public bool Parsed { get; init; }
}
