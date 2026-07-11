using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class EvaluationExampleResult
{
    [JsonPropertyName("example_id")]
    public string ExampleId { get; init; } = string.Empty;

    [JsonPropertyName("prompt")]
    public string Prompt { get; init; } = string.Empty;

    [JsonPropertyName("expected_output")]
    public string ExpectedOutput { get; init; } = string.Empty;

    [JsonPropertyName("model_output")]
    public string ModelOutput { get; init; } = string.Empty;

    [JsonPropertyName("score")]
    public double Score { get; init; }

    [JsonPropertyName("passed")]
    public bool Passed { get; init; }

    [JsonPropertyName("tags")]
    public List<string> Tags { get; init; } = [];

    [JsonPropertyName("notes")]
    public string? Notes { get; init; }

    /// <summary>Set when the backend call for this example failed after retries;
    /// the example is recorded as a scored-0 failure instead of aborting the run.</summary>
    [JsonPropertyName("error")]
    public string? Error { get; init; }

    [JsonPropertyName("manual_score")]
    public double? ManualScore { get; set; }

    [JsonPropertyName("manual_notes")]
    public string? ManualNotes { get; set; }

    public string DisplayName
    {
        get
        {
            var status = string.IsNullOrEmpty(Error) ? (Passed ? "pass" : "fail") : "error";
            var manual = ManualScore is null ? "unscored" : $"manual {ManualScore:0.##}";
            return $"{ExampleId} | {status} | auto {Score:0.##} | {manual}";
        }
    }

    public string DetailText => string.Join(
        Environment.NewLine,
        [
            $"Example: {ExampleId}",
            $"Auto score: {Score:0.##}",
            $"Passed: {(Passed ? "yes" : "no")}",
            .. string.IsNullOrEmpty(Error) ? Array.Empty<string>() : new[] { $"Backend error: {Error}" },
            $"Tags: {(Tags.Count == 0 ? "none" : string.Join(", ", Tags))}",
            "",
            "Prompt:",
            Prompt,
            "",
            "Expected output:",
            ExpectedOutput,
            "",
            "Model output:",
            ModelOutput,
        ]
    );

    /// <summary>Status chip for the results list (slice 6): error &gt; fail &gt; pass.</summary>
    public string StatusLabel => !string.IsNullOrEmpty(Error) ? "ERROR" : Passed ? "PASS" : "FAIL";

    /// <summary>Nocturne hex for the status chip (error = warn amber, pass = ok green, fail = bad red).</summary>
    public string StatusColor => !string.IsNullOrEmpty(Error) ? "#d9a35f" : Passed ? "#6bbf9a" : "#d76d6d";

    /// <summary>Fixed pixel width of the per-example score-bar track (Evaluation screen: a compact inline
    /// bar shown before the numeric score). The filled inner bar is <see cref="ScoreBarWidth"/> of this.</summary>
    public const double ScoreBarTrackWidth = 130.0;

    /// <summary>Filled width (px) of the score bar — the score as a fraction of 100 across the fixed
    /// <see cref="ScoreBarTrackWidth"/> track. Clamped to the track so an out-of-range score can't overflow.</summary>
    public double ScoreBarWidth => Math.Clamp(Score, 0, 100) / 100.0 * ScoreBarTrackWidth;

    /// <summary>Nocturne fill for the score bar: Ok green when the example passed its run threshold, else
    /// Warn amber (a failing bar reads as caution — the red hex stays reserved for the status chip).</summary>
    public string ScoreBarColor => Passed ? "#6bbf9a" : "#d9a35f";
}
