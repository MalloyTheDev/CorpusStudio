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

    [JsonPropertyName("manual_score")]
    public double? ManualScore { get; set; }

    [JsonPropertyName("manual_notes")]
    public string? ManualNotes { get; set; }

    public string DisplayName
    {
        get
        {
            var status = Passed ? "pass" : "fail";
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
}
