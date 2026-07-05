using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>
/// A named, persisted Evaluation drilldown: a saved combination of the status,
/// tag, failure-reason, and score-band filters so a reviewer can reapply the
/// same failure slice across runs.
/// </summary>
public sealed class EvaluationFailureFilter
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; init; } = "All";

    [JsonPropertyName("tag")]
    public string Tag { get; init; } = "All";

    [JsonPropertyName("failure_reason")]
    public string FailureReason { get; init; } = "All";

    [JsonPropertyName("score_band")]
    public string ScoreBand { get; init; } = "All";

    [JsonIgnore]
    public string DisplayName
    {
        get
        {
            var parts = new List<string>();
            if (!IsAll(Status)) parts.Add(Status.ToLowerInvariant());
            if (!IsAll(Tag)) parts.Add($"tag:{Tag}");
            if (!IsAll(FailureReason)) parts.Add($"reason:{FailureReason}");
            if (!IsAll(ScoreBand)) parts.Add($"band:{ScoreBand}");
            var scope = parts.Count == 0 ? "all results" : string.Join(", ", parts);
            return $"{Name} ({scope})";
        }
    }

    private static bool IsAll(string value) =>
        string.IsNullOrWhiteSpace(value)
        || string.Equals(value, "All", StringComparison.OrdinalIgnoreCase);
}
