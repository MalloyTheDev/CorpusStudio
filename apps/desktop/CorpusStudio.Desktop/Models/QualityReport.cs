using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class QualityReport
{
    [JsonPropertyName("example_count")]
    public int ExampleCount { get; init; }

    [JsonPropertyName("empty_row_count")]
    public int EmptyRowCount { get; init; }

    [JsonPropertyName("duplicate_exact_count")]
    public int DuplicateExactCount { get; init; }

    [JsonPropertyName("duplicate_normalized_count")]
    public int DuplicateNormalizedCount { get; init; }

    [JsonPropertyName("low_information_count")]
    public int LowInformationCount { get; init; }

    [JsonPropertyName("low_information_token_threshold")]
    public int LowInformationTokenThreshold { get; init; }

    [JsonPropertyName("synthetic_pattern_count")]
    public int SyntheticPatternCount { get; init; }

    [JsonPropertyName("synthetic_pattern_warnings")]
    public IReadOnlyList<string> SyntheticPatternWarnings { get; init; } = [];

    [JsonPropertyName("synthetic_pattern_issues")]
    public IReadOnlyList<SyntheticPatternIssue> SyntheticPatternIssues { get; init; } = [];

    [JsonPropertyName("synthetic_pattern_clusters")]
    public IReadOnlyList<SyntheticPatternCluster> SyntheticPatternClusters { get; init; } = [];
}

public sealed class SyntheticPatternCluster
{
    [JsonPropertyName("kind")]
    public string Kind { get; init; } = string.Empty;

    [JsonPropertyName("label")]
    public string Label { get; init; } = string.Empty;

    [JsonPropertyName("severity")]
    public string Severity { get; init; } = string.Empty;

    [JsonPropertyName("member_count")]
    public int MemberCount { get; init; }

    [JsonPropertyName("row_numbers")]
    public IReadOnlyList<int> RowNumbers { get; init; } = [];

    [JsonPropertyName("suggestion")]
    public string Suggestion { get; init; } = string.Empty;

    public string DisplayName
    {
        get
        {
            var severity = string.IsNullOrWhiteSpace(Severity) ? "unknown" : Severity;
            var kind = string.IsNullOrWhiteSpace(Kind) ? "synthetic_pattern" : Kind;
            var label = string.IsNullOrWhiteSpace(Label) ? "(no sample)" : Label;
            return $"[{severity}] {kind} ×{MemberCount} | {label}";
        }
    }
}

public sealed class SyntheticPatternIssue
{
    [JsonPropertyName("kind")]
    public string Kind { get; init; } = string.Empty;

    [JsonPropertyName("severity")]
    public string Severity { get; init; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; init; } = string.Empty;

    [JsonPropertyName("row_numbers")]
    public IReadOnlyList<int> RowNumbers { get; init; } = [];

    [JsonPropertyName("suggestion")]
    public string Suggestion { get; init; } = string.Empty;

    public string DisplayName
    {
        get
        {
            var severity = string.IsNullOrWhiteSpace(Severity) ? "unknown" : Severity;
            var kind = string.IsNullOrWhiteSpace(Kind) ? "synthetic_pattern" : Kind;
            var rows = RowNumbers.Count == 0
                ? "rows unknown"
                : $"row(s) {string.Join(", ", RowNumbers.Take(4))}";
            return $"[{severity}] {kind} | {rows}";
        }
    }
}
