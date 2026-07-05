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

    [JsonPropertyName("pii_finding_count")]
    public int PiiFindingCount { get; init; }

    [JsonPropertyName("pii_findings")]
    public IReadOnlyList<PiiFinding> PiiFindings { get; init; } = [];

    [JsonPropertyName("token_length_threshold")]
    public int TokenLengthThreshold { get; init; }

    [JsonPropertyName("token_length_outlier_count")]
    public int TokenLengthOutlierCount { get; init; }

    [JsonPropertyName("token_length_outliers")]
    public IReadOnlyList<TokenLengthOutlier> TokenLengthOutliers { get; init; } = [];

    [JsonPropertyName("category_imbalances")]
    public IReadOnlyList<CategoryImbalance> CategoryImbalances { get; init; } = [];
}

public sealed class TokenLengthOutlier
{
    [JsonPropertyName("row_number")]
    public int RowNumber { get; init; }

    [JsonPropertyName("token_count")]
    public int TokenCount { get; init; }
}

public sealed class CategoryImbalance
{
    [JsonPropertyName("field")]
    public string Field { get; init; } = string.Empty;

    [JsonPropertyName("dominant_value")]
    public string DominantValue { get; init; } = string.Empty;

    [JsonPropertyName("dominant_count")]
    public int DominantCount { get; init; }

    [JsonPropertyName("total")]
    public int Total { get; init; }

    [JsonPropertyName("share")]
    public double Share { get; init; }

    [JsonPropertyName("distinct_values")]
    public int DistinctValues { get; init; }

    public string DisplayName =>
        $"'{Field}' = '{DominantValue}' in {DominantCount}/{Total} rows ({Share:P0}, {DistinctValues} distinct)";
}

public sealed class PiiFinding
{
    [JsonPropertyName("kind")]
    public string Kind { get; init; } = string.Empty;

    [JsonPropertyName("severity")]
    public string Severity { get; init; } = string.Empty;

    [JsonPropertyName("match_count")]
    public int MatchCount { get; init; }

    [JsonPropertyName("row_numbers")]
    public IReadOnlyList<int> RowNumbers { get; init; } = [];

    [JsonPropertyName("sample")]
    public string Sample { get; init; } = string.Empty;

    [JsonPropertyName("suggestion")]
    public string Suggestion { get; init; } = string.Empty;

    public string DisplayName
    {
        get
        {
            var severity = string.IsNullOrWhiteSpace(Severity) ? "unknown" : Severity;
            var kind = string.IsNullOrWhiteSpace(Kind) ? "pii" : Kind;
            var rows = RowNumbers.Count == 0
                ? "rows unknown"
                : $"row(s) {string.Join(", ", RowNumbers.Take(4))}";
            return $"[{severity}] {kind} ×{MatchCount} | {rows}";
        }
    }
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
