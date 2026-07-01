using System.Globalization;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class QualityHistoryEntry
{
    [JsonPropertyName("recorded_at")]
    public DateTimeOffset RecordedAt { get; init; }

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

    [JsonPropertyName("synthetic_pattern_count")]
    public int SyntheticPatternCount { get; init; }

    [JsonIgnore]
    public int IssueCount =>
        EmptyRowCount
        + DuplicateExactCount
        + DuplicateNormalizedCount
        + LowInformationCount
        + SyntheticPatternCount;

    [JsonIgnore]
    public string DisplayName =>
        $"{RecordedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture)} - Examples: {ExampleCount}, Issues: {IssueCount}";

    public static QualityHistoryEntry FromReport(QualityReport report)
    {
        return new QualityHistoryEntry
        {
            RecordedAt = DateTimeOffset.UtcNow,
            ExampleCount = report.ExampleCount,
            EmptyRowCount = report.EmptyRowCount,
            DuplicateExactCount = report.DuplicateExactCount,
            DuplicateNormalizedCount = report.DuplicateNormalizedCount,
            LowInformationCount = report.LowInformationCount,
            SyntheticPatternCount = report.SyntheticPatternCount,
        };
    }
}
