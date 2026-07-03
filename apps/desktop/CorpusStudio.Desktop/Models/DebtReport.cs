using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>A dataset debt report (mirrors the engine DebtReport). The engine owns all
/// computation; the desktop only parses, colors, and ranks for display.</summary>
public sealed class DebtReport
{
    [JsonPropertyName("example_count")]
    public int ExampleCount { get; set; }

    [JsonPropertyName("has_data")]
    public bool HasData { get; set; }

    [JsonPropertyName("grade")]
    public string Grade { get; set; } = "N/A";

    [JsonPropertyName("items")]
    public List<DebtItem> Items { get; set; } = [];

    /// <summary>Foreground hex for a debt grade. N/A (and any unknown, including the "—"
    /// unknown/stale state) is neutral GRAY — never green — so an unassessable or stale
    /// dataset can never read as a good grade. Pure/testable.</summary>
    public static string GradeColor(string grade)
    {
        return grade switch
        {
            "A" => "#16A34A",  // green
            "B" => "#65A30D",  // lime
            "C" => "#CA8A04",  // amber
            "D" => "#EA580C",  // orange
            "F" => "#DC2626",  // red
            _ => "#64748B",    // gray — N/A, "—", unknown: never green
        };
    }
}

public sealed class DebtItem
{
    [JsonPropertyName("category")]
    public string Category { get; set; } = string.Empty;

    [JsonPropertyName("severity")]
    public string Severity { get; set; } = "low";

    [JsonPropertyName("count")]
    public int Count { get; set; }

    [JsonPropertyName("rate")]
    public double? Rate { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("remediation")]
    public string Remediation { get; set; } = string.Empty;
}

/// <summary>A view row: a debt item with a severity badge, a normalized measure, and a
/// one-line display (highest-severity items are already ordered first by the engine).</summary>
public sealed class DebtDisplayItem
{
    public DebtDisplayItem(DebtItem item)
    {
        Item = item;
    }

    public DebtItem Item { get; }

    public string SeverityBadge => Item.Severity switch
    {
        "critical" => "⛔ CRITICAL",
        "high" => "⚠ HIGH",
        "moderate" => "△ MODERATE",
        _ => "· low",
    };

    /// <summary>Rate as a percentage where the engine gave one (rate is null for
    /// presence-based debts like secrets/PII and imbalance), else a raw count.</summary>
    public string Measure => Item.Rate.HasValue
        ? $"{Item.Rate.Value * 100:0.0}% ({Item.Count})"
        : $"count {Item.Count}";

    public string DisplayName =>
        $"{SeverityBadge}  {Item.Category} — {Item.Message} ({Measure})  ·  Fix: {Item.Remediation}";
}
