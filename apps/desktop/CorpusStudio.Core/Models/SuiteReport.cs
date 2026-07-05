using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>One row in `suite-list`: a registered evaluation suite (the filename stem is
/// the key). A malformed registry file comes back with Valid=false + an Error.</summary>
public sealed class SuiteSummary
{
    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("case_count")]
    public int CaseCount { get; init; }

    [JsonPropertyName("valid")]
    public bool Valid { get; init; } = true;

    [JsonPropertyName("error")]
    public string? Error { get; init; }

    /// <summary>List label, e.g. "release-gate — 2 case(s)" or "broken — invalid".</summary>
    public string DisplayLabel =>
        Valid ? $"{Name} — {CaseCount} case(s)" : $"{Name} — invalid";
}

/// <summary>Per-metric roll-up in a SuiteReport — the suite verdict is never a single
/// number folded across metrics.</summary>
public sealed class SuiteMetricRollup
{
    [JsonPropertyName("metric")]
    public string Metric { get; init; } = string.Empty;

    [JsonPropertyName("total")]
    public int Total { get; init; }

    [JsonPropertyName("passed")]
    public int Passed { get; init; }

    [JsonPropertyName("warned")]
    public int Warned { get; init; }

    [JsonPropertyName("blocked")]
    public int Blocked { get; init; }

    [JsonPropertyName("errored")]
    public int Errored { get; init; }

    public string DisplayLabel =>
        $"{Metric}: {Passed}/{Total} pass"
        + (Blocked + Errored > 0 ? $" ({Blocked} block, {Errored} error)" : string.Empty);
}

/// <summary>One case's outcome inside a SuiteReport.</summary>
public sealed class SuiteCaseResult
{
    [JsonPropertyName("case")]
    public string Case { get; init; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("metric")]
    public string Metric { get; init; } = string.Empty;

    [JsonPropertyName("dataset_fingerprint")]
    public string? DatasetFingerprint { get; init; }

    [JsonPropertyName("examples_tested")]
    public int? ExamplesTested { get; init; }

    [JsonPropertyName("average_score")]
    public double? AverageScore { get; init; }

    [JsonPropertyName("pass_rate")]
    public double? PassRate { get; init; }

    [JsonPropertyName("gate")]
    public GateReport? Gate { get; init; }

    [JsonPropertyName("error")]
    public string? Error { get; init; }

    [JsonPropertyName("status")]
    public string Status { get; init; } = string.Empty;

    public string StatusColor => SuiteReport.ColorForStatus(Status);

    /// <summary>Right-hand detail: the score for a scored case, or the error message.</summary>
    public string Detail => Error is { Length: > 0 }
        ? Error
        : AverageScore is { } score ? $"score {score:0.0}" : string.Empty;

    /// <summary>One-line case summary for the list (avoids read-only Run.Text bindings).</summary>
    public string CaseLine => $"{Case}   {Model} · {Metric}   {Detail}".TrimEnd();
}

/// <summary>Deserialized result of the engine `suite-run` command.</summary>
public sealed class SuiteReport
{
    [JsonPropertyName("suite")]
    public string Suite { get; init; } = string.Empty;

    [JsonPropertyName("generated_at")]
    public string? GeneratedAt { get; init; }

    [JsonPropertyName("cases")]
    public IReadOnlyList<SuiteCaseResult> Cases { get; init; } = [];

    [JsonPropertyName("per_metric")]
    public IReadOnlyList<SuiteMetricRollup> PerMetric { get; init; } = [];

    [JsonPropertyName("overall_status")]
    public string OverallStatus { get; init; } = string.Empty;

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = string.Empty;

    /// <summary>Status → hex color. Like GateReport.StatusColor, but an errored case is
    /// red (not neutral); unknown/empty stays gray so it never reads as a fake pass.</summary>
    public static string ColorForStatus(string? status)
    {
        return (status ?? string.Empty).ToLowerInvariant() switch
        {
            "pass" => "#16A34A",   // green
            "warn" => "#D97706",   // amber
            "block" => "#DC2626",  // red
            "error" => "#DC2626",  // red — an errored case is not a pass
            _ => "#64748B",        // gray — null/empty/unknown
        };
    }
}
