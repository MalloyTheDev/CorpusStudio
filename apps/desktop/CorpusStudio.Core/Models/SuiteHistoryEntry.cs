using System.Globalization;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>One point in a suite's run history (issue #190), mirroring the engine's
/// <c>SuiteHistoryEntry</c>: the run time, the aggregate verdict, and per-status case counts. The counts
/// are a sum of case outcomes across metrics — a count, never a folded quality score.</summary>
public sealed class SuiteHistoryEntry
{
    [JsonPropertyName("generated_at")]
    public string? GeneratedAt { get; init; }

    [JsonPropertyName("overall_status")]
    public string OverallStatus { get; init; } = string.Empty;

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

    [JsonPropertyName("summary")]
    public string Summary { get; init; } = string.Empty;

    /// <summary>Verdict color for the trend row (reuses the suite report's status→color map).</summary>
    [JsonIgnore]
    public string StatusColor => SuiteReport.ColorForStatus(OverallStatus);

    /// <summary>Pass rate (0–1) for this run, used by the trend sparkline. 0 when the run had no cases.</summary>
    [JsonIgnore]
    public double PassRate => Total > 0 ? (double)Passed / Total : 0;

    /// <summary>Sparkline bar height in px: pass rate scaled into 4–36 px (a floor so a 0% run is
    /// still a visible tick). Colour comes from <see cref="StatusColor"/>.</summary>
    [JsonIgnore]
    public double SparkBarHeight => 4 + PassRate * 32;

    [JsonIgnore]
    public string DisplayLine =>
        $"{GeneratedAt ?? "?"} · {OverallStatus.ToUpper(CultureInfo.InvariantCulture)} · {Passed}/{Total} passed"
        + (Blocked > 0 ? $", {Blocked} blocked" : string.Empty)
        + (Errored > 0 ? $", {Errored} errored" : string.Empty);
}
