using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace CorpusStudio.Desktop.Models;

/// <summary>One bar in the dashboard/Quality "debt trend" mini-chart (v1.2.9): the quality
/// issue rate for a single recorded quality run. The bar height is normalized against the
/// tallest bar in the series so the chart is self-scaling. Pure/display-only — all values are
/// derived from the existing <see cref="QualityHistoryEntry"/> records; nothing new is
/// persisted.</summary>
public sealed class DebtTrendPoint
{
    public string Label { get; init; } = string.Empty;      // "MM-dd HH:mm"
    public int IssueCount { get; init; }
    public int ExampleCount { get; init; }
    public double IssueRatePer100 { get; init; }            // issues per 100 rows
    public double BarHeight { get; init; }                  // px, 0..MaxBarPx
    public bool IsLatest { get; init; }
    public string BarColor { get; init; } = "#BFDBFE";

    public string Tooltip =>
        $"{Label}: {IssueCount} issue{(IssueCount == 1 ? string.Empty : "s")} in "
        + $"{ExampleCount} row{(ExampleCount == 1 ? string.Empty : "s")} "
        + $"({IssueRatePer100.ToString("0.0", CultureInfo.InvariantCulture)}/100)";
}

/// <summary>The computed debt trend: the ordered bars (oldest → newest), a direction verdict,
/// and an honest one-line summary.</summary>
public sealed class DebtTrendResult
{
    public IReadOnlyList<DebtTrendPoint> Points { get; init; } = [];

    /// <summary>True when there are at least two runs — enough for a direction.</summary>
    public bool HasTrend { get; init; }

    public string Direction { get; init; } = string.Empty;      // "Improving ▼" / "Worsening ▲" / "Stable ·"
    public string DirectionColor { get; init; } = "#64748B";
    public string Summary { get; init; } = string.Empty;
}

/// <summary>Builds a debt trend from the recorded quality history. The A–F debt grade itself is
/// not trended: presence-based PII/secrets aren't in the history, so a reconstructed grade would
/// be a fabrication. Instead this trends the quality issue <em>rate</em> (issues ÷ rows), the
/// measurable, historical part of debt. Pure/testable — no clock, no I/O.</summary>
public static class DebtTrend
{
    public const double MaxBarPx = 56.0;

    private static double RatePer100(QualityHistoryEntry e) =>
        e.ExampleCount > 0 ? (double)e.IssueCount / e.ExampleCount * 100.0 : 0.0;

    public static DebtTrendResult Build(IReadOnlyList<QualityHistoryEntry>? history)
    {
        if (history is null || history.Count == 0)
        {
            return new DebtTrendResult { Summary = "Run quality checks to build a debt trend." };
        }

        // History arrives newest-first; a left-to-right chart wants oldest-first.
        var ordered = history.OrderBy(e => e.RecordedAt).ToList();
        var maxRate = ordered.Max(RatePer100);
        var latestIndex = ordered.Count - 1;

        var points = new List<DebtTrendPoint>(ordered.Count);
        for (var i = 0; i < ordered.Count; i++)
        {
            var e = ordered[i];
            var rate = RatePer100(e);
            var height = maxRate > 0 ? rate / maxRate * MaxBarPx : 0.0;
            // A small visible floor so a nonzero (but tiny) rate never renders as an empty gap,
            // and a zero rate still shows a baseline sliver.
            height = Math.Max(height, rate > 0 ? 3.0 : 1.0);
            var isLatest = i == latestIndex;

            points.Add(new DebtTrendPoint
            {
                Label = e.RecordedAt.LocalDateTime.ToString("MM-dd HH:mm", CultureInfo.InvariantCulture),
                IssueCount = e.IssueCount,
                ExampleCount = e.ExampleCount,
                IssueRatePer100 = rate,
                BarHeight = height,
                IsLatest = isLatest,
                BarColor = isLatest ? "#2563EB" : "#BFDBFE",
            });
        }

        if (ordered.Count < 2)
        {
            return new DebtTrendResult
            {
                Points = points,
                HasTrend = false,
                Summary = "One quality run recorded — run quality again to see the trend.",
            };
        }

        var firstRate = RatePer100(ordered[0]);
        var lastRate = RatePer100(ordered[latestIndex]);
        var delta = lastRate - firstRate;
        const double epsilon = 0.05;

        var (direction, color) = delta < -epsilon
            ? ("Improving ▼", "#16A34A")   // ▼ rate dropped
            : delta > epsilon
                ? ("Worsening ▲", "#DC2626")   // ▲ rate rose
                : ("Stable ·", "#64748B");

        var summary =
            $"Issue rate {firstRate.ToString("0.0", CultureInfo.InvariantCulture)} → "
            + $"{lastRate.ToString("0.0", CultureInfo.InvariantCulture)} per 100 rows across {ordered.Count} runs. "
            + "Presence-based PII/secrets are graded live in the Debt tab, not trended here.";

        return new DebtTrendResult
        {
            Points = points,
            HasTrend = true,
            Direction = direction,
            DirectionColor = color,
            Summary = summary,
        };
    }
}
