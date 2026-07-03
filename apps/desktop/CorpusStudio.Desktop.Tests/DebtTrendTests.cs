using System;
using System.Collections.Generic;
using CorpusStudio.Desktop.Models;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class DebtTrendTests
{
    // Only EmptyRowCount is set, so IssueCount == issues.
    private static QualityHistoryEntry E(int minutesAgo, int examples, int issues) => new()
    {
        RecordedAt = new DateTimeOffset(2026, 7, 3, 12, 0, 0, TimeSpan.Zero).AddMinutes(-minutesAgo),
        ExampleCount = examples,
        EmptyRowCount = issues,
    };

    [Fact]
    public void Build_Empty_HasNoTrendOrPoints()
    {
        var result = DebtTrend.Build(new List<QualityHistoryEntry>());
        Assert.False(result.HasTrend);
        Assert.Empty(result.Points);
        Assert.Contains("Run quality checks", result.Summary);
    }

    [Fact]
    public void Build_SingleRun_OneBarNoDirection()
    {
        var result = DebtTrend.Build(new[] { E(0, 100, 5) });
        Assert.Single(result.Points);
        Assert.False(result.HasTrend);
        Assert.Contains("One quality run", result.Summary);
        Assert.True(result.Points[0].IsLatest);
    }

    [Fact]
    public void Build_OrdersOldestToNewest_AndFlagsLatest()
    {
        // Passed newest-first, as the loader returns them.
        var result = DebtTrend.Build(new[] { E(0, 100, 2), E(10, 100, 5), E(20, 100, 10) });

        Assert.Equal(3, result.Points.Count);
        // Points[0] is the oldest (20 min ago), Points[^1] the newest (0 min ago).
        Assert.Equal(10, result.Points[0].IssueCount);
        Assert.Equal(2, result.Points[^1].IssueCount);
        Assert.True(result.Points[^1].IsLatest);
        Assert.False(result.Points[0].IsLatest);
        Assert.Equal("#2563EB", result.Points[^1].BarColor);
        Assert.Equal("#BFDBFE", result.Points[0].BarColor);
    }

    [Fact]
    public void Build_Improving_WhenRateDrops()
    {
        // oldest 10/100, newest 2/100 -> rate fell -> improving.
        var result = DebtTrend.Build(new[] { E(0, 100, 2), E(10, 100, 5), E(20, 100, 10) });
        Assert.True(result.HasTrend);
        Assert.StartsWith("Improving", result.Direction);
        Assert.Equal("#16A34A", result.DirectionColor);
        Assert.Contains("10.0", result.Summary);
        Assert.Contains("2.0", result.Summary);
        Assert.Contains("PII/secrets", result.Summary);   // honesty caveat present
    }

    [Fact]
    public void Build_Worsening_WhenRateRises()
    {
        var result = DebtTrend.Build(new[] { E(0, 100, 12), E(20, 100, 3) });
        Assert.StartsWith("Worsening", result.Direction);
        Assert.Equal("#DC2626", result.DirectionColor);
    }

    [Fact]
    public void Build_Stable_WhenRateUnchanged()
    {
        var result = DebtTrend.Build(new[] { E(0, 100, 4), E(20, 100, 4) });
        Assert.StartsWith("Stable", result.Direction);
        Assert.Equal("#64748B", result.DirectionColor);
    }

    [Fact]
    public void Build_NormalizesTallestBarToMax()
    {
        var result = DebtTrend.Build(new[] { E(0, 100, 2), E(20, 100, 10) });
        // The highest-rate run (oldest, 10/100) is the tallest bar.
        Assert.Equal(DebtTrend.MaxBarPx, result.Points[0].BarHeight, 3);
        Assert.True(result.Points[^1].BarHeight < DebtTrend.MaxBarPx);
        Assert.True(result.Points[^1].BarHeight >= 3.0); // nonzero rate keeps a visible floor
    }

    [Fact]
    public void Build_RatePer100_IsIssuesOverExamples()
    {
        var result = DebtTrend.Build(new[] { E(0, 200, 3) });
        Assert.Equal(1.5, result.Points[0].IssueRatePer100, 3); // 3/200*100
    }

    [Fact]
    public void Build_ZeroExamples_RateIsZeroNotDivideByZero()
    {
        var result = DebtTrend.Build(new[] { E(0, 0, 0), E(20, 100, 5) });
        // The empty run has rate 0 and a baseline sliver, not NaN.
        Assert.Equal(0.0, result.Points[^1].IssueRatePer100);
        Assert.True(result.Points[^1].BarHeight >= 1.0);
    }
}
