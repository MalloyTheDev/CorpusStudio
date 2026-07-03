using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class QualityMetricTests
{
    // ---- QualityMetric factory ---------------------------------------------------

    [Fact]
    public void Info_IsNeutralGray()
    {
        var m = QualityMetric.Info("Examples", 128);
        Assert.Equal("128", m.Value);
        Assert.Equal("info", m.Severity);
        Assert.Equal("#64748B", m.StatusColor);
        Assert.Equal("•", m.StatusIcon);
    }

    [Fact]
    public void Issue_Zero_IsGreenOk()
    {
        var m = QualityMetric.Issue("Empty rows", 0);
        Assert.Equal("ok", m.Severity);
        Assert.Equal("#16A34A", m.StatusColor);
        Assert.Equal("✓", m.StatusIcon);
    }

    [Fact]
    public void Issue_Nonzero_IsAmberWarn()
    {
        var m = QualityMetric.Issue("Empty rows", 3);
        Assert.Equal("warn", m.Severity);
        Assert.Equal("#D97706", m.StatusColor);
        Assert.Equal("⚠", m.StatusIcon);
    }

    [Fact]
    public void Issue_Severe_NonzeroIsRedProblem_ZeroStillGreen()
    {
        var problem = QualityMetric.Issue("Possible PII / secrets", 2, severe: true);
        Assert.Equal("problem", problem.Severity);
        Assert.Equal("#DC2626", problem.StatusColor);
        Assert.Equal("⛔", problem.StatusIcon);

        // A severe metric at zero is still a clean green tick, not red.
        var clean = QualityMetric.Issue("Possible PII / secrets", 0, severe: true);
        Assert.Equal("ok", clean.Severity);
        Assert.Equal("#16A34A", clean.StatusColor);
    }

    // ---- ApplyQualityReport -> metric grid + PII-aware status --------------------

    private static QualityReport Report(
        int examples = 100, int empty = 0, int dupExact = 0, int dupNorm = 0,
        int lowInfo = 0, int synthetic = 0, int pii = 0, int tokenOutliers = 0) => new()
    {
        ExampleCount = examples,
        EmptyRowCount = empty,
        DuplicateExactCount = dupExact,
        DuplicateNormalizedCount = dupNorm,
        LowInformationCount = lowInfo,
        SyntheticPatternCount = synthetic,
        PiiFindingCount = pii,
        PiiFindings = pii > 0
            ? new[] { new PiiFinding { Kind = "api_key", Severity = "high", MatchCount = pii } }
            : [],
        TokenLengthOutlierCount = tokenOutliers,
        TokenLengthOutliers = tokenOutliers > 0
            ? new[] { new TokenLengthOutlier { RowNumber = 1, TokenCount = 5000 } }
            : [],
    };

    [Fact]
    public void ApplyQualityReport_Clean_SevenMetrics_GreenStatus()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyQualityReport(Report());

        Assert.True(vm.HasQualityMetrics);
        Assert.Equal(7, vm.QualityMetrics.Count);
        Assert.Equal("Examples", vm.QualityMetrics[0].Label);
        Assert.Equal("100", vm.QualityMetrics[0].Value);
        Assert.Equal("info", vm.QualityMetrics[0].Severity);
        Assert.All(vm.QualityMetrics.Skip(1), m => Assert.Equal("ok", m.Severity)); // all issue counts 0

        Assert.Equal("No basic quality issues found.", vm.QualityStatusLine);
        Assert.Equal("#15803D", vm.QualityStatusColor);
        Assert.False(vm.HasQualityDetail);
    }

    [Fact]
    public void ApplyQualityReport_CoreIssue_AmberStatus()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyQualityReport(Report(empty: 3));

        var emptyMetric = vm.QualityMetrics.Single(m => m.Label == "Empty rows");
        Assert.Equal("warn", emptyMetric.Severity);
        Assert.Equal("3", emptyMetric.Value);

        Assert.Equal("Review the flagged rows before export.", vm.QualityStatusLine);
        Assert.Equal("#B45309", vm.QualityStatusColor);
    }

    [Fact]
    public void ApplyQualityReport_Pii_RedStatus_And_DetailPresent()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyQualityReport(Report(pii: 2));

        var piiMetric = vm.QualityMetrics.Single(m => m.Label == "Possible PII / secrets");
        Assert.Equal("problem", piiMetric.Severity);
        Assert.Equal("⛔", piiMetric.StatusIcon);

        // PII escalates the banner to red even though only PII is nonzero.
        Assert.Contains("PII / secrets detected", vm.QualityStatusLine);
        Assert.Equal("#B91C1C", vm.QualityStatusColor);

        Assert.True(vm.HasQualityDetail);
        Assert.Contains("PII / secrets detected", vm.QualityDetail);
    }

    [Fact]
    public void ApplyQualityReport_TokenOutliers_ShowInDetailNotStatus()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyQualityReport(Report(tokenOutliers: 2));

        // Token outliers aren't a core-issue metric, so the banner stays green, but the detail
        // block surfaces them.
        Assert.Equal("No basic quality issues found.", vm.QualityStatusLine);
        Assert.True(vm.HasQualityDetail);
        Assert.Contains("Token-length outliers", vm.QualityDetail);
    }

    [Fact]
    public void ApplyQualityReport_StillBuildsQualitySummaryForDashboard()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyQualityReport(Report(examples: 42, empty: 1));

        // The full text summary is unchanged (the dashboard card binds it).
        Assert.Contains("Examples: 42", vm.QualitySummary);
        Assert.Contains("Empty rows: 1", vm.QualitySummary);
        Assert.Contains("Status:", vm.QualitySummary);
    }

    [Fact]
    public void SetQualityError_ClearsMetrics_FallsBackToText()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyQualityReport(Report(empty: 2));
        Assert.True(vm.HasQualityMetrics);

        vm.SetQualityError("engine exploded");
        Assert.False(vm.HasQualityMetrics);
        Assert.Empty(vm.QualityMetrics);
        Assert.Contains("engine exploded", vm.QualitySummary);
    }

    [Fact]
    public void SetQualityInProgress_ClearsMetrics()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyQualityReport(Report());
        vm.SetQualityInProgress();
        Assert.False(vm.HasQualityMetrics);
        Assert.Empty(vm.QualityMetrics);
    }
}
