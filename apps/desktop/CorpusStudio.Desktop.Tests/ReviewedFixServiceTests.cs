using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class ReviewedFixServiceTests
{
    private static ReviewedFixRecord NewFix(string exampleId, int rowNumber = 1, double originalScore = 40) =>
        new()
        {
            ExampleId = exampleId,
            RowNumber = rowNumber,
            SchemaId = "instruction",
            OriginalScore = originalScore,
            FailureReason = "score_below_threshold",
            SourceReport = "2026-07-01 report",
        };

    private static EvaluationExampleResult Result(string exampleId, double score, bool passed) =>
        new() { ExampleId = exampleId, Score = score, Passed = passed };

    [Fact]
    public void LoadReviewedFixes_ReturnsEmpty_WhenFileMissing()
    {
        using var project = new TempProjectDirectory();
        Assert.Empty(new PythonEngineService().LoadReviewedFixes(project.Path));
    }

    [Fact]
    public void RecordReviewedFix_Throws_WhenExampleIdMissing()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        Assert.Throws<InvalidOperationException>(() =>
            service.RecordReviewedFix(project.Path, new ReviewedFixRecord { ExampleId = "" }));
    }

    [Fact]
    public void RecordReviewedFix_CreatesVersionOne_Edited_AndRoundTrips()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var saved = service.RecordReviewedFix(project.Path, NewFix("row-1"));

        Assert.Equal(1, saved.Version);
        Assert.Equal(ReviewedFixRecord.StatusEdited, saved.Status);
        Assert.Null(saved.LatestScore);

        var entry = Assert.Single(service.LoadReviewedFixes(project.Path));
        Assert.Equal("row-1", entry.ExampleId);
        Assert.Equal("score_below_threshold", entry.FailureReason);
        Assert.Equal("2026-07-01 report", entry.SourceReport);
    }

    [Fact]
    public void RecordReviewedFix_IncrementsVersion_ForSameExample()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        service.RecordReviewedFix(project.Path, NewFix("row-1"));
        var second = service.RecordReviewedFix(project.Path, NewFix("row-1"));

        Assert.Equal(2, second.Version);
        var versions = service.LoadReviewedFixes(project.Path)
            .Where(fix => fix.ExampleId == "row-1")
            .Select(fix => fix.Version)
            .OrderBy(version => version)
            .ToArray();
        Assert.Equal(new[] { 1, 2 }, versions);
    }

    [Fact]
    public void RecordReviewedFix_StartsAtVersionOne_PerExample()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var first = service.RecordReviewedFix(project.Path, NewFix("row-1"));
        var other = service.RecordReviewedFix(project.Path, NewFix("row-2"));

        Assert.Equal(1, first.Version);
        Assert.Equal(1, other.Version);
    }

    [Fact]
    public void Reconcile_MarksResolved_WhenExampleNowPasses()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        service.RecordReviewedFix(project.Path, NewFix("row-1"));

        var reconciled = service.ReconcileReviewedFixes(
            project.Path,
            new List<EvaluationExampleResult> { Result("row-1", 92, passed: true) }
        );

        var entry = Assert.Single(reconciled);
        Assert.Equal(ReviewedFixRecord.StatusResolved, entry.Status);
        Assert.Equal((double?)92, entry.LatestScore);
    }

    [Fact]
    public void Reconcile_MarksStillFailing_WhenExampleStillFails()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        service.RecordReviewedFix(project.Path, NewFix("row-1"));

        var reconciled = service.ReconcileReviewedFixes(
            project.Path,
            new List<EvaluationExampleResult> { Result("row-1", 30, passed: false) }
        );

        var entry = Assert.Single(reconciled);
        Assert.Equal(ReviewedFixRecord.StatusStillFailing, entry.Status);
        Assert.Equal((double?)30, entry.LatestScore);
    }

    [Fact]
    public void Reconcile_LeavesFixEdited_WhenExampleAbsentFromRun()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        service.RecordReviewedFix(project.Path, NewFix("row-1"));

        var reconciled = service.ReconcileReviewedFixes(
            project.Path,
            new List<EvaluationExampleResult> { Result("row-2", 90, passed: true) }
        );

        var entry = Assert.Single(reconciled);
        Assert.Equal(ReviewedFixRecord.StatusEdited, entry.Status);
        Assert.Null(entry.LatestScore);
    }

    [Fact]
    public void Reconcile_UpdatesOnlyLatestVersion()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        service.RecordReviewedFix(project.Path, NewFix("row-1")); // v1
        service.RecordReviewedFix(project.Path, NewFix("row-1")); // v2

        service.ReconcileReviewedFixes(
            project.Path,
            new List<EvaluationExampleResult> { Result("row-1", 95, passed: true) }
        );

        var byVersion = service.LoadReviewedFixes(project.Path).ToDictionary(fix => fix.Version);
        Assert.Equal(ReviewedFixRecord.StatusEdited, byVersion[1].Status);
        Assert.Equal(ReviewedFixRecord.StatusResolved, byVersion[2].Status);
    }
}
