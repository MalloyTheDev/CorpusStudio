using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class EvaluationFailureFilterServiceTests
{
    [Fact]
    public void Load_ReturnsEmpty_WhenFileMissing()
    {
        using var project = new TempProjectDirectory();
        Assert.Empty(new PythonEngineService().LoadEvaluationFailureFilters(project.Path));
    }

    [Fact]
    public void Save_Throws_WhenNameMissing()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        Assert.Throws<InvalidOperationException>(() =>
            service.SaveEvaluationFailureFilter(project.Path, new EvaluationFailureFilter { Name = "   " }));
    }

    [Fact]
    public void Save_AppliesDefaults_AndRoundTrips()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var saved = service.SaveEvaluationFailureFilter(project.Path, new EvaluationFailureFilter
        {
            Name = "Failing chat",
            Status = "Failed",
            Tag = "",
            FailureReason = "score_below_threshold",
            ScoreBand = "0-49",
        });

        Assert.Equal("All", saved.Tag); // empty coerced to "All"

        var entry = Assert.Single(service.LoadEvaluationFailureFilters(project.Path));
        Assert.Equal("Failing chat", entry.Name);
        Assert.Equal("Failed", entry.Status);
        Assert.Equal("score_below_threshold", entry.FailureReason);
        Assert.Equal("0-49", entry.ScoreBand);
    }

    [Fact]
    public void Save_UpsertsByName_CaseInsensitive()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        service.SaveEvaluationFailureFilter(project.Path, new EvaluationFailureFilter { Name = "View", Status = "Failed" });
        service.SaveEvaluationFailureFilter(project.Path, new EvaluationFailureFilter { Name = "view", Status = "Passed" });

        var entry = Assert.Single(service.LoadEvaluationFailureFilters(project.Path));
        Assert.Equal("Passed", entry.Status);
    }

    [Fact]
    public void Load_ReturnsSortedByName()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        service.SaveEvaluationFailureFilter(project.Path, new EvaluationFailureFilter { Name = "Zeta" });
        service.SaveEvaluationFailureFilter(project.Path, new EvaluationFailureFilter { Name = "Alpha" });

        var names = service.LoadEvaluationFailureFilters(project.Path).Select(filter => filter.Name).ToArray();
        Assert.Equal(new[] { "Alpha", "Zeta" }, names);
    }
}
