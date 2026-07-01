using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class AiAssistRewriteBatchServiceTests
{
    private static AiAssistRewriteBatch NewBatch() => new()
    {
        SchemaId = "instruction",
        RowNumbers = new List<int> { 3, 1, 3, -2, 1 },
        IssueCount = 2,
        IssueSummary = "  medium synthetic_pattern: repeated opening  ",
        SourceDraft = "draft text\n",
        Instruction = "  Rewrite the affected rows.  ",
    };

    [Fact]
    public void Load_ReturnsEmpty_WhenFileMissing()
    {
        using var project = new TempProjectDirectory();
        Assert.Empty(new PythonEngineService().LoadAiAssistRewriteBatches(project.Path));
    }

    [Fact]
    public void Save_Throws_WhenSourceDraftMissing()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        Assert.Throws<InvalidOperationException>(() =>
            service.SaveAiAssistRewriteBatch(project.Path, new AiAssistRewriteBatch { Instruction = "x" }));
    }

    [Fact]
    public void Save_Throws_WhenInstructionMissing()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        Assert.Throws<InvalidOperationException>(() =>
            service.SaveAiAssistRewriteBatch(project.Path, new AiAssistRewriteBatch { SourceDraft = "x" }));
    }

    [Fact]
    public void Save_NormalizesRowNumbersAndTrims_AndRoundTrips()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var saved = service.SaveAiAssistRewriteBatch(project.Path, NewBatch());

        Assert.Equal(new[] { 1, 3 }, saved.RowNumbers); // distinct, positive, ordered
        Assert.Equal("Rewrite the affected rows.", saved.Instruction);

        var entry = Assert.Single(service.LoadAiAssistRewriteBatches(project.Path));
        Assert.Equal(saved.BatchId, entry.BatchId);
        Assert.Equal(new[] { 1, 3 }, entry.RowNumbers);
    }

    [Fact]
    public void Save_UpsertsByBatchId()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var first = service.SaveAiAssistRewriteBatch(project.Path, NewBatch());
        service.SaveAiAssistRewriteBatch(project.Path, new AiAssistRewriteBatch
        {
            BatchId = first.BatchId,
            SchemaId = "instruction",
            RowNumbers = new List<int> { 5 },
            SourceDraft = "new draft",
            Instruction = "new instruction",
        });

        var entry = Assert.Single(service.LoadAiAssistRewriteBatches(project.Path));
        Assert.Equal(first.BatchId, entry.BatchId);
        Assert.Equal(new[] { 5 }, entry.RowNumbers);
        Assert.Equal("new instruction", entry.Instruction);
    }
}
