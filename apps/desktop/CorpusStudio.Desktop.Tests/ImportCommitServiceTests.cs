using System.IO;
using System.Text;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>
/// Data-integrity coverage for the JSONL import commit + quarantine roundtrip:
/// only valid rows reach examples.jsonl, failed rows are quarantined and
/// reloadable, and blank lines are skipped without shifting row numbers.
/// </summary>
public sealed class ImportCommitServiceTests
{
    private static string WriteImport(string dir, params string[] lines)
    {
        var path = Path.Combine(dir, "import.jsonl");
        File.WriteAllText(path, string.Join("\n", lines) + "\n", new UTF8Encoding(false));
        return path;
    }

    private static ImportPreviewReport ReportWithFailures(string path, params int[] failedRowNumbers)
    {
        return new ImportPreviewReport
        {
            Valid = failedRowNumbers.Length == 0,
            SchemaId = "instruction",
            Path = path,
            FailedRows = failedRowNumbers
                .Select(rowNumber => new ImportFailure
                {
                    RowNumber = rowNumber,
                    RawPreview = "preview",
                    Errors = [new ValidationIssue { Message = "Missing required field: output" }],
                })
                .ToList(),
        };
    }

    [Fact]
    public void Commit_AppendsOnlyValidRows_SkipsFailedAndBlankRows()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        // row 1 valid, row 2 blank, row 3 valid, row 4 failed (missing output).
        var importPath = WriteImport(
            project.Path,
            "{\"instruction\":\"a\",\"output\":\"1\"}",
            "",
            "{\"instruction\":\"b\",\"output\":\"2\"}",
            "{\"instruction\":\"c\"}"
        );
        var report = ReportWithFailures(importPath, 4);

        var result = service.CommitJsonlImportToProjectExamples(project.Path, importPath, report);

        Assert.Equal(2, result.ImportedCount);
        Assert.Equal(1, result.QuarantinedCount);
        Assert.NotNull(result.QuarantinePath);

        var committed = File.ReadAllLines(Path.Combine(project.Path, "examples.jsonl"), Encoding.UTF8)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .ToArray();
        Assert.Equal(2, committed.Length);
        Assert.Contains(committed, line => line.Contains("\"instruction\":\"a\""));
        Assert.Contains(committed, line => line.Contains("\"instruction\":\"b\""));
        Assert.DoesNotContain(committed, line => line.Contains("\"instruction\":\"c\""));
    }

    [Fact]
    public void Commit_QuarantinesFailedRows_AndTheyReload()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var importPath = WriteImport(
            project.Path,
            "{\"instruction\":\"good\",\"output\":\"ok\"}",
            "{\"instruction\":\"bad\"}"
        );
        var report = ReportWithFailures(importPath, 2);

        service.CommitJsonlImportToProjectExamples(project.Path, importPath, report);

        var quarantined = service.LoadImportQuarantineItems(project.Path);
        var item = Assert.Single(quarantined);
        Assert.Equal(2, item.RowNumber);
        Assert.Contains("bad", item.Raw);
        Assert.Equal(importPath, item.SourcePath);
        Assert.Contains(item.Errors, error => error.Message.Contains("output"));
    }

    [Fact]
    public void Commit_WithNoFailures_CommitsEverything_AndWritesNoQuarantine()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var importPath = WriteImport(
            project.Path,
            "{\"instruction\":\"a\",\"output\":\"1\"}",
            "{\"instruction\":\"b\",\"output\":\"2\"}"
        );
        var report = ReportWithFailures(importPath);

        var result = service.CommitJsonlImportToProjectExamples(project.Path, importPath, report);

        Assert.Equal(2, result.ImportedCount);
        Assert.Equal(0, result.QuarantinedCount);
        Assert.Null(result.QuarantinePath);
        Assert.Empty(service.LoadImportQuarantineItems(project.Path));
    }

    [Fact]
    public void Commit_AppendsToExistingExamples_WithoutOverwriting()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        var examplesPath = Path.Combine(project.Path, "examples.jsonl");
        File.WriteAllText(examplesPath, "{\"instruction\":\"existing\",\"output\":\"0\"}\n", new UTF8Encoding(false));

        var importPath = WriteImport(project.Path, "{\"instruction\":\"new\",\"output\":\"1\"}");
        var result = service.CommitJsonlImportToProjectExamples(project.Path, importPath, ReportWithFailures(importPath));

        Assert.Equal(1, result.ImportedCount);
        var committed = File.ReadAllLines(examplesPath, Encoding.UTF8)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .ToArray();
        Assert.Equal(2, committed.Length);
        Assert.Contains(committed, line => line.Contains("\"instruction\":\"existing\""));
        Assert.Contains(committed, line => line.Contains("\"instruction\":\"new\""));
    }
}
