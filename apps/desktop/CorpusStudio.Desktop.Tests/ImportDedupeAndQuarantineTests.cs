using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Import idempotency + quarantine clearing (audit item 3). Re-importing the same
/// rows no longer doubles the dataset, and a repaired/retried quarantine row is removed from
/// its *_rejected.jsonl instead of orphaning.</summary>
public sealed class ImportDedupeAndQuarantineTests
{
    private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

    private static string ExamplesPath(string dir) => Path.Combine(dir, "examples.jsonl");

    private static string WriteImport(string dir, params string[] lines)
    {
        var path = Path.Combine(dir, "import.jsonl");
        File.WriteAllText(path, string.Join("\n", lines) + "\n", Utf8NoBom);
        return path;
    }

    private static ImportPreviewReport ValidReport(string path) =>
        new() { Valid = true, SchemaId = "instruction", Path = path };

    private static int NonBlankLineCount(string path) =>
        File.ReadLines(path, Encoding.UTF8).Count(l => !string.IsNullOrWhiteSpace(l));

    // --- (a) idempotent import commit -------------------------------------------------

    [Fact]
    public void Commit_SkipsRowsAlreadyInExamples()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        File.WriteAllText(ExamplesPath(project.Path),
            "{\"instruction\":\"a\",\"output\":\"1\"}\n", Utf8NoBom);

        var importPath = WriteImport(project.Path,
            "{\"instruction\":\"a\",\"output\":\"1\"}",   // already present
            "{\"instruction\":\"b\",\"output\":\"2\"}");   // new
        var result = service.CommitJsonlImportToProjectExamples(project.Path, importPath, ValidReport(importPath));

        Assert.Equal(1, result.ImportedCount);
        Assert.Equal(1, result.SkippedDuplicateCount);
        Assert.Equal(2, NonBlankLineCount(ExamplesPath(project.Path)));
    }

    [Fact]
    public void Commit_SameFileTwice_IsIdempotent()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        var importPath = WriteImport(project.Path,
            "{\"instruction\":\"a\",\"output\":\"1\"}",
            "{\"instruction\":\"b\",\"output\":\"2\"}");

        var first = service.CommitJsonlImportToProjectExamples(project.Path, importPath, ValidReport(importPath));
        var second = service.CommitJsonlImportToProjectExamples(project.Path, importPath, ValidReport(importPath));

        Assert.Equal(2, first.ImportedCount);
        Assert.Equal(0, second.ImportedCount);
        Assert.Equal(2, second.SkippedDuplicateCount);
        Assert.Equal(2, NonBlankLineCount(ExamplesPath(project.Path))); // not 4
    }

    [Fact]
    public void Commit_DedupesWithinTheSameBatch()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        var importPath = WriteImport(project.Path,
            "{\"instruction\":\"dup\",\"output\":\"x\"}",
            "{\"instruction\":\"dup\",\"output\":\"x\"}"); // identical

        var result = service.CommitJsonlImportToProjectExamples(project.Path, importPath, ValidReport(importPath));

        Assert.Equal(1, result.ImportedCount);
        Assert.Equal(1, result.SkippedDuplicateCount);
    }

    [Fact]
    public void Commit_DedupeIsKeyOrderIndependent()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        File.WriteAllText(ExamplesPath(project.Path),
            "{\"instruction\":\"a\",\"output\":\"1\"}\n", Utf8NoBom);

        // Same content, different key order — must be recognized as a duplicate.
        var importPath = WriteImport(project.Path, "{\"output\":\"1\",\"instruction\":\"a\"}");
        var result = service.CommitJsonlImportToProjectExamples(project.Path, importPath, ValidReport(importPath));

        Assert.Equal(0, result.ImportedCount);
        Assert.Equal(1, result.SkippedDuplicateCount);
        Assert.Equal(1, NonBlankLineCount(ExamplesPath(project.Path)));
    }

    // --- (b) quarantine clearing ------------------------------------------------------

    private static string WriteQuarantineFile(string dir, params (int row, string source, string raw)[] entries)
    {
        var quarantineDir = Path.Combine(dir, "import_quarantine");
        Directory.CreateDirectory(quarantineDir);
        var path = Path.Combine(quarantineDir, "20260101000000_src_rejected.jsonl");
        var lines = entries.Select(e => JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["source_path"] = e.source,
            ["row_number"] = e.row,
            ["raw"] = e.raw,
            ["errors"] = System.Array.Empty<object>(),
        }));
        File.WriteAllText(path, string.Join("\n", lines) + "\n", Utf8NoBom);
        return path;
    }

    [Fact]
    public void RemoveImportQuarantineItem_DropsMatchingEntry_KeepsOthers()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        var qpath = WriteQuarantineFile(project.Path,
            (2, "src.jsonl", "{\"bad\":1}"),
            (5, "src.jsonl", "{\"bad\":2}"));

        var target = new ImportQuarantineItem
        {
            RowNumber = 2, SourcePath = "src.jsonl", Raw = "{\"bad\":1}", QuarantinePath = qpath,
        };
        service.RemoveImportQuarantineItem(target);

        var remaining = service.LoadImportQuarantineItems(project.Path);
        var item = Assert.Single(remaining);
        Assert.Equal(5, item.RowNumber);
    }

    [Fact]
    public void RemoveImportQuarantineItem_DeletesFileWhenLastEntryRemoved()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        var qpath = WriteQuarantineFile(project.Path, (2, "src.jsonl", "{\"bad\":1}"));

        service.RemoveImportQuarantineItem(new ImportQuarantineItem
        {
            RowNumber = 2, SourcePath = "src.jsonl", Raw = "{\"bad\":1}", QuarantinePath = qpath,
        });

        Assert.False(File.Exists(qpath));
        Assert.Empty(service.LoadImportQuarantineItems(project.Path));
    }

    // --- retry wiring -----------------------------------------------------------------

    [Fact]
    public void Retry_ThenTakePendingRetryItem_ReturnsItemOnce()
    {
        var vm = new MainWindowViewModel();
        var item = new ImportQuarantineItem { RowNumber = 3, SourcePath = "s", Raw = "{\"x\":1}", QuarantinePath = "q" };
        vm.Quarantine.SetItems(new[] { item });
        vm.Quarantine.SelectedImportQuarantineItem = item;

        vm.RetrySelectedImportQuarantineItem();
        Assert.Equal(item.Raw, vm.DraftText);

        Assert.Same(item, vm.TakePendingRetryItem()); // repaired row is available to clear
        Assert.Null(vm.TakePendingRetryItem());        // consumed exactly once
    }

    // --- bounded quarantine retention (Tier-3) ----------------------------------------

    [Fact]
    public void PruneQuarantineFiles_KeepsNewestAndDropsOldest()
    {
        using var project = new TempProjectDirectory();
        var quarantineDir = Path.Combine(project.Path, "import_quarantine");
        Directory.CreateDirectory(quarantineDir);

        // 55 files with fixed-width, sortable timestamp prefixes (chronological == ordinal).
        for (int i = 0; i < 55; i++)
        {
            File.WriteAllText(
                Path.Combine(quarantineDir, $"202601{i:D2}000000_src_rejected.jsonl"), "{}\n", Utf8NoBom);
        }

        PythonEngineService.PruneQuarantineFiles(quarantineDir, 50);

        var remaining = Directory.EnumerateFiles(quarantineDir, "*_rejected.jsonl")
            .Select(Path.GetFileName)
            .OrderBy(name => name, System.StringComparer.Ordinal)
            .ToList();
        Assert.Equal(50, remaining.Count);
        Assert.Equal("20260105000000_src_rejected.jsonl", remaining.First()); // oldest 5 pruned
        Assert.Equal("20260154000000_src_rejected.jsonl", remaining.Last());  // newest retained
    }

    [Fact]
    public void PruneQuarantineFiles_NoOpWhenUnderLimit()
    {
        using var project = new TempProjectDirectory();
        var quarantineDir = Path.Combine(project.Path, "import_quarantine");
        Directory.CreateDirectory(quarantineDir);
        File.WriteAllText(Path.Combine(quarantineDir, "20260101000000_a_rejected.jsonl"), "{}\n", Utf8NoBom);

        PythonEngineService.PruneQuarantineFiles(quarantineDir, 50);

        Assert.Single(Directory.EnumerateFiles(quarantineDir, "*_rejected.jsonl"));
    }

    // --- auto-capture a version after import (closes the "silent import" gap) ----------

    [Fact]
    public void ShouldAutoCapture_OnlyWhenRowsWereAdded()
    {
        Assert.True(new ImportCommitResult(ImportedCount: 3, QuarantinedCount: 0, QuarantinePath: null).ShouldAutoCapture);
        // all duplicates skipped -> nothing changed -> no snapshot
        Assert.False(new ImportCommitResult(ImportedCount: 0, QuarantinedCount: 0, QuarantinePath: null, SkippedDuplicateCount: 4).ShouldAutoCapture);
    }

    [Fact]
    public void AutoCaptureLabel_DescribesTheImport()
    {
        Assert.Equal("After import (+5 rows)",
            new ImportCommitResult(ImportedCount: 5, QuarantinedCount: 0, QuarantinePath: null).AutoCaptureLabel);
    }
}
