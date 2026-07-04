using System.IO;
using System.Linq;
using System.Text;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Atomicity + correctness of the examples.jsonl append paths (audit item 1).
/// Both the draft-save and import-commit appends now go through a temp+File.Replace swap
/// instead of File.AppendAllText, so a crash mid-append cannot tear the last line. These
/// tests pin the observable guarantees: existing rows preserved, byte-identical UTF-8
/// (no BOM), and no leftover temp file.</summary>
public sealed class AtomicAppendTests
{
    private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

    private static string ExamplesPath(string dir) => Path.Combine(dir, "examples.jsonl");

    private static void AssertNoTempResidue(string dir) =>
        Assert.Empty(Directory.GetFiles(dir, "*.tmp-*"));

    private static void AssertNoBom(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var hasBom = bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF;
        Assert.False(hasBom, "examples.jsonl must not carry a UTF-8 BOM.");
    }

    private static string[] NonBlankLines(string path) =>
        File.ReadAllLines(path, Encoding.UTF8).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();

    [Fact]
    public void AppendDraft_PreservesExisting_AppendsNewRow_NoBom_NoTempResidue()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        File.WriteAllText(ExamplesPath(project.Path),
            "{\"instruction\":\"existing\",\"output\":\"0\"}\n", Utf8NoBom);

        var count = service.AppendDraftToProjectExamples(
            project.Path, "{\"instruction\":\"new\",\"output\":\"1\"}");

        Assert.Equal(1, count);
        var lines = NonBlankLines(ExamplesPath(project.Path));
        Assert.Equal(2, lines.Length);
        Assert.Contains(lines, l => l.Contains("\"instruction\":\"existing\""));
        Assert.Contains(lines, l => l.Contains("\"instruction\":\"new\""));
        AssertNoBom(ExamplesPath(project.Path));
        AssertNoTempResidue(project.Path);
    }

    [Fact]
    public void AppendDraft_ToMissingFile_CreatesIt()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        Assert.False(File.Exists(ExamplesPath(project.Path)));

        service.AppendDraftToProjectExamples(project.Path, "{\"instruction\":\"a\",\"output\":\"1\"}");

        Assert.True(File.Exists(ExamplesPath(project.Path)));
        Assert.Single(NonBlankLines(ExamplesPath(project.Path)));
        AssertNoBom(ExamplesPath(project.Path));
        AssertNoTempResidue(project.Path);
    }

    [Fact]
    public void CommitImport_AppendsAtomically_NoTempResidue_NoBom()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();
        File.WriteAllText(ExamplesPath(project.Path),
            "{\"instruction\":\"existing\",\"output\":\"0\"}\n", Utf8NoBom);

        var importPath = Path.Combine(project.Path, "import.jsonl");
        File.WriteAllText(importPath, "{\"instruction\":\"new\",\"output\":\"1\"}\n", Utf8NoBom);
        var report = new ImportPreviewReport { Valid = true, SchemaId = "instruction", Path = importPath };

        var result = service.CommitJsonlImportToProjectExamples(project.Path, importPath, report);

        Assert.Equal(1, result.ImportedCount);
        var lines = NonBlankLines(ExamplesPath(project.Path));
        Assert.Equal(2, lines.Length);
        Assert.Contains(lines, l => l.Contains("\"instruction\":\"existing\""));
        Assert.Contains(lines, l => l.Contains("\"instruction\":\"new\""));
        AssertNoBom(ExamplesPath(project.Path));
        AssertNoTempResidue(project.Path);
    }

    [Fact]
    public void RepeatedAppends_AccumulateInOrder_NoTempResidue()
    {
        using var project = new TempProjectDirectory();
        var service = new PythonEngineService();

        service.AppendDraftToProjectExamples(project.Path, "{\"instruction\":\"a\",\"output\":\"1\"}");
        service.AppendDraftToProjectExamples(project.Path, "{\"instruction\":\"b\",\"output\":\"2\"}");
        service.AppendDraftToProjectExamples(project.Path, "{\"instruction\":\"c\",\"output\":\"3\"}");

        var lines = NonBlankLines(ExamplesPath(project.Path));
        Assert.Equal(3, lines.Length);
        Assert.Contains("\"instruction\":\"a\"", lines[0]);
        Assert.Contains("\"instruction\":\"c\"", lines[2]);
        AssertNoTempResidue(project.Path);
    }
}
