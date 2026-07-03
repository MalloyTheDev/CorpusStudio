using System;
using System.IO;
using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Workspace System slices 3-5: template scaffolding, the explorer tree, and safe
/// document open/save. Pure logic + tolerant, root-bounded I/O — no existing behavior is
/// touched and examples.jsonl is never mutated except by an explicit save.</summary>
public sealed class WorkspaceSystemTests : IDisposable
{
    private readonly string _tempRoot;

    public WorkspaceSystemTests()
    {
        _tempRoot = Path.Combine(Path.GetTempPath(), "cs-wss-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true); }
        catch (IOException) { /* best-effort */ }
    }

    private string NewDir(string name)
    {
        var dir = Path.Combine(_tempRoot, name);
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static WorkspaceProjectManifest Manifest(string id = "demo", string schema = "instruction", string template = "standard") =>
        new() { ProjectId = id, Name = id, SchemaId = schema, TemplateId = template };

    // ---- Templates: BuildPlan (pure) ---------------------------------------------

    [Fact]
    public void Templates_ListsFive()
    {
        var ids = new ProjectTemplateService().Templates.Select(t => t.Id).ToArray();
        Assert.Equal(new[] { "empty", "minimal", "standard", "full", "schema" }, ids);
    }

    [Fact]
    public void BuildPlan_Empty_OnlyManifestDir_NoExamples()
    {
        var plan = new ProjectTemplateService().BuildPlan("empty", "instruction", "Demo", "demo");
        Assert.Contains(".corpus", plan.Directories);
        Assert.DoesNotContain(plan.Files, f => f.RelativePath == "examples.jsonl");
    }

    [Fact]
    public void BuildPlan_Minimal_HasExamplesReadmeAssets_ButNotWorkingDirs()
    {
        var plan = new ProjectTemplateService().BuildPlan("minimal", "instruction", "Demo", "demo");
        Assert.Contains(plan.Files, f => f.RelativePath == "examples.jsonl");
        Assert.Contains(plan.Files, f => f.RelativePath == "README.md");
        Assert.Contains("assets", plan.Directories);
        Assert.DoesNotContain("splits", plan.Directories);
        Assert.DoesNotContain("reports", plan.Directories);
    }

    [Fact]
    public void BuildPlan_Standard_HasWorkingDirsAndCard()
    {
        var plan = new ProjectTemplateService().BuildPlan("standard", "instruction", "Demo", "demo");
        foreach (var dir in new[] { "imports", "imports/quarantine", "splits", "reports", "exports", "training_configs" })
            Assert.Contains(dir, plan.Directories);
        Assert.Contains(plan.Files, f => f.RelativePath == "dataset_card.json");
        Assert.Contains(plan.Files, f => f.RelativePath == ".corpus/workspace.json");
    }

    [Fact]
    public void BuildPlan_Full_HasAssetKindsAndGeneratedDirs()
    {
        var plan = new ProjectTemplateService().BuildPlan("full", "chat", "Demo", "demo");
        foreach (var dir in new[] { "assets/images", "assets/audio", "reports/quality", "training_runs", "model_artifacts", "dataset_versions" })
            Assert.Contains(dir, plan.Directories);
    }

    [Fact]
    public void BuildPlan_SchemaStarter_Instruction_SeedsRow()
    {
        var row = "{\"instruction\":\"Explain a variable.\",\"output\":\"It stores a value.\"}";
        var plan = new ProjectTemplateService().BuildPlan("schema", "instruction", "Demo", "demo", row);
        var examples = plan.Files.Single(f => f.RelativePath == "examples.jsonl");
        Assert.Contains("instruction", examples.Content);
        Assert.Contains("assets/misc", plan.Directories);
    }

    [Fact]
    public void BuildPlan_SchemaStarter_Image_LeavesExamplesEmpty_AddsGuidance()
    {
        var row = "{\"image\":\"assets/images/x.jpg\",\"caption\":\"...\"}";
        var plan = new ProjectTemplateService().BuildPlan("schema", "image_caption", "Demo", "demo", row);
        var examples = plan.Files.Single(f => f.RelativePath == "examples.jsonl");
        Assert.Equal(string.Empty, examples.Content);          // never seeded
        Assert.Contains("assets/images", plan.Directories);
        Assert.Contains(plan.Files, f => f.RelativePath == "README.md" && f.Content.Contains("image_caption"));
        Assert.NotNull(plan.Note);
    }

    // ---- Templates: Scaffold (guarded I/O) ---------------------------------------

    [Fact]
    public void Scaffold_CreatesPlanAndWritesManifest()
    {
        var root = Path.Combine(_tempRoot, "created");   // does not exist yet
        var service = new ProjectTemplateService();
        var plan = service.BuildPlan("standard", "instruction", "Demo", "demo");

        var result = service.Scaffold(root, plan, Manifest());
        Assert.True(result.Ok, result.Error);
        Assert.True(File.Exists(Path.Combine(root, "examples.jsonl")));
        Assert.True(Directory.Exists(Path.Combine(root, "splits")));

        var manifest = new WorkspaceManifestService().Read(root);
        Assert.True(manifest.Ok);
        Assert.Equal("instruction", manifest.Manifest!.SchemaId);
    }

    [Fact]
    public void Scaffold_RefusesNonEmptyFolder_WithoutConfirm()
    {
        var root = NewDir("nonempty");
        File.WriteAllText(Path.Combine(root, "stray.txt"), "hi");
        var service = new ProjectTemplateService();
        var result = service.Scaffold(root, service.BuildPlan("minimal", "instruction", "D", "d"), Manifest());
        Assert.False(result.Ok);
        Assert.Contains("not empty", result.Error);
    }

    [Fact]
    public void Scaffold_InitializesNonEmptyFolder_WhenConfirmed()
    {
        var root = NewDir("initme");
        File.WriteAllText(Path.Combine(root, "examples.jsonl"), "{\"instruction\":\"x\",\"output\":\"y\"}\n");
        var before = File.ReadAllText(Path.Combine(root, "examples.jsonl"));
        var service = new ProjectTemplateService();

        // "empty" template only adds the manifest — good for initializing an existing dataset.
        var result = service.Scaffold(root, service.BuildPlan("empty", "instruction", "D", "d"), Manifest(), allowNonEmpty: true);
        Assert.True(result.Ok, result.Error);
        Assert.True(new WorkspaceManifestService().HasManifest(root));
        Assert.Equal(before, File.ReadAllText(Path.Combine(root, "examples.jsonl"))); // existing rows untouched
    }

    [Fact]
    public void Scaffold_RefusesTraversalInPlan()
    {
        var root = Path.Combine(_tempRoot, "trav");
        var plan = new WorkspaceScaffoldPlan();
        plan.Directories.Add("../escape");
        var result = new ProjectTemplateService().Scaffold(root, plan, Manifest());
        Assert.False(result.Ok);
        Assert.Contains("unsafe", result.Error);
        Assert.False(Directory.Exists(Path.Combine(_tempRoot, "escape")));
    }

    // ---- Explorer tree -----------------------------------------------------------

    private string BuildSampleWorkspace()
    {
        var root = NewDir("ws-tree-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, ".corpus"));
        File.WriteAllText(Path.Combine(root, ".corpus", "project.json"), "{}");
        Directory.CreateDirectory(Path.Combine(root, "assets", "images"));
        Directory.CreateDirectory(Path.Combine(root, "reports", "quality"));
        File.WriteAllText(Path.Combine(root, "reports", "quality", "q.json"), "{}");
        Directory.CreateDirectory(Path.Combine(root, "node_modules"));   // junk -> ignored
        Directory.CreateDirectory(Path.Combine(root, ".git"));           // junk -> ignored
        File.WriteAllText(Path.Combine(root, "examples.jsonl"), "{}");
        File.WriteAllText(Path.Combine(root, "README.md"), "# hi");
        return root;
    }

    [Fact]
    public void Explorer_BuildTree_FoldersFirst_ThenAlphabetical()
    {
        var tree = new WorkspaceExplorerService().BuildTree(BuildSampleWorkspace());
        var names = tree.Children.Select(c => c.Name).ToList();

        var firstFileIndex = names.FindIndex(n => n is "examples.jsonl" or "README.md");
        var lastDirIndex = tree.Children.FindLastIndex(c => c.IsDirectory);
        Assert.True(lastDirIndex < firstFileIndex, "all folders must precede all files");

        var dirs = tree.Children.Where(c => c.IsDirectory).Select(c => c.Name).ToList();
        Assert.Equal(dirs.OrderBy(n => n, StringComparer.OrdinalIgnoreCase), dirs);
    }

    [Fact]
    public void Explorer_BuildTree_IgnoresJunkDirectories_ButKeepsCorpus()
    {
        var tree = new WorkspaceExplorerService().BuildTree(BuildSampleWorkspace());
        Assert.DoesNotContain(tree.Children, c => c.Name == "node_modules");
        Assert.DoesNotContain(tree.Children, c => c.Name == ".git");
        Assert.Contains(tree.Children, c => c.Name == ".corpus");
    }

    [Fact]
    public void Explorer_BuildTree_ClassifiesAndFlagsGeneratedAndCore()
    {
        var tree = new WorkspaceExplorerService().BuildTree(BuildSampleWorkspace());

        var examples = tree.Children.Single(c => c.Name == "examples.jsonl");
        Assert.Equal(WorkspaceFileKind.Jsonl, examples.FileKind);
        Assert.True(examples.IsDatasetCoreFile);

        var reports = tree.Children.Single(c => c.Name == "reports");
        var q = reports.Children.Single(c => c.Name == "quality").Children.Single();
        Assert.True(q.IsGeneratedArtifact);
    }

    [Fact]
    public void Explorer_CreateFolder_InsideRoot()
    {
        var root = NewDir("ws-newfolder");
        var result = new WorkspaceExplorerService().CreateFolder(root, "assets/images");
        Assert.True(result.Ok, result.Error);
        Assert.True(Directory.Exists(Path.Combine(root, "assets", "images")));
        Assert.Equal(WorkspaceFileKind.Folder, result.FileKind);
    }

    [Fact]
    public void Explorer_CreateFile_ClassifiesKind()
    {
        var root = NewDir("ws-newfile");
        var result = new WorkspaceExplorerService().CreateFile(root, "notes.md");
        Assert.True(result.Ok, result.Error);
        Assert.Equal(WorkspaceFileKind.Markdown, result.FileKind);
        Assert.True(File.Exists(Path.Combine(root, "notes.md")));
    }

    [Theory]
    [InlineData("../evil.txt")]
    [InlineData("a/../../evil.txt")]
    public void Explorer_Create_RefusesTraversal(string relative)
    {
        var root = NewDir("ws-trav-" + Math.Abs(relative.GetHashCode()));
        var result = new WorkspaceExplorerService().CreateFile(root, relative);
        Assert.False(result.Ok);
    }

    [Fact]
    public void Explorer_Create_RefusesOverwrite()
    {
        var root = NewDir("ws-overwrite");
        var svc = new WorkspaceExplorerService();
        Assert.True(svc.CreateFile(root, "dup.txt").Ok);
        Assert.False(svc.CreateFile(root, "dup.txt").Ok);   // second time refused
    }

    // ---- Documents ---------------------------------------------------------------

    [Fact]
    public void Document_OpenText_LoadsContent_NotDirty()
    {
        var root = NewDir("ws-doc");
        File.WriteAllText(Path.Combine(root, "README.md"), "# hello");
        var open = new WorkspaceDocumentService().Open(root, "README.md");
        Assert.True(open.Ok, open.Error);
        Assert.Equal("# hello", open.Document!.TextContent);
        Assert.False(open.Document.IsDirty);
        Assert.False(open.Document.IsReadOnly);
    }

    [Fact]
    public void Document_DirtyTracking_ThenSaveClears()
    {
        var root = NewDir("ws-doc-save");
        var path = Path.Combine(root, "notes.txt");
        File.WriteAllText(path, "one");
        var svc = new WorkspaceDocumentService();
        var doc = svc.Open(root, "notes.txt").Document!;

        doc.TextContent = "one two";
        Assert.True(doc.IsDirty);

        Assert.Null(svc.Save(doc));
        Assert.False(doc.IsDirty);
        Assert.Equal("one two", File.ReadAllText(path));
    }

    [Fact]
    public void Document_GeneratedReport_IsReadOnly_SaveRefused()
    {
        var root = NewDir("ws-doc-gen");
        Directory.CreateDirectory(Path.Combine(root, "reports", "quality"));
        var rel = "reports/quality/q.json";
        File.WriteAllText(Path.Combine(root, "reports", "quality", "q.json"), "{\"grade\":\"B\"}");

        var svc = new WorkspaceDocumentService();
        var doc = svc.Open(root, rel).Document!;
        Assert.True(doc.IsReadOnly);

        doc.TextContent = "tampered";     // ignored for dirty because read-only
        Assert.False(doc.IsDirty);
        Assert.NotNull(svc.Save(doc));    // save refused
        Assert.Equal("{\"grade\":\"B\"}", File.ReadAllText(Path.Combine(root, "reports", "quality", "q.json")));
    }

    [Fact]
    public void Document_Image_OpensMetadataOnly_NoCrash()
    {
        var root = NewDir("ws-doc-img");
        Directory.CreateDirectory(Path.Combine(root, "assets", "images"));
        File.WriteAllBytes(Path.Combine(root, "assets", "images", "a.png"), new byte[] { 0x89, 0x50, 0x4E, 0x47 });

        var doc = new WorkspaceDocumentService().Open(root, "assets/images/a.png").Document!;
        Assert.Equal(WorkspaceFileKind.Image, doc.FileKind);
        Assert.True(doc.IsReadOnly);
        Assert.False(string.IsNullOrEmpty(doc.ImagePreviewPath));
        Assert.Equal(string.Empty, doc.TextContent);
    }

    [Fact]
    public void Document_OpeningExamples_DoesNotMutateRows()
    {
        var root = NewDir("ws-doc-examples");
        var path = Path.Combine(root, "examples.jsonl");
        var original = "{\"instruction\":\"a\",\"output\":\"b\"}\n{\"instruction\":\"c\",\"output\":\"d\"}\n";
        File.WriteAllText(path, original);

        var doc = new WorkspaceDocumentService().Open(root, "examples.jsonl").Document!;
        Assert.False(doc.IsReadOnly);                 // editable, but explicit-save only
        Assert.Contains("single writer", doc.StatusMessage);
        Assert.Equal(original, File.ReadAllText(path)); // open never rewrote it
    }

    [Fact]
    public void Document_TooLarge_OpensReadOnlyPreview()
    {
        var root = NewDir("ws-doc-big");
        var path = Path.Combine(root, "big.txt");
        File.WriteAllText(path, new string('x', 5_000_000));

        var svc = new WorkspaceDocumentService { MaxEditableBytes = 1_000_000 };
        var doc = svc.Open(root, "big.txt").Document!;
        Assert.True(doc.IsReadOnly);
        Assert.Contains("read-only preview", doc.StatusMessage);
    }

    [Fact]
    public void FileMetadata_HumanSize_UsesInvariantCulture()
    {
        // Force a comma-decimal locale: the size must still render with a dot.
        var prior = System.Threading.Thread.CurrentThread.CurrentCulture;
        try
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-DE");
            Assert.Equal("1.5 KB", new WorkspaceFileMetadata { SizeBytes = 1536 }.HumanSize);
            Assert.Equal("512 B", new WorkspaceFileMetadata { SizeBytes = 512 }.HumanSize);
        }
        finally
        {
            System.Threading.Thread.CurrentThread.CurrentCulture = prior;
        }
    }
}
