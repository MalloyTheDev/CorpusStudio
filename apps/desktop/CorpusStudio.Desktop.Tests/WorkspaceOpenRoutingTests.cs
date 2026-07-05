using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Open / Initialize Folder routing (v1.2.4, slice 3c) and the open-as-workspace
/// bridge. Pure decision logic + the MainWindowViewModel add/select bridge — no dialogs or
/// engine calls.</summary>
public sealed class WorkspaceOpenRoutingTests : IDisposable
{
    private readonly string _tempRoot;

    public WorkspaceOpenRoutingTests()
    {
        _tempRoot = Path.Combine(Path.GetTempPath(), "cs-open-" + Guid.NewGuid().ToString("N"));
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

    // ---- Classify (pure) ---------------------------------------------------------

    [Theory]
    [InlineData(true, false, false, WorkspaceOpenAction.OpenManifest)]
    [InlineData(true, true, false, WorkspaceOpenAction.OpenManifest)]     // manifest wins
    [InlineData(false, true, false, WorkspaceOpenAction.OfferInitializeDataset)]
    [InlineData(false, false, true, WorkspaceOpenAction.OfferCreateEmpty)]
    [InlineData(false, false, false, WorkspaceOpenAction.RejectOther)]
    public void Classify_CoversFourCases(bool hasManifest, bool hasExamples, bool isEmpty, WorkspaceOpenAction expected)
    {
        Assert.Equal(expected, WorkspaceOpenRouting.Classify(hasManifest, hasExamples, isEmpty));
    }

    // ---- ShouldReplaceWorkspace (the unsaved-work guard, pure) -------------------

    [Fact]
    public void ShouldReplaceWorkspace_NoUnsavedWork_ProceedsWithoutPrompting()
    {
        var prompted = false;
        var proceed = WorkspaceOpenRouting.ShouldReplaceWorkspace(
            hasUnsavedWork: false,
            confirmDiscard: () => { prompted = true; return false; });

        Assert.True(proceed);       // a clean workspace opens...
        Assert.False(prompted);     // ...and the user is never prompted
    }

    [Fact]
    public void ShouldReplaceWorkspace_UnsavedWork_PromptsAndHonorsTheChoice()
    {
        Assert.True(WorkspaceOpenRouting.ShouldReplaceWorkspace(true, confirmDiscard: () => true));   // discard confirmed
        Assert.False(WorkspaceOpenRouting.ShouldReplaceWorkspace(true, confirmDiscard: () => false)); // cancelled -> no open
    }

    // ---- ShouldReplaceWorkspaceAsync (the async dialog-seam form, Phase 0) -------

    [Fact]
    public async Task ShouldReplaceWorkspaceAsync_NoUnsavedWork_ProceedsWithoutPrompting()
    {
        var prompted = false;
        var proceed = await WorkspaceOpenRouting.ShouldReplaceWorkspaceAsync(
            hasUnsavedWork: false,
            confirmDiscard: () => { prompted = true; return Task.FromResult(false); });

        Assert.True(proceed);    // a clean workspace opens...
        Assert.False(prompted);  // ...and the async prompt is never awaited
    }

    [Fact]
    public async Task ShouldReplaceWorkspaceAsync_UnsavedWork_HonorsTheAwaitedChoice()
    {
        Assert.True(await WorkspaceOpenRouting.ShouldReplaceWorkspaceAsync(true, () => Task.FromResult(true)));   // confirmed
        Assert.False(await WorkspaceOpenRouting.ShouldReplaceWorkspaceAsync(true, () => Task.FromResult(false))); // cancelled
    }

    // ---- Inspect (over real folders) ---------------------------------------------

    [Fact]
    public void Inspect_ManifestFolder_OpensManifest()
    {
        var folder = NewDir("has-manifest");
        var manifests = new WorkspaceManifestService();
        manifests.Write(folder, new WorkspaceProjectManifest { ProjectId = "p", Name = "P", SchemaId = "instruction" });

        Assert.Equal(WorkspaceOpenAction.OpenManifest, WorkspaceOpenRouting.Inspect(folder, manifests));
    }

    [Fact]
    public void Inspect_DatasetNoManifest_OffersInitialize()
    {
        var folder = NewDir("dataset");
        File.WriteAllText(Path.Combine(folder, "examples.jsonl"), "{\"instruction\":\"x\",\"output\":\"y\"}\n");

        Assert.Equal(WorkspaceOpenAction.OfferInitializeDataset,
            WorkspaceOpenRouting.Inspect(folder, new WorkspaceManifestService()));
    }

    [Fact]
    public void Inspect_EmptyFolder_OffersCreate()
    {
        var folder = NewDir("empty");
        Assert.Equal(WorkspaceOpenAction.OfferCreateEmpty,
            WorkspaceOpenRouting.Inspect(folder, new WorkspaceManifestService()));
    }

    [Fact]
    public void Inspect_RandomFolder_Rejects()
    {
        var folder = NewDir("random");
        File.WriteAllText(Path.Combine(folder, "notes.txt"), "hello");
        Assert.Equal(WorkspaceOpenAction.RejectOther,
            WorkspaceOpenRouting.Inspect(folder, new WorkspaceManifestService()));
    }

    // ---- DeriveOpenArgs ----------------------------------------------------------

    [Fact]
    public void DeriveOpenArgs_RecognizedManifest_UsesItsFields()
    {
        var manifest = new WorkspaceProjectManifest { ProjectId = "p1", Name = "Demo", SchemaId = "chat" };
        var (id, name, schema) = WorkspaceOpenRouting.DeriveOpenArgs(manifest, "folder");
        Assert.Equal("p1", id);
        Assert.Equal("Demo", name);
        Assert.Equal("chat", schema);
    }

    [Fact]
    public void DeriveOpenArgs_NoManifest_UsesFolderNameAndDefaultSchema()
    {
        var (id, name, schema) = WorkspaceOpenRouting.DeriveOpenArgs(null, "myfolder");
        Assert.Equal("myfolder", id);
        Assert.Equal("myfolder", name);
        Assert.Equal("instruction", schema);
    }

    [Fact]
    public void DeriveOpenArgs_UnrecognizedManifest_FallsBackToFolder()
    {
        var manifest = new WorkspaceProjectManifest { Format = "something_else", ProjectId = "ignored", SchemaId = "chat" };
        var (id, name, schema) = WorkspaceOpenRouting.DeriveOpenArgs(manifest, "myfolder");
        Assert.Equal("myfolder", id);
        Assert.Equal("myfolder", name);
        Assert.Equal("instruction", schema);   // unrecognized -> defaults, not the manifest's fields
    }

    [Fact]
    public void DeriveOpenArgs_RecognizedButBlankFields_FallBackToFolder()
    {
        var manifest = new WorkspaceProjectManifest { ProjectId = "", Name = "  ", SchemaId = "" };
        var (id, name, schema) = WorkspaceOpenRouting.DeriveOpenArgs(manifest, "wsname");
        Assert.Equal("wsname", id);
        Assert.Equal("wsname", name);
        Assert.Equal("instruction", schema);
    }

    // ---- VM bridge: open an arbitrary path as the active workspace ----------------

    [Fact]
    public void AddProject_ArbitraryPath_BecomesActiveWorkspace()
    {
        var vm = new MainWindowViewModel();
        var path = Path.Combine(_tempRoot, "some", "folder");

        vm.AddProject("demo", "Demo", "instruction", "instruction", path);

        Assert.Equal(path, vm.ActiveProjectPath);
        Assert.True(vm.HasActiveProject);
        Assert.Contains(vm.Projects, p => p.ProjectPath == path);
    }

    [Fact]
    public void SelectingExistingProject_ReusesIt_NoDuplicate()
    {
        var vm = new MainWindowViewModel();
        var path = Path.Combine(_tempRoot, "ws");
        vm.AddProject("demo", "Demo", "instruction", "instruction", path);

        // Re-open the same folder: the code-behind re-selects the existing item.
        var existing = vm.Projects.First(p => p.ProjectPath == path);
        vm.SelectProject(existing, "instruction");

        Assert.Single(vm.Projects, p => p.ProjectPath == path);
        Assert.Equal(path, vm.ActiveProjectPath);
    }
}
