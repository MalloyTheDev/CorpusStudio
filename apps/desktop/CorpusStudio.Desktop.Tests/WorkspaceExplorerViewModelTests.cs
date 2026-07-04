using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Universal Workspace Explorer view-model (v1.2.4, slice 3b): tree, document
/// tabs, viewers, and guarded create — over a temp workspace. examples.jsonl is never
/// mutated except by an explicit save.</summary>
public sealed class WorkspaceExplorerViewModelTests : IDisposable
{
    private readonly string _tempRoot;

    public WorkspaceExplorerViewModelTests()
    {
        _tempRoot = Path.Combine(Path.GetTempPath(), "cs-exp-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true); }
        catch (IOException) { /* best-effort */ }
    }

    private string NewWorkspace()
    {
        var root = Path.Combine(_tempRoot, "ws-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        File.WriteAllText(Path.Combine(root, "examples.jsonl"), "{\"instruction\":\"a\",\"output\":\"b\"}\n");
        File.WriteAllText(Path.Combine(root, "README.md"), "# hi");
        Directory.CreateDirectory(Path.Combine(root, "reports", "quality"));
        File.WriteAllText(Path.Combine(root, "reports", "quality", "q.json"), "{\"grade\":\"B\"}");
        return root;
    }

    private static WorkspaceExplorerViewModel Vm(string root, string name = "Test")
    {
        var vm = new WorkspaceExplorerViewModel();
        vm.SetWorkspaceRoot(root, name);
        return vm;
    }

    private static WorkspaceTreeNode FileNode(string relativePath) =>
        new() { RelativePath = relativePath, IsDirectory = false };

    [Fact]
    public void SetWorkspaceRoot_BuildsTree()
    {
        var vm = Vm(NewWorkspace());
        Assert.True(vm.HasWorkspace);
        Assert.NotNull(vm.RootNode);
        Assert.Contains(vm.RootNode!.Children, c => c.Name == "README.md");
    }

    [Fact]
    public async Task OpenText_LoadsContent_NotDirty_IsTextViewer()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "notes.txt"), "hello");
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("notes.txt"));

        Assert.Single(vm.OpenDocuments);
        Assert.Equal("hello", vm.ActiveDocument!.TextContent);
        Assert.False(vm.ActiveDocument.IsDirty);
        Assert.True(vm.IsTextDocument);
        Assert.False(vm.IsNoDocument);
    }

    [Fact]
    public async Task OpenGeneratedReport_IsReadOnly()
    {
        var vm = Vm(NewWorkspace());
        await vm.OpenNodeAsync(FileNode("reports/quality/q.json"));
        Assert.True(vm.ActiveDocument!.IsReadOnly);
        Assert.True(vm.IsTextDocument);   // json renders in the text viewer, read-only
    }

    [Fact]
    public async Task DirtyTracking_ThenSave_WritesAndClears()
    {
        var root = NewWorkspace();
        var path = Path.Combine(root, "a.txt");
        File.WriteAllText(path, "one");
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("a.txt"));
        vm.ActiveDocument!.TextContent = "one two";
        Assert.True(vm.ActiveDocument.IsDirty);

        Assert.Null(vm.SaveActiveDocument());
        Assert.False(vm.ActiveDocument.IsDirty);
        Assert.Equal("one two", File.ReadAllText(path));
    }

    [Fact]
    public async Task OpeningExamples_IsExamplesFile_Editable_NoMutation()
    {
        var root = NewWorkspace();
        var path = Path.Combine(root, "examples.jsonl");
        var original = File.ReadAllText(path);
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("examples.jsonl"));

        Assert.True(vm.IsExamplesFile);
        Assert.False(vm.ActiveDocument!.IsReadOnly);       // editable, explicit-save only
        Assert.Equal(original, File.ReadAllText(path));    // opening never rewrote it
    }

    [Fact]
    public async Task ActiveDocumentIsDatasetFile_OnlyForThisProjectsExamplesJsonl()
    {
        var root = NewWorkspace();  // contains examples.jsonl + README.md
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("examples.jsonl"));
        Assert.True(vm.ActiveDocumentIsDatasetFile(root));                   // saving this = dataset change
        Assert.False(vm.ActiveDocumentIsDatasetFile("C:/some/other/proj"));  // a different project's dataset
        Assert.False(vm.ActiveDocumentIsDatasetFile(null));                  // guard

        await vm.OpenNodeAsync(FileNode("README.md"));
        Assert.False(vm.ActiveDocumentIsDatasetFile(root));                  // a non-dataset file
    }

    [Fact]
    public async Task CreateFile_InsideRoot_Opens_And_RefusesTraversal()
    {
        var root = NewWorkspace();
        var vm = Vm(root);

        Assert.Null(await vm.CreateFileAsync("sub/new.md"));
        Assert.True(File.Exists(Path.Combine(root, "sub", "new.md")));
        Assert.Contains(vm.OpenDocuments, d => d.RelativePath.EndsWith("new.md"));

        Assert.NotNull(await vm.CreateFileAsync("../escape.txt"));    // refused; error surfaced
        Assert.False(File.Exists(Path.Combine(_tempRoot, "escape.txt")));
    }

    [Fact]
    public void CreateFolder_InsideRoot()
    {
        var root = NewWorkspace();
        Assert.Null(Vm(root).CreateFolder("assets/images"));
        Assert.True(Directory.Exists(Path.Combine(root, "assets", "images")));
    }

    [Fact]
    public async Task CloseDocument_ReactivatesNeighbour()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "a.txt"), "a");
        File.WriteAllText(Path.Combine(root, "b.txt"), "b");
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("a.txt"));
        var a = vm.ActiveDocument!;
        await vm.OpenNodeAsync(FileNode("b.txt"));
        Assert.Equal(2, vm.OpenDocuments.Count);

        vm.CloseDocument(vm.ActiveDocument!);   // close b (active)
        Assert.Single(vm.OpenDocuments);
        Assert.Same(a, vm.ActiveDocument);      // neighbour reactivated
    }

    [Fact]
    public async Task SwitchingRoot_ClearsOpenDocuments()
    {
        var root1 = NewWorkspace();
        File.WriteAllText(Path.Combine(root1, "a.txt"), "a");
        var vm = Vm(root1);
        await vm.OpenNodeAsync(FileNode("a.txt"));
        Assert.Single(vm.OpenDocuments);

        vm.SetWorkspaceRoot(NewWorkspace(), "Two");
        Assert.Empty(vm.OpenDocuments);
        Assert.Null(vm.ActiveDocument);
    }

    [Fact]
    public async Task SameRoot_KeepsOpenTabs()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "a.txt"), "a");
        var vm = Vm(root);
        await vm.OpenNodeAsync(FileNode("a.txt"));

        vm.SetWorkspaceRoot(root, "Test");   // unchanged root -> no rebuild
        Assert.Single(vm.OpenDocuments);
    }

    [Fact]
    public async Task OpenImage_IsImageViewer_NoText()
    {
        var root = NewWorkspace();
        Directory.CreateDirectory(Path.Combine(root, "assets"));
        File.WriteAllBytes(Path.Combine(root, "assets", "a.png"), new byte[] { 0x89, 0x50, 0x4E, 0x47 });
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("assets/a.png"));
        Assert.True(vm.IsImageDocument);
        Assert.False(vm.IsTextDocument);
        Assert.Equal(string.Empty, vm.ActiveDocument!.TextContent);
    }

    [Fact]
    public void Reset_ClearsEverything()
    {
        var vm = Vm(NewWorkspace());
        vm.Reset();
        Assert.False(vm.HasWorkspace);
        Assert.Null(vm.RootNode);
        Assert.Empty(vm.OpenDocuments);
    }

    [Fact]
    public async Task OpenDirectory_IsIgnored()
    {
        var vm = Vm(NewWorkspace());
        await vm.OpenNodeAsync(new WorkspaceTreeNode { RelativePath = "reports", IsDirectory = true });
        Assert.Empty(vm.OpenDocuments);
        Assert.True(vm.IsNoDocument);
    }

    // ---- Polish (v1.2.5): chip label + active-tab flag ---------------------------

    [Theory]
    [InlineData(".jsonl", "JSONL")]
    [InlineData(".md", "MD")]
    [InlineData(".JSON", "JSON")]
    [InlineData("", "")]
    public void TreeNode_ChipLabel_UppercasesWithoutDot(string extension, string expected)
    {
        Assert.Equal(expected, new WorkspaceTreeNode { Extension = extension }.ChipLabel);
    }

    [Fact]
    public async Task ActiveDocument_ExactlyOneTabIsActive()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "a.txt"), "a");
        File.WriteAllText(Path.Combine(root, "b.txt"), "b");
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("a.txt"));
        var a = vm.ActiveDocument!;
        await vm.OpenNodeAsync(FileNode("b.txt"));
        var b = vm.ActiveDocument!;

        Assert.False(a.IsActive);
        Assert.True(b.IsActive);
        Assert.Single(vm.OpenDocuments, d => d.IsActive);

        vm.ActiveDocument = a;               // switch back
        Assert.True(a.IsActive);
        Assert.False(b.IsActive);
        Assert.Single(vm.OpenDocuments, d => d.IsActive);
    }

    [Fact]
    public async Task ClosingActive_ReactivatedNeighbour_BecomesActive()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "a.txt"), "a");
        File.WriteAllText(Path.Combine(root, "b.txt"), "b");
        var vm = Vm(root);

        await vm.OpenNodeAsync(FileNode("a.txt"));
        var a = vm.ActiveDocument!;
        await vm.OpenNodeAsync(FileNode("b.txt"));

        vm.CloseDocument(vm.ActiveDocument!);   // close b (active)
        Assert.Same(a, vm.ActiveDocument);
        Assert.True(a.IsActive);
    }
}
