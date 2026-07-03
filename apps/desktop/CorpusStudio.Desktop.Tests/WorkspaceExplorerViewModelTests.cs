using System;
using System.IO;
using System.Linq;
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
    public void OpenText_LoadsContent_NotDirty_IsTextViewer()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "notes.txt"), "hello");
        var vm = Vm(root);

        vm.OpenNode(FileNode("notes.txt"));

        Assert.Single(vm.OpenDocuments);
        Assert.Equal("hello", vm.ActiveDocument!.TextContent);
        Assert.False(vm.ActiveDocument.IsDirty);
        Assert.True(vm.IsTextDocument);
        Assert.False(vm.IsNoDocument);
    }

    [Fact]
    public void OpenGeneratedReport_IsReadOnly()
    {
        var vm = Vm(NewWorkspace());
        vm.OpenNode(FileNode("reports/quality/q.json"));
        Assert.True(vm.ActiveDocument!.IsReadOnly);
        Assert.True(vm.IsTextDocument);   // json renders in the text viewer, read-only
    }

    [Fact]
    public void DirtyTracking_ThenSave_WritesAndClears()
    {
        var root = NewWorkspace();
        var path = Path.Combine(root, "a.txt");
        File.WriteAllText(path, "one");
        var vm = Vm(root);

        vm.OpenNode(FileNode("a.txt"));
        vm.ActiveDocument!.TextContent = "one two";
        Assert.True(vm.ActiveDocument.IsDirty);

        Assert.Null(vm.SaveActiveDocument());
        Assert.False(vm.ActiveDocument.IsDirty);
        Assert.Equal("one two", File.ReadAllText(path));
    }

    [Fact]
    public void OpeningExamples_IsExamplesFile_Editable_NoMutation()
    {
        var root = NewWorkspace();
        var path = Path.Combine(root, "examples.jsonl");
        var original = File.ReadAllText(path);
        var vm = Vm(root);

        vm.OpenNode(FileNode("examples.jsonl"));

        Assert.True(vm.IsExamplesFile);
        Assert.False(vm.ActiveDocument!.IsReadOnly);       // editable, explicit-save only
        Assert.Equal(original, File.ReadAllText(path));    // opening never rewrote it
    }

    [Fact]
    public void CreateFile_InsideRoot_Opens_And_RefusesTraversal()
    {
        var root = NewWorkspace();
        var vm = Vm(root);

        Assert.Null(vm.CreateFile("sub/new.md"));
        Assert.True(File.Exists(Path.Combine(root, "sub", "new.md")));
        Assert.Contains(vm.OpenDocuments, d => d.RelativePath.EndsWith("new.md"));

        Assert.NotNull(vm.CreateFile("../escape.txt"));    // refused; error surfaced
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
    public void CloseDocument_ReactivatesNeighbour()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "a.txt"), "a");
        File.WriteAllText(Path.Combine(root, "b.txt"), "b");
        var vm = Vm(root);

        vm.OpenNode(FileNode("a.txt"));
        var a = vm.ActiveDocument!;
        vm.OpenNode(FileNode("b.txt"));
        Assert.Equal(2, vm.OpenDocuments.Count);

        vm.CloseDocument(vm.ActiveDocument!);   // close b (active)
        Assert.Single(vm.OpenDocuments);
        Assert.Same(a, vm.ActiveDocument);      // neighbour reactivated
    }

    [Fact]
    public void SwitchingRoot_ClearsOpenDocuments()
    {
        var root1 = NewWorkspace();
        File.WriteAllText(Path.Combine(root1, "a.txt"), "a");
        var vm = Vm(root1);
        vm.OpenNode(FileNode("a.txt"));
        Assert.Single(vm.OpenDocuments);

        vm.SetWorkspaceRoot(NewWorkspace(), "Two");
        Assert.Empty(vm.OpenDocuments);
        Assert.Null(vm.ActiveDocument);
    }

    [Fact]
    public void SameRoot_KeepsOpenTabs()
    {
        var root = NewWorkspace();
        File.WriteAllText(Path.Combine(root, "a.txt"), "a");
        var vm = Vm(root);
        vm.OpenNode(FileNode("a.txt"));

        vm.SetWorkspaceRoot(root, "Test");   // unchanged root -> no rebuild
        Assert.Single(vm.OpenDocuments);
    }

    [Fact]
    public void OpenImage_IsImageViewer_NoText()
    {
        var root = NewWorkspace();
        Directory.CreateDirectory(Path.Combine(root, "assets"));
        File.WriteAllBytes(Path.Combine(root, "assets", "a.png"), new byte[] { 0x89, 0x50, 0x4E, 0x47 });
        var vm = Vm(root);

        vm.OpenNode(FileNode("assets/a.png"));
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
    public void OpenDirectory_IsIgnored()
    {
        var vm = Vm(NewWorkspace());
        vm.OpenNode(new WorkspaceTreeNode { RelativePath = "reports", IsDirectory = true });
        Assert.Empty(vm.OpenDocuments);
        Assert.True(vm.IsNoDocument);
    }
}
