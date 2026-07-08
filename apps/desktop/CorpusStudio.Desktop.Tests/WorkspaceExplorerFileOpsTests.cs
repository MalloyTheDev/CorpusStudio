using System;
using System.IO;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Issue #200: Explorer tree Rename/Delete file operations. Guards the safety rules — stay within
/// the workspace, reject the root / clobbers / invalid names / missing sources — and the VM's refusal to
/// rename or delete dataset core files.</summary>
public sealed class WorkspaceExplorerFileOpsTests
{
    private static string NewWorkspace()
    {
        var dir = Path.Combine(Path.GetTempPath(), "cs_ws_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void RenamePath_RenamesFile_AndRejectsClobberAndInvalidNames()
    {
        var root = NewWorkspace();
        try
        {
            var svc = new WorkspaceExplorerService();
            var a = Path.Combine(root, "a.txt");
            File.WriteAllText(a, "x");
            File.WriteAllText(Path.Combine(root, "b.txt"), "y");

            Assert.False(svc.RenamePath(root, a, "b.txt").Ok); // clobber
            Assert.False(svc.RenamePath(root, a, "sub/c.txt").Ok); // path separator
            Assert.False(svc.RenamePath(root, a, "..").Ok); // traversal name
            Assert.False(svc.RenamePath(root, a, "  ").Ok); // empty

            var ok = svc.RenamePath(root, a, "c.txt");
            Assert.True(ok.Ok);
            Assert.False(File.Exists(a));
            Assert.True(File.Exists(Path.Combine(root, "c.txt")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void RenamePath_RefusesRoot_Outside_AndMissing()
    {
        var root = NewWorkspace();
        var outside = Path.Combine(Path.GetTempPath(), "cs_out_" + Guid.NewGuid().ToString("N") + ".txt");
        File.WriteAllText(outside, "z");
        try
        {
            var svc = new WorkspaceExplorerService();
            Assert.False(svc.RenamePath(root, root, "x").Ok); // the root itself
            Assert.False(svc.RenamePath(root, Path.Combine(root, "missing.txt"), "x").Ok); // missing source
            Assert.False(svc.RenamePath(root, outside, "x").Ok); // escapes the workspace
        }
        finally
        {
            File.Delete(outside);
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void DeletePath_DeletesFileAndFolderRecursively_ButRefusesRootAndMissing()
    {
        var root = NewWorkspace();
        try
        {
            var svc = new WorkspaceExplorerService();
            var file = Path.Combine(root, "f.txt");
            File.WriteAllText(file, "x");
            var sub = Path.Combine(root, "sub");
            Directory.CreateDirectory(sub);
            File.WriteAllText(Path.Combine(sub, "g.txt"), "y");

            Assert.True(svc.DeletePath(root, file).Ok);
            Assert.False(File.Exists(file));
            Assert.True(svc.DeletePath(root, sub).Ok); // recursive folder delete
            Assert.False(Directory.Exists(sub));
            Assert.False(svc.DeletePath(root, root).Ok); // the root itself
            Assert.False(svc.DeletePath(root, Path.Combine(root, "nope")).Ok); // missing
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void RenameNode_And_DeleteNode_RefuseDatasetCoreFiles()
    {
        var root = NewWorkspace();
        try
        {
            var vm = new WorkspaceExplorerViewModel();
            vm.SetWorkspaceRoot(root, "ws");
            var core = new WorkspaceTreeNode
            {
                Name = "examples.jsonl",
                FullPath = Path.Combine(root, "examples.jsonl"),
                IsDatasetCoreFile = true,
            };

            Assert.Contains("core file", vm.RenameNode(core, "x.jsonl"));
            Assert.Contains("core file", vm.DeleteNode(core));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
