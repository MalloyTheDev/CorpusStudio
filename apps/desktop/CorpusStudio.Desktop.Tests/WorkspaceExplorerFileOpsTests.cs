using System;
using System.IO;
using System.Threading.Tasks;
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

    /// <summary>Scriptable dialog seam: returns a fixed prompt answer + confirm verdict and records the
    /// last surfaced message, so the command-level tests can assert prompt/confirm/error behaviour (#223).</summary>
    private sealed class ScriptedDialogService : IDialogService
    {
        private readonly string? _promptAnswer;
        private readonly bool _confirm;
        public ScriptedDialogService(string? promptAnswer, bool confirm)
        {
            _promptAnswer = promptAnswer;
            _confirm = confirm;
        }

        public string? LastShownMessage { get; private set; }
        public int ShownCount { get; private set; }
        public Task<bool> ConfirmAsync(string message, string title, DialogButtons buttons = DialogButtons.YesNo, DialogSeverity severity = DialogSeverity.Question, bool defaultAffirmative = false) => Task.FromResult(_confirm);
        public Task ShowAsync(string message, string title, DialogSeverity severity = DialogSeverity.Information)
        {
            LastShownMessage = message;
            ShownCount++;
            return Task.CompletedTask;
        }
        public Task<string?> PromptAsync(string title, string message, string defaultValue = "") => Task.FromResult(_promptAnswer);
    }

    private static WorkspaceTreeNode FileNode(string root, string name) => new()
    {
        Name = name,
        FullPath = Path.Combine(root, name),
        RelativePath = name,
    };

    [Fact]
    public async Task NewFileCommand_CreatesFile_WhenPromptReturnsAName()
    {
        var root = NewWorkspace();
        try
        {
            var dialogs = new ScriptedDialogService("notes.md", confirm: true);
            var vm = new WorkspaceExplorerViewModel(dialogs: dialogs);
            vm.SetWorkspaceRoot(root, "ws");

            await vm.PromptNewFileAsync();

            Assert.True(File.Exists(Path.Combine(root, "notes.md")));
            Assert.Equal(0, dialogs.ShownCount); // success surfaces no error dialog
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task NewFileCommand_Cancelled_CreatesNothing()
    {
        var root = NewWorkspace();
        try
        {
            var dialogs = new ScriptedDialogService(promptAnswer: null, confirm: true); // user cancelled the prompt
            var vm = new WorkspaceExplorerViewModel(dialogs: dialogs);
            vm.SetWorkspaceRoot(root, "ws");

            await vm.PromptNewFileAsync();

            Assert.Empty(Directory.GetFiles(root));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task NewFolderCommand_CreatesFolder_WhenPromptReturnsAName()
    {
        var root = NewWorkspace();
        try
        {
            var dialogs = new ScriptedDialogService("docs", confirm: true);
            var vm = new WorkspaceExplorerViewModel(dialogs: dialogs);
            vm.SetWorkspaceRoot(root, "ws");

            await vm.PromptNewFolderAsync();

            Assert.True(Directory.Exists(Path.Combine(root, "docs")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task RenameCommand_RenamesOnDisk_WhenPromptReturnsANewName()
    {
        var root = NewWorkspace();
        try
        {
            File.WriteAllText(Path.Combine(root, "a.txt"), "x");
            var dialogs = new ScriptedDialogService("b.txt", confirm: true);
            var vm = new WorkspaceExplorerViewModel(dialogs: dialogs);
            vm.SetWorkspaceRoot(root, "ws");

            await vm.PromptRenameNodeAsync(FileNode(root, "a.txt"));

            Assert.False(File.Exists(Path.Combine(root, "a.txt")));
            Assert.True(File.Exists(Path.Combine(root, "b.txt")));
            Assert.Equal(0, dialogs.ShownCount);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task RenameCommand_Cancelled_LeavesFileUntouched()
    {
        var root = NewWorkspace();
        try
        {
            File.WriteAllText(Path.Combine(root, "a.txt"), "x");
            var dialogs = new ScriptedDialogService(promptAnswer: null, confirm: true);
            var vm = new WorkspaceExplorerViewModel(dialogs: dialogs);
            vm.SetWorkspaceRoot(root, "ws");

            await vm.PromptRenameNodeAsync(FileNode(root, "a.txt"));

            Assert.True(File.Exists(Path.Combine(root, "a.txt"))); // unchanged
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task DeleteCommand_DeletesOnConfirm_KeepsOnDecline()
    {
        var root = NewWorkspace();
        try
        {
            var keep = Path.Combine(root, "keep.txt");
            File.WriteAllText(keep, "x");
            var vmDecline = new WorkspaceExplorerViewModel(dialogs: new ScriptedDialogService(null, confirm: false));
            vmDecline.SetWorkspaceRoot(root, "ws");
            await vmDecline.ConfirmDeleteNodeAsync(FileNode(root, "keep.txt"));
            Assert.True(File.Exists(keep)); // declined → still there

            var vmConfirm = new WorkspaceExplorerViewModel(dialogs: new ScriptedDialogService(null, confirm: true));
            vmConfirm.SetWorkspaceRoot(root, "ws");
            await vmConfirm.ConfirmDeleteNodeAsync(FileNode(root, "keep.txt"));
            Assert.False(File.Exists(keep)); // confirmed → gone
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task RenameCommand_DatasetCoreFile_SurfacesErrorAndDoesNotRename()
    {
        var root = NewWorkspace();
        try
        {
            File.WriteAllText(Path.Combine(root, "examples.jsonl"), "{}");
            var dialogs = new ScriptedDialogService("renamed.jsonl", confirm: true);
            var vm = new WorkspaceExplorerViewModel(dialogs: dialogs);
            vm.SetWorkspaceRoot(root, "ws");
            var core = new WorkspaceTreeNode
            {
                Name = "examples.jsonl",
                FullPath = Path.Combine(root, "examples.jsonl"),
                RelativePath = "examples.jsonl",
                IsDatasetCoreFile = true,
            };

            await vm.PromptRenameNodeAsync(core);

            Assert.Equal(1, dialogs.ShownCount);
            Assert.Contains("core file", dialogs.LastShownMessage);
            Assert.True(File.Exists(Path.Combine(root, "examples.jsonl"))); // not renamed
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
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
