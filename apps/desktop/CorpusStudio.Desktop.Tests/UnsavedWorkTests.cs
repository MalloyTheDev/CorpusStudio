using System;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Unsaved-work tracking + the DraftText cross-project leak fix (audit item 4).
/// The leak (a draft typed against project A being written into project B) is a data
/// bug; IsDraftDirty/HasUnsavedWork drive the switch/close "discard?" prompts (View code).</summary>
public sealed class UnsavedWorkTests
{
    private static DatasetProjectListItem Project(string id, string schema = "instruction") =>
        new(new DatasetProject(id, id, schema, new DateTime(2026, 1, 1), new DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    [Fact]
    public void Draft_IsCleanInitially_DirtyAfterTyping()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.WritingStudio.IsDraftDirty);
        Assert.False(vm.HasUnsavedWork);

        vm.WritingStudio.DraftText = "{\"instruction\":\"typed\",\"output\":\"x\"}";

        Assert.True(vm.WritingStudio.IsDraftDirty);
        Assert.True(vm.HasUnsavedWork);
    }

    [Fact]
    public void LoadDraft_SetsCleanBaseline()
    {
        var vm = new MainWindowViewModel();
        vm.WritingStudio.DraftText = "dirty";
        Assert.True(vm.WritingStudio.IsDraftDirty);

        vm.WritingStudio.LoadDraft("{\"instruction\":\"loaded\",\"output\":\"y\"}");

        Assert.False(vm.WritingStudio.IsDraftDirty);            // a loaded draft is not "unsaved work"
        Assert.Contains("loaded", vm.WritingStudio.DraftText);
    }

    [Fact]
    public void MarkDraftClean_ClearsDirty()
    {
        var vm = new MainWindowViewModel();
        vm.WritingStudio.DraftText = "edited";
        Assert.True(vm.WritingStudio.IsDraftDirty);

        vm.WritingStudio.MarkDraftClean();

        Assert.False(vm.WritingStudio.IsDraftDirty);
    }

    [Fact]
    public void SelectProject_ResetsDraft_SoItCannotLeakIntoTheNewProject()
    {
        var vm = new MainWindowViewModel();
        vm.WritingStudio.DraftText = "{\"instruction\":\"typed in project A\",\"output\":\"x\"}";
        Assert.True(vm.WritingStudio.IsDraftDirty);

        vm.SelectProject(Project("b"), "Instruction");

        Assert.False(vm.WritingStudio.IsDraftDirty);                              // reset to a clean template
        Assert.DoesNotContain("typed in project A", vm.WritingStudio.DraftText);  // A's draft did NOT carry over
    }

    [Fact]
    public void HasUnsavedWork_TrueWhenAnExplorerDocumentIsDirty()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.HasUnsavedWork);

        var doc = new OpenWorkspaceDocument();
        doc.MarkClean("a");
        doc.TextContent = "b"; // an unsaved edit

        vm.Explorer.OpenDocuments.Add(doc);

        Assert.True(vm.Explorer.HasDirtyDocuments);
        Assert.True(vm.HasUnsavedWork);
    }
}
