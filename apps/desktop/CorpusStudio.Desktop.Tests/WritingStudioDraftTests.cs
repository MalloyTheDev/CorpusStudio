using System.Text.Json;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The Writing Studio structured Instruction/Input/Output form is a live projection of the
/// single <c>DraftText</c> JSON buffer (so save/validate/dirty-tracking are unchanged and the WPF head
/// keeps binding the raw buffer). These lock the projection + the honesty gate that keeps a non-
/// instruction / malformed draft on the raw editor.</summary>
public sealed class WritingStudioDraftTests
{
    private static string Field(string draftJson, string key)
        => JsonDocument.Parse(draftJson).RootElement.GetProperty(key).GetString() ?? "";

    [Fact]
    public void Fields_ProjectFromTheDraftJson()
    {
        var vm = new WritingStudioViewModel();
        vm.LoadDraft("""{"instruction": "Reset my password", "input": "", "output": "Click Forgot password."}""");

        Assert.True(vm.IsInstructionShapedDraft);
        Assert.Equal("Reset my password", vm.DraftInstruction);
        Assert.Equal("", vm.DraftInput);
        Assert.Equal("Click Forgot password.", vm.DraftOutput);
    }

    [Fact]
    public void SettingAField_WritesBackToDraftText_AndPreservesSiblings()
    {
        var vm = new WritingStudioViewModel();
        vm.LoadDraft("""{"instruction": "old", "input": "ctx", "output": "out"}""");

        vm.DraftInstruction = "new instruction";

        // DraftText is the single source of truth save/validate read — the edit round-trips into it,
        // and the sibling fields are untouched.
        Assert.Equal("new instruction", Field(vm.DraftText, "instruction"));
        Assert.Equal("ctx", Field(vm.DraftText, "input"));
        Assert.Equal("out", Field(vm.DraftText, "output"));
        Assert.Equal("new instruction", vm.DraftInstruction);
    }

    [Fact]
    public void EditingAField_MarksTheDraftDirty()
    {
        var vm = new WritingStudioViewModel();
        vm.LoadDraft("""{"instruction": "i", "input": "", "output": "o"}""");
        Assert.False(vm.IsDraftDirty);           // freshly loaded = clean baseline

        vm.DraftOutput = "a better answer";
        Assert.True(vm.IsDraftDirty);
    }

    [Fact]
    public void NonInstructionDraft_StaysOnTheRawEditor_FieldsEmpty()
    {
        var vm = new WritingStudioViewModel();
        // A chat-schema draft has "messages", not instruction/output — the structured form must NOT
        // claim it (editing instruction/output would corrupt a shape it can't round-trip).
        vm.LoadDraft("""{"messages": [{"role": "user", "content": "hi"}]}""");

        Assert.False(vm.IsInstructionShapedDraft);
        Assert.Equal("", vm.DraftInstruction);
        Assert.Equal("", vm.DraftOutput);
    }

    [Fact]
    public void MalformedDraft_IsNotStructured_AndFieldSetIsANoOp()
    {
        var vm = new WritingStudioViewModel();
        vm.LoadDraft("not json at all {");

        Assert.False(vm.IsInstructionShapedDraft);
        Assert.Equal("", vm.DraftInstruction);

        vm.DraftInstruction = "x";               // must not rewrite an unparseable buffer
        Assert.Equal("not json at all {", vm.DraftText);
    }
}
