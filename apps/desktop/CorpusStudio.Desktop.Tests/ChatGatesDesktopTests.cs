using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Chat Gates desktop surface (v1.3 M2): the "Run chat gates" button is contextual —
/// shown only for chat datasets, where conversation-structure gates apply.</summary>
public sealed class ChatGatesDesktopTests
{
    [Fact]
    public void IsChatProject_TrueOnlyForChatSchema()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.IsChatProject);   // no active project

        vm.AddProject("p1", "Chat", "chat", "Chat", null);
        Assert.True(vm.IsChatProject);

        vm.AddProject("p2", "Instr", "instruction", "Instruction", null);
        Assert.False(vm.IsChatProject);   // switched to a non-chat schema
    }
}
