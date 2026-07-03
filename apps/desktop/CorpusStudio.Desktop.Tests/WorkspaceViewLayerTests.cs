using System;
using System.IO;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Workspace System view layer (v1.2.4, slice 3a): the shell-mode toggle and the
/// Start Center's Recent Workspaces view-model. Pure logic — the XAML is verified by
/// running the app.</summary>
public sealed class WorkspaceViewLayerTests : IDisposable
{
    private readonly string _tempRoot;

    public WorkspaceViewLayerTests()
    {
        _tempRoot = Path.Combine(Path.GetTempPath(), "cs-vl-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true); }
        catch (IOException) { /* best-effort */ }
    }

    private RecentWorkspaceService Service(Func<string, bool>? exists = null) =>
        new(Path.Combine(_tempRoot, "app-" + Guid.NewGuid().ToString("N")), exists);

    private string P(string name) => Path.Combine(_tempRoot, name);

    // ---- Shell mode --------------------------------------------------------------

    [Fact]
    public void Shell_DefaultsToStudio_SoAppOpensAsToday()
    {
        var vm = new MainWindowViewModel();
        Assert.True(vm.IsStudio);
        Assert.False(vm.IsStartCenter);
        Assert.False(vm.IsFiles);
    }

    [Fact]
    public void Shell_TogglesBetweenModes()
    {
        var vm = new MainWindowViewModel();

        vm.ShowStartCenter();
        Assert.True(vm.IsStartCenter);
        Assert.False(vm.IsStudio);
        Assert.False(vm.IsFiles);

        vm.ShowFiles();
        Assert.True(vm.IsFiles);
        Assert.False(vm.IsStartCenter);

        vm.ShowStudio();
        Assert.True(vm.IsStudio);
        Assert.False(vm.IsFiles);
    }

    // ---- Start Center VM ---------------------------------------------------------

    [Fact]
    public void StartCenter_Empty_HasNoRecents()
    {
        var vm = new StartCenterViewModel(Service());
        Assert.False(vm.HasRecents);
        Assert.Equal("No recent workspaces", vm.RecentCountLabel);
    }

    [Fact]
    public void StartCenter_RecordOpened_AddsMostRecentFirst()
    {
        var vm = new StartCenterViewModel(Service(_ => true));
        vm.RecordOpened(P("a"), "Alpha", "instruction", "2026-07-03T00:00:00Z");
        vm.RecordOpened(P("b"), "Beta", "chat", "2026-07-03T01:00:00Z");

        Assert.True(vm.HasRecents);
        Assert.Equal(2, vm.Recents.Count);
        Assert.Equal("Beta", vm.Recents[0].Name);   // most recent first
    }

    [Fact]
    public void StartCenter_CountLabel_SingularAndPlural()
    {
        var vm = new StartCenterViewModel(Service(_ => true));
        vm.RecordOpened(P("a"), "A", "instruction", "2026-07-03T00:00:00Z");
        Assert.Equal("1 workspace", vm.RecentCountLabel);
        vm.RecordOpened(P("b"), "B", "chat", "2026-07-03T01:00:00Z");
        Assert.Equal("2 workspaces", vm.RecentCountLabel);
    }

    [Fact]
    public void StartCenter_MissingPath_FlaggedNotDropped()
    {
        var vm = new StartCenterViewModel(Service(_ => false));   // every folder reports gone
        vm.RecordOpened(P("gone"), "Gone", "instruction", "2026-07-03T00:00:00Z");

        Assert.Single(vm.Recents);
        Assert.True(vm.Recents[0].IsMissing);
    }

    [Fact]
    public void StartCenter_Pin_Unpin_Remove()
    {
        var vm = new StartCenterViewModel(Service(_ => true));
        var p = P("x");
        vm.RecordOpened(p, "X", "instruction", "2026-07-03T00:00:00Z");

        vm.SetPinned(p, true);
        Assert.True(vm.Recents[0].IsPinned);

        vm.SetPinned(p, false);
        Assert.False(vm.Recents[0].IsPinned);

        vm.Remove(p);
        Assert.Empty(vm.Recents);
    }

    [Fact]
    public void StartCenter_RecordOpened_PersistsAcrossInstances()
    {
        var svc = Service(_ => true);
        new StartCenterViewModel(svc).RecordOpened(P("keep"), "Keep", "code", "2026-07-03T00:00:00Z");

        // A fresh VM over the same registry sees it.
        var reopened = new StartCenterViewModel(svc);
        Assert.Single(reopened.Recents);
        Assert.Equal("Keep", reopened.Recents[0].Name);
    }

    // ---- Display item ------------------------------------------------------------

    [Theory]
    [InlineData("instruction", "IN")]
    [InlineData("chat", "CH")]
    [InlineData("image_caption", "IM")]
    [InlineData(null, "WS")]
    [InlineData("", "WS")]
    public void DisplayItem_SchemaTag(string? schema, string expected)
    {
        var item = new RecentWorkspaceDisplayItem(new RecentWorkspaceRecord { Path = "C:/ws/x", Name = "X", SchemaId = schema });
        Assert.Equal(expected, item.SchemaTag);
    }

    [Fact]
    public void DisplayItem_PinTitle_Toggles()
    {
        Assert.Equal("Pin to top", new RecentWorkspaceDisplayItem(new RecentWorkspaceRecord { Path = "a", IsPinned = false }).PinTitle);
        Assert.Equal("Unpin", new RecentWorkspaceDisplayItem(new RecentWorkspaceRecord { Path = "a", IsPinned = true }).PinTitle);
    }
}
