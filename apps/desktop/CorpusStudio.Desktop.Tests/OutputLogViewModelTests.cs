using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class OutputLogViewModelTests
{
    private static EngineLogEntry Entry(string status = "ok") =>
        new() { Command = "gate-run", Status = status };

    // ---- EngineLogEntry model ----------------------------------------------------

    [Theory]
    [InlineData("ok", "✓", "#16A34A")]
    [InlineData("error", "✕", "#DC2626")]
    [InlineData("cancelled", "⊘", "#64748B")]
    [InlineData("weird", "•", "#64748B")]
    public void EngineLogEntry_StatusIconAndColor(string status, string icon, string color)
    {
        var e = new EngineLogEntry { Status = status };
        Assert.Equal(icon, e.StatusIcon);
        Assert.Equal(color, e.StatusColor);
    }

    [Fact]
    public void FromInvocation_ExtractsVerbAndArgs_OkHasNoDetail()
    {
        var e = EngineLogEntry.FromInvocation(
            ["gate-run", "examples.jsonl", "instruction", "--scope", "dataset"],
            exitCode: 0, durationMs: 42, stderr: "a warning on stderr", timestamp: "12:00:00");

        Assert.Equal("gate-run", e.Command);
        Assert.Equal("examples.jsonl instruction --scope dataset", e.ArgsSummary);
        Assert.Equal("ok", e.Status);
        Assert.Equal("42 ms", e.DurationLabel);
        Assert.False(e.HasDetail);           // stderr is not surfaced for a successful command
        Assert.Equal(string.Empty, e.Detail);
    }

    [Fact]
    public void FromInvocation_NonZeroExit_IsErrorWithStderrDetail()
    {
        var e = EngineLogEntry.FromInvocation(
            ["quality", "missing.jsonl"], exitCode: 1, durationMs: 5,
            stderr: "Traceback...\nValueError: non-object line", timestamp: "12:00:01");

        Assert.Equal("error", e.Status);
        Assert.True(e.HasDetail);
        Assert.Contains("non-object line", e.Detail);
    }

    [Fact]
    public void FromInvocation_Cancelled_OverridesExitCode()
    {
        var e = EngineLogEntry.FromInvocation(
            ["ai-assist", "--backend", "ollama"], exitCode: -1, durationMs: 900,
            stderr: null, timestamp: "12:00:02", cancelled: true);

        Assert.Equal("cancelled", e.Status);
        Assert.Equal("ai-assist", e.Command);
        Assert.False(e.HasDetail);
    }

    [Fact]
    public void FromInvocation_TruncatesLongArgs()
    {
        var longPath = new string('x', 200);
        var e = EngineLogEntry.FromInvocation(["export", longPath], 0, 1, null, "12:00:03");
        Assert.EndsWith("…", e.ArgsSummary);
        Assert.True(e.ArgsSummary.Length < 200);
    }

    [Fact]
    public void FromInvocation_EmptyArgv_UsesEnginePlaceholder()
    {
        var e = EngineLogEntry.FromInvocation([], 0, 1, null, "12:00:04");
        Assert.Equal("(engine)", e.Command);
        Assert.Equal(string.Empty, e.ArgsSummary);
    }

    // ---- EngineLogEntry Dashboard "Recent Activity" projection -------------------

    [Theory]
    [InlineData("quality", "ok", "IcoBroom")]
    [InlineData("import-preview", "ok", "IcoImport")]
    [InlineData("import-convert", "ok", "IcoImport")]
    [InlineData("gate-run", "ok", "IcoCheckCircleFill")]
    [InlineData("validate", "ok", "IcoCheckCircleFill")]
    [InlineData("capture-version", "ok", "IcoGitCommit")]
    [InlineData("split", "ok", "IcoListDashes")]
    [InlineData("export", "ok", "IcoListDashes")]
    [InlineData("new-project", "ok", "IcoListDashes")]
    public void ActivityGlyphKey_MapsRealVerbToGlyph(string command, string status, string expected)
    {
        var e = new EngineLogEntry { Command = command, Status = status };
        Assert.Equal(expected, e.ActivityGlyphKey);
    }

    [Fact]
    public void ActivityGlyphKey_FailedCommand_SurfacesWarning_RegardlessOfVerb()
    {
        // A failed run is flagged with a warning glyph even for a verb that otherwise has its own icon.
        Assert.Equal("IcoWarningFill", new EngineLogEntry { Command = "quality", Status = "error" }.ActivityGlyphKey);
        Assert.Equal("IcoWarningFill", new EngineLogEntry { Command = "gate-run", Status = "error" }.ActivityGlyphKey);
    }

    [Fact]
    public void ActivityGlyphKey_CancelledKeepsVerbGlyph()
    {
        // Cancellation is not a failure, so the verb glyph still applies (only "error" overrides).
        Assert.Equal("IcoBroom", new EngineLogEntry { Command = "quality", Status = "cancelled" }.ActivityGlyphKey);
    }

    [Theory]
    [InlineData("quality", "ok", "Ran quality check")]
    [InlineData("import-convert", "ok", "Imported rows")]
    [InlineData("gate-run", "ok", "Ran gates")]
    [InlineData("validate", "ok", "Validated draft")]
    [InlineData("capture-version", "ok", "Captured version")]
    [InlineData("quality", "error", "Ran quality check — failed")]
    [InlineData("gate-run", "cancelled", "Ran gates — cancelled")]
    public void ActivityMessage_RendersRealVerbAndOutcome(string command, string status, string expected)
    {
        var e = new EngineLogEntry { Command = command, Status = status };
        Assert.Equal(expected, e.ActivityMessage);
    }

    [Fact]
    public void ActivityMessage_UnknownVerb_ShownVerbatim_NeverFabricated()
    {
        var e = new EngineLogEntry { Command = "some-new-verb", Status = "ok" };
        Assert.Equal("some-new-verb", e.ActivityMessage);
    }

    [Fact]
    public void ActivityMeta_CombinesRealTimestampAndDuration()
    {
        Assert.Equal("12:00:03 · 42 ms", new EngineLogEntry { Timestamp = "12:00:03", DurationMs = 42 }.ActivityMeta);
        Assert.Equal("12:00:03", new EngineLogEntry { Timestamp = "12:00:03", DurationMs = 0 }.ActivityMeta);
        Assert.Equal("42 ms", new EngineLogEntry { Timestamp = "", DurationMs = 42 }.ActivityMeta);
        Assert.Equal(string.Empty, new EngineLogEntry { Timestamp = "", DurationMs = 0 }.ActivityMeta);
    }

    [Fact]
    public void FromInvocation_FeedsActivityProjection_FromRealData()
    {
        // The projection reads only the entry's own fields — a real invocation flows straight through.
        var e = EngineLogEntry.FromInvocation(
            ["quality", "examples.jsonl"], exitCode: 0, durationMs: 8, stderr: null, timestamp: "09:15:00");
        Assert.Equal("IcoBroom", e.ActivityGlyphKey);
        Assert.Equal("Ran quality check", e.ActivityMessage);
        Assert.Equal("09:15:00 · 8 ms", e.ActivityMeta);
    }

    // ---- MainWindowViewModel Output panel ----------------------------------------

    [Fact]
    public void AppendEngineLog_AddsAndTogglesEmptyState()
    {
        var vm = new MainWindowViewModel();
        Assert.True(vm.IsNoOutput);
        Assert.False(vm.HasOutput);

        vm.AppendEngineLog(Entry());
        Assert.False(vm.IsNoOutput);
        Assert.True(vm.HasOutput);
        Assert.Single(vm.OutputLog);
        Assert.Contains("1 engine command this session", vm.OutputSummary);
    }

    [Fact]
    public void AppendEngineLog_RingBuffer_CapsAtMaxDroppingOldest()
    {
        var vm = new MainWindowViewModel();
        for (var i = 0; i < MainWindowViewModel.MaxOutputLogEntries + 25; i++)
        {
            vm.AppendEngineLog(new EngineLogEntry { Command = $"cmd{i}" });
        }

        Assert.Equal(MainWindowViewModel.MaxOutputLogEntries, vm.OutputLog.Count);
        // Oldest entries were dropped; the newest is retained (last in).
        Assert.Equal($"cmd{MainWindowViewModel.MaxOutputLogEntries + 24}", vm.OutputLog[^1].Command);
        Assert.Contains("oldest trimmed", vm.OutputSummary);
    }

    [Fact]
    public void ClearOutputLog_EmptiesAndResetsState()
    {
        var vm = new MainWindowViewModel();
        vm.AppendEngineLog(Entry());
        vm.ClearOutputLog();
        Assert.Empty(vm.OutputLog);
        Assert.True(vm.IsNoOutput);
        Assert.False(vm.HasOutput);
    }

    [Fact]
    public void ToggleOutputPanel_FlipsVisibility()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.OutputPanelVisible);
        vm.ToggleOutputPanel();
        Assert.True(vm.OutputPanelVisible);
        vm.ToggleOutputPanel();
        Assert.False(vm.OutputPanelVisible);
    }

    [Fact]
    public void BottomPanels_AreMutuallyExclusive()
    {
        var vm = new MainWindowViewModel();

        vm.ShowProblemsPanel();
        Assert.True(vm.ProblemsPanelVisible);
        Assert.False(vm.OutputPanelVisible);

        vm.ShowOutputPanel();               // opening Output closes Problems
        Assert.True(vm.OutputPanelVisible);
        Assert.False(vm.ProblemsPanelVisible);

        vm.ShowProblemsPanel();             // and vice versa
        Assert.True(vm.ProblemsPanelVisible);
        Assert.False(vm.OutputPanelVisible);
    }
}
