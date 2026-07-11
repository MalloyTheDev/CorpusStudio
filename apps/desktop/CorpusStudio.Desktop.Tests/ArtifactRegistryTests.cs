using System.IO;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class ArtifactRegistryTests
{
    private static ModelArtifactRecord Record(string status = "candidate", string? fingerprint = "fp", string path = "C:/weights/adapter") => new()
    {
        ArtifactId = "run-1-abcd1234",
        RunId = "run-1",
        CreatedAt = "t",
        UpdatedAt = "t",
        Path = path,
        Kind = "adapter",
        Status = status,
        Fingerprint = fingerprint,
    };

    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    // --- integrity (pure, injected fingerprint) ------------------------------

    [Fact]
    public void Integrity_MissingWhenPathGone()
    {
        var record = Record(path: "C:/definitely/not/here-xyz");
        Assert.Equal("missing", PythonEngineService.ComputeArtifactIntegrity(record, _ => "fp"));
    }

    [Fact]
    public void Integrity_OkWhenFingerprintMatches()
    {
        // Point at an existing path (the temp dir) so the missing-check passes.
        var record = Record(path: Path.GetTempPath(), fingerprint: "fp");
        Assert.Equal("ok", PythonEngineService.ComputeArtifactIntegrity(record, _ => "fp"));
    }

    [Fact]
    public void Integrity_ModifiedWhenFingerprintDiffers()
    {
        var record = Record(path: Path.GetTempPath(), fingerprint: "old");
        Assert.Equal("modified", PythonEngineService.ComputeArtifactIntegrity(record, _ => "new"));
    }

    [Fact]
    public void Integrity_OkWhenNoStoredFingerprint()
    {
        var record = Record(path: Path.GetTempPath(), fingerprint: null);
        Assert.Equal("ok", PythonEngineService.ComputeArtifactIntegrity(record, _ => "anything"));
    }

    // --- register requires a run --------------------------------------------

    [Fact]
    public void RegisterArtifact_ThrowsWhenRunMissing()
    {
        var service = new PythonEngineService();
        Assert.Throws<System.InvalidOperationException>(
            () => service.RegisterArtifact("no-such-project", "ghost-run", "C:/weights"));
    }

    // --- display formatting --------------------------------------------------

    [Fact]
    public void DisplayItem_FormatsStatusIntegrityAndResolvedModel()
    {
        var item = new ArtifactDisplayItem(Record(status: "kept"), "modified", "Qwen/Qwen2.5-Coder-7B");
        Assert.Contains("[kept]", item.DisplayName);
        Assert.Contains("⚠ modified", item.DisplayName);
        Assert.Contains("Qwen/Qwen2.5-Coder-7B", item.DisplayName);
    }

    // --- Nocturne card display helpers (production re-skin) ------------------

    [Fact]
    public void PrimaryName_UsesResolvedBaseElseExplicitUnknown()
    {
        Assert.Equal("llama-3.1-8b", new ArtifactDisplayItem(Record(), "ok", "llama-3.1-8b").PrimaryName);
        Assert.Equal("(base unknown)", new ArtifactDisplayItem(Record(), "ok", "  ").PrimaryName);
    }

    [Theory]
    [InlineData("ok", "integrity: present")]
    [InlineData("modified", "integrity: modified")]
    [InlineData("missing", "integrity: missing")]
    public void IntegrityLabel_PhrasedForTheChip(string integrity, string expected)
    {
        Assert.Equal(expected, new ArtifactDisplayItem(Record(), integrity, "base").IntegrityLabel);
    }

    [Fact]
    public void IntegrityFlags_AreMutuallyExclusive()
    {
        var ok = new ArtifactDisplayItem(Record(), "ok", "base");
        Assert.True(ok.IsOk);
        Assert.False(ok.IsFlagged);

        var modified = new ArtifactDisplayItem(Record(), "modified", "base");
        Assert.True(modified.IsModified);
        Assert.True(modified.IsFlagged);
        Assert.False(modified.IsOk);

        var missing = new ArtifactDisplayItem(Record(), "missing", "base");
        Assert.True(missing.IsMissing);
        Assert.True(missing.IsFlagged);
    }

    [Fact]
    public void IntegrityBlockMessage_OnlyForFlaggedStates()
    {
        Assert.Equal(string.Empty, new ArtifactDisplayItem(Record(), "ok", "base").IntegrityBlockMessage);
        Assert.Contains("weights changed since eval",
            new ArtifactDisplayItem(Record(), "modified", "base").IntegrityBlockMessage);
        Assert.Contains("missing",
            new ArtifactDisplayItem(Record(), "missing", "base").IntegrityBlockMessage);
    }

    [Fact]
    public void CreatedDisplay_FormatsIsoAndFallsBackToRaw()
    {
        var iso = new ArtifactDisplayItem(
            new ModelArtifactRecord { CreatedAt = "2026-07-10T14:32:00Z" }, "ok", "base");
        Assert.Equal("2026-07-10 14:32 UTC", iso.CreatedDisplay);

        // A non-timestamp value (legacy/test data) is shown unchanged — never invented.
        var raw = new ArtifactDisplayItem(new ModelArtifactRecord { CreatedAt = "t" }, "ok", "base");
        Assert.Equal("t", raw.CreatedDisplay);
    }

    [Fact]
    public void HasArtifacts_TracksTheCollection()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.Artifacts.HasArtifacts);
        vm.Artifacts.ApplyArtifacts([new ArtifactDisplayItem(Record(status: "kept"), "ok", "base")]);
        Assert.True(vm.Artifacts.HasArtifacts);
        vm.Artifacts.ApplyArtifacts([]);
        Assert.False(vm.Artifacts.HasArtifacts);
    }

    [Fact]
    public void HasArtifactDetail_TrueOnlyForRealContent()
    {
        var vm = new MainWindowViewModel();
        // Idle hint => no detail surface.
        Assert.False(vm.Artifacts.HasArtifactDetail);

        vm.Artifacts.SetArtifactDetail("# Weight Card — abc");
        Assert.True(vm.Artifacts.HasArtifactDetail);

        // A promote-gate verdict is real detail too.
        vm.Artifacts.ApplyPromoteGate(GateReport("block",
            new GateResult { GateId = "integrity", Status = "block", Message = "modified" }));
        Assert.True(vm.Artifacts.HasArtifactDetail);

        // Reset returns to the idle hint => surface collapses again.
        vm.Artifacts.Reset();
        Assert.False(vm.Artifacts.HasArtifactDetail);
    }

    [Fact]
    public void ApplyArtifacts_SummarizesKeptAndFlagged()
    {
        var vm = new MainWindowViewModel();
        vm.Artifacts.ApplyArtifacts(
        [
            new ArtifactDisplayItem(Record(status: "kept"), "ok", "base"),
            new ArtifactDisplayItem(Record(status: "candidate"), "missing", "base"),
        ]);
        Assert.Equal(2, vm.Artifacts.ModelArtifacts.Count);
        Assert.Contains("1 kept", vm.Artifacts.ArtifactSummary);
        Assert.Contains("1 with integrity issues", vm.Artifacts.ArtifactSummary);
    }

    [Fact]
    public void ApplyArtifacts_EmptyShowsNone()
    {
        var vm = new MainWindowViewModel();
        vm.Artifacts.ApplyArtifacts([]);
        Assert.Contains("No artifacts registered", vm.Artifacts.ArtifactSummary);
    }

    // --- promote gate verdict (v0.9.1) ---------------------------------------

    private static GateReport GateReport(string overall, params GateResult[] results) => new()
    {
        Scope = "model_artifact",
        OverallStatus = overall,
        Results = results,
    };

    [Fact]
    public void ApplyPromoteGate_BlockRefusesKeep()
    {
        var vm = new MainWindowViewModel();
        var allowed = vm.Artifacts.ApplyPromoteGate(GateReport("block",
            new GateResult { GateId = "integrity", Status = "block", Message = "Artifact weights are modified." }));
        Assert.False(allowed);
        Assert.Contains("Keep blocked", vm.Artifacts.ArtifactDetail);
        Assert.Contains("modified", vm.Artifacts.ArtifactDetail);
    }

    [Fact]
    public void ApplyPromoteGate_WarnAllowsKeepWithNote()
    {
        var vm = new MainWindowViewModel();
        var allowed = vm.Artifacts.ApplyPromoteGate(GateReport("warn",
            new GateResult { GateId = "regression", Status = "warn", Message = "Unverified linkage." }));
        Assert.True(allowed);
        Assert.Contains("warned", vm.Artifacts.ArtifactDetail);
        Assert.Contains("Unverified linkage", vm.Artifacts.ArtifactDetail);
    }

    [Fact]
    public void ApplyPromoteGate_PassAllowsKeep()
    {
        var vm = new MainWindowViewModel();
        var allowed = vm.Artifacts.ApplyPromoteGate(GateReport("pass",
            new GateResult { GateId = "integrity", Status = "pass", Message = "ok" }));
        Assert.True(allowed);
        Assert.Contains("promote gate passed", vm.Artifacts.ArtifactDetail);
    }

    [Fact]
    public void SetArtifactDetail_ShowsCardMarkdown()
    {
        var vm = new MainWindowViewModel();
        vm.Artifacts.SetArtifactDetail("# Weight Card — abc");
        Assert.Contains("Weight Card", vm.Artifacts.ArtifactDetail);
    }

    // --- per-project lifecycle (Phase-2 extraction: give Artifacts the same reset
    //     guard its siblings Debt/Versions already have) -----------------------

    [Fact]
    public void SelectProject_ClearsArtifactState()
    {
        // Artifacts are per-project on disk and the project-load path does not eagerly refresh
        // the list, so a project switch must clear the tab — otherwise the previous project's
        // artifacts/selection/panes linger and a Keep/Reject would act on a stale artifact id
        // against the newly selected project.
        var vm = new MainWindowViewModel();
        vm.Artifacts.ApplyArtifacts([new ArtifactDisplayItem(Record(status: "kept"), "ok", "base")]);
        vm.Artifacts.SelectedModelArtifact = vm.Artifacts.ModelArtifacts[0];
        vm.Artifacts.SetArtifactDetail("# Weight Card — abc");

        vm.SelectProject(Project("other"));

        Assert.Empty(vm.Artifacts.ModelArtifacts);
        Assert.Null(vm.Artifacts.SelectedModelArtifact);
        Assert.Contains("Register a model artifact", vm.Artifacts.ArtifactSummary);
        Assert.DoesNotContain("Weight Card", vm.Artifacts.ArtifactDetail);
    }

    // --- promote-gate bypass closed (Tier-3 hardening) -----------------------

    [Fact]
    public void UpdateArtifactStatus_RefusesKept_MustUseGatedPromote()
    {
        // Promotion to 'kept' must go through the gated engine path (PromoteArtifactAsync),
        // never this direct C# writer — otherwise the promote gate is bypassed. The refusal
        // happens before any filesystem access, so a dummy path is fine.
        var service = new PythonEngineService();
        var ex = Assert.Throws<System.ArgumentException>(
            () => service.UpdateArtifactStatus("any-project", "any-id", "kept"));
        Assert.Contains("PromoteArtifactAsync", ex.Message);
    }

    [Fact]
    public void UpdateArtifactStatus_RejectsUnknownStatus()
    {
        var service = new PythonEngineService();
        Assert.Throws<System.ArgumentException>(
            () => service.UpdateArtifactStatus("any-project", "any-id", "bogus"));
    }

    [Fact]
    public void UpdateArtifactStatus_StillWritesRejected()
    {
        var project = Path.Combine(Path.GetTempPath(), "cs-artifact-" + System.Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(project, "model_artifacts"));
        try
        {
            var record = Record(status: "candidate");
            var file = Path.Combine(project, "model_artifacts", record.ArtifactId + ".json");
            File.WriteAllText(file, System.Text.Json.JsonSerializer.Serialize(record));

            var updated = new PythonEngineService().UpdateArtifactStatus(project, record.ArtifactId, "rejected");
            Assert.Equal("rejected", updated.Status);

            var onDisk = System.Text.Json.JsonSerializer.Deserialize<ModelArtifactRecord>(File.ReadAllText(file));
            Assert.Equal("rejected", onDisk!.Status);
        }
        finally
        {
            try { Directory.Delete(project, recursive: true); } catch (IOException) { }
        }
    }
}
