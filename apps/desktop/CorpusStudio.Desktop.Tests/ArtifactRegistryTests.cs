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

    [Fact]
    public void ApplyArtifacts_SummarizesKeptAndFlagged()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyArtifacts(
        [
            new ArtifactDisplayItem(Record(status: "kept"), "ok", "base"),
            new ArtifactDisplayItem(Record(status: "candidate"), "missing", "base"),
        ]);
        Assert.Equal(2, vm.ModelArtifacts.Count);
        Assert.Contains("1 kept", vm.ArtifactSummary);
        Assert.Contains("1 with integrity issues", vm.ArtifactSummary);
    }

    [Fact]
    public void ApplyArtifacts_EmptyShowsNone()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyArtifacts([]);
        Assert.Contains("No artifacts registered", vm.ArtifactSummary);
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
        var allowed = vm.ApplyPromoteGate(GateReport("block",
            new GateResult { GateId = "integrity", Status = "block", Message = "Artifact weights are modified." }));
        Assert.False(allowed);
        Assert.Contains("Keep blocked", vm.ArtifactDetail);
        Assert.Contains("modified", vm.ArtifactDetail);
    }

    [Fact]
    public void ApplyPromoteGate_WarnAllowsKeepWithNote()
    {
        var vm = new MainWindowViewModel();
        var allowed = vm.ApplyPromoteGate(GateReport("warn",
            new GateResult { GateId = "regression", Status = "warn", Message = "Unverified linkage." }));
        Assert.True(allowed);
        Assert.Contains("warned", vm.ArtifactDetail);
        Assert.Contains("Unverified linkage", vm.ArtifactDetail);
    }

    [Fact]
    public void ApplyPromoteGate_PassAllowsKeep()
    {
        var vm = new MainWindowViewModel();
        var allowed = vm.ApplyPromoteGate(GateReport("pass",
            new GateResult { GateId = "integrity", Status = "pass", Message = "ok" }));
        Assert.True(allowed);
        Assert.Contains("promote gate passed", vm.ArtifactDetail);
    }

    [Fact]
    public void SetArtifactDetail_ShowsCardMarkdown()
    {
        var vm = new MainWindowViewModel();
        vm.SetArtifactDetail("# Weight Card — abc");
        Assert.Contains("Weight Card", vm.ArtifactDetail);
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
