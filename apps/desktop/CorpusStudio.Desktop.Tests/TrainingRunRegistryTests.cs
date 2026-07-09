using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class TrainingRunRegistryTests
{
    private static TrainingRunRecord Record(string runId, string status, int? pid = null) => new()
    {
        RunId = runId,
        CreatedAt = "2026-07-02T18:00:00Z",
        UpdatedAt = "2026-07-02T18:00:00Z",
        Status = status,
        BaseModel = "Qwen/Qwen2.5-Coder-7B",
        Target = "axolotl_yaml",
        Pid = pid,
    };

    // --- reconciliation (pure, injected liveness) ----------------------------

    [Fact]
    public void Reconcile_FlipsDeadRunningToInterrupted()
    {
        var records = new[] { Record("r1", "running", pid: 999), Record("r2", "succeeded") };
        PythonEngineService.ReconcileRunningRecords(records, _ => false, "t2");
        Assert.Equal("interrupted", records[0].Status);
        Assert.Equal("t2", records[0].UpdatedAt);
        Assert.Contains("reconciled", records[0].Notes);
        Assert.Equal("succeeded", records[1].Status); // terminal untouched
    }

    [Fact]
    public void Reconcile_KeepsAliveRunning()
    {
        var records = new[] { Record("r1", "running", pid: 123) };
        PythonEngineService.ReconcileRunningRecords(records, _ => true, "t2");
        Assert.Equal("running", records[0].Status);
    }

    [Fact]
    public void Reconcile_PidlessRunningIsInterrupted()
    {
        // Liveness is decided by the injected check; a pidless run resolves to dead.
        var records = new[] { Record("r1", "running", pid: null) };
        PythonEngineService.ReconcileRunningRecords(records, r => r.Pid is not null, "t2");
        Assert.Equal("interrupted", records[0].Status);
    }

    [Fact]
    public void MintRunId_IsChronologicallySortable()
    {
        var id = PythonEngineService.MintTrainingRunId();
        Assert.Equal(24, id.Length);
        Assert.Equal('-', id[15]); // yyyyMMddTHHmmss '-' suffix
    }

    // --- VM history formatting -----------------------------------------------

    [Fact]
    public void ApplyTrainingRunHistory_Empty_ShowsNone()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingRunHistory([]);
        Assert.Contains("No training runs recorded", vm.Training.TrainingRunHistorySummary);
    }

    [Fact]
    public void ApplyTrainingRunHistory_ShowsStatusCheckpointsAndEvalLinks()
    {
        var vm = new MainWindowViewModel();
        var record = Record("20260702T183000-abc", "succeeded");
        record.Checkpoints = ["checkpoint-10", "checkpoint-20"];
        record.BeforeEvalPath = "before.json";
        record.ExitCode = 0;
        vm.Training.ApplyTrainingRunHistory([record]);

        Assert.Contains("[succeeded] 20260702T183000-abc", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("2 checkpoint(s)", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("before-eval ✓", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("after-eval –", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("exit 0", vm.Training.TrainingRunHistorySummary);
    }

    [Fact]
    public void ApplyTrainingRunHistory_ShowsTheReproducibilityManifest()
    {
        var vm = new MainWindowViewModel();
        var record = Record("20260702T183000-abc", "succeeded");
        record.Provenance = new RunProvenance
        {
            DatasetFingerprint = "0123456789abcdef0123456789abcdef",
            DatasetRowCount = 42,
            ConfigSha256 = "fedcba9876543210fedcba9876543210",
            EngineVersion = "1.3.0",
        };

        vm.Training.ApplyTrainingRunHistory([record]);

        Assert.Contains("provenance:", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("data 0123456789ab (42 rows)", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("config fedcba987654", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("engine 1.3.0", vm.Training.TrainingRunHistorySummary);
    }

    [Fact]
    public void SaveTrainingRunRecord_RejectsUnsafeRunId()
    {
        var service = new PythonEngineService();
        var record = Record("bad id/with slash", "running");
        Assert.Throws<System.ArgumentException>(() => service.SaveTrainingRunRecord("proj", record));
    }

    [Fact]
    public void ApplyTrainingRunHistory_NullCheckpoints_DoesNotThrow()
    {
        var vm = new MainWindowViewModel();
        var record = Record("20260702T183000-x", "succeeded");
        record.Checkpoints = null!; // simulates "checkpoints": null in JSON
        vm.Training.ApplyTrainingRunHistory([record]);
        Assert.Contains("0 checkpoint(s)", vm.Training.TrainingRunHistorySummary);
    }

    [Fact]
    public void ApplyTrainingRunGate_FormatsVerdict()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingRunGate(new GateReport
        {
            Scope = "training_run",
            OverallStatus = "block",
            Results = [new GateResult { GateId = "regression", Status = "block", Message = "Trained model regressed: dropped 10." }],
        });
        Assert.Contains("Regression gate: ⛔ BLOCK", vm.Training.TrainingRunGateSummary);
        Assert.Contains("regressed", vm.Training.TrainingRunGateSummary);
    }

    [Fact]
    public void ApplyTrainingRunGate_WarnUnverified()
    {
        var vm = new MainWindowViewModel();
        vm.Training.ApplyTrainingRunGate(new GateReport
        {
            Scope = "training_run",
            OverallStatus = "warn",
            Results = [new GateResult { GateId = "regression", Status = "warn", Message = "Unverified linkage." }],
        });
        Assert.Contains("⚠ WARN", vm.Training.TrainingRunGateSummary);
        Assert.Contains("Unverified linkage", vm.Training.TrainingRunGateSummary);
    }

    [Fact]
    public void SetTrainingRunGateError_ShowsMessage()
    {
        var vm = new MainWindowViewModel();
        vm.Training.SetTrainingRunGateError("engine missing");
        Assert.Contains("Regression gate could not run", vm.Training.TrainingRunGateSummary);
        Assert.Contains("engine missing", vm.Training.TrainingRunGateSummary);
    }

    [Fact]
    public void SetTrainingRunHistoryError_ShowsMessage()
    {
        var vm = new MainWindowViewModel();
        vm.Training.SetTrainingRunHistoryError("disk gone");
        Assert.Contains("Run history could not load", vm.Training.TrainingRunHistorySummary);
        Assert.Contains("disk gone", vm.Training.TrainingRunHistorySummary);
    }
}
