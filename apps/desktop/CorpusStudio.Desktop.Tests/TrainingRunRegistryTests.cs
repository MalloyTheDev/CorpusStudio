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
        vm.ApplyTrainingRunHistory([]);
        Assert.Contains("No training runs recorded", vm.TrainingRunHistorySummary);
    }

    [Fact]
    public void ApplyTrainingRunHistory_ShowsStatusCheckpointsAndEvalLinks()
    {
        var vm = new MainWindowViewModel();
        var record = Record("20260702T183000-abc", "succeeded");
        record.Checkpoints = ["checkpoint-10", "checkpoint-20"];
        record.BeforeEvalPath = "before.json";
        record.ExitCode = 0;
        vm.ApplyTrainingRunHistory([record]);

        Assert.Contains("[succeeded] 20260702T183000-abc", vm.TrainingRunHistorySummary);
        Assert.Contains("2 checkpoint(s)", vm.TrainingRunHistorySummary);
        Assert.Contains("before-eval ✓", vm.TrainingRunHistorySummary);
        Assert.Contains("after-eval –", vm.TrainingRunHistorySummary);
        Assert.Contains("exit 0", vm.TrainingRunHistorySummary);
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
        vm.ApplyTrainingRunHistory([record]);
        Assert.Contains("0 checkpoint(s)", vm.TrainingRunHistorySummary);
    }

    [Fact]
    public void SetTrainingRunHistoryError_ShowsMessage()
    {
        var vm = new MainWindowViewModel();
        vm.SetTrainingRunHistoryError("disk gone");
        Assert.Contains("Run history could not load", vm.TrainingRunHistorySummary);
        Assert.Contains("disk gone", vm.TrainingRunHistorySummary);
    }
}
