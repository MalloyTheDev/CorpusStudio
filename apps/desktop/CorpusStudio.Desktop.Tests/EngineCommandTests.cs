using System;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The first engine-run orchestration moved off the desktop code-behind into a shared async
/// command (RunDatasetDebtCommand → RunDatasetDebtAsync), behind the IEngineService seam so it is
/// testable with a fake engine (no Python shell-out).</summary>
public sealed class EngineCommandTests
{
    private sealed class FakeEngine : IEngineService
    {
        private readonly DebtReport _debt;
        private readonly GateReport _gate;
        public string? LastProjectPath { get; private set; }
        public FakeEngine(DebtReport debt, GateReport? gate = null)
        {
            _debt = debt;
            _gate = gate ?? new GateReport();
        }

        public Task<DebtReport> GetDatasetDebtAsync(string projectPath)
        {
            LastProjectPath = projectPath;
            return Task.FromResult(_debt);
        }

        public Task<GateReport> RunDatasetGatesAsync(string projectPath, string schemaId, bool exportScope = false)
        {
            LastProjectPath = projectPath;
            return Task.FromResult(_gate);
        }

        public Task<GateReport> RunChatGatesAsync(string projectPath) => Task.FromResult(_gate);
        public Task<ArenaReport> RunArenaAsync(string promptsText, System.Collections.Generic.IReadOnlyList<string> models,
            string? judgeModel = null, string? projectPath = null) => Task.FromResult(new ArenaReport());
        public Task<SuiteReport> RunSuiteAsync(string projectPath, string suiteName) => Task.FromResult(new SuiteReport());
        public Task<System.Collections.Generic.IReadOnlyList<SuiteHistoryEntry>> GetSuiteHistoryAsync(string projectPath, string suiteName)
            => Task.FromResult<System.Collections.Generic.IReadOnlyList<SuiteHistoryEntry>>(new System.Collections.Generic.List<SuiteHistoryEntry>());
        public Task<BenchmarkReport> RunBenchmarkAsync(string projectPath, string schemaId, string backend, System.Collections.Generic.IReadOnlyList<string> models, string? baseUrl, int? limit, double scoreThreshold, int timeoutSeconds) => Task.FromResult(new BenchmarkReport());
        public int PreviewAccepted { get; set; }
        public int PreviewRejected { get; set; }
        public bool CommitCalled { get; private set; }
        public Task<ImportPreviewReport> PreviewImportAsync(string importPath, string schemaId) => Task.FromResult(new ImportPreviewReport { AcceptedRows = PreviewAccepted, RejectedRows = PreviewRejected });
        public ImportCommitResult CommitJsonlImportToProjectExamples(string projectPath, string importPath, ImportPreviewReport report) { CommitCalled = true; return new ImportCommitResult(0, 0, null); }
        public bool ValidateReturnsValid { get; set; }
        public bool AppendCalled { get; private set; }
        public int AppendDraftToProjectExamples(string projectPath, string draftText) { AppendCalled = true; return 1; }
        public System.Collections.Generic.IReadOnlyList<SavedExampleItem> LoadExamples(string projectPath) => new System.Collections.Generic.List<SavedExampleItem>();
        public void RemoveImportQuarantineItem(ImportQuarantineItem item) { }
        public System.Collections.Generic.IReadOnlyList<ImportQuarantineItem> LoadImportQuarantineItems(string projectPath) => new System.Collections.Generic.List<ImportQuarantineItem>();
        public bool RestoreCalled { get; private set; }
        public Task<RestoreResult> RestoreDatasetVersionInPlaceAsync(string projectPath, string versionId, string undoLabel) { RestoreCalled = true; return Task.FromResult(new RestoreResult()); }
        public string ExportPreferenceRanking(string projectPath, System.Collections.Generic.IReadOnlyList<PreferenceReviewItem> items) => string.Empty;
        public System.Collections.Generic.IReadOnlyList<(ModelArtifactRecord Record, string Integrity)> LoadArtifacts(string projectPath, System.Func<ModelArtifactRecord, string>? integrityOf = null) => new System.Collections.Generic.List<(ModelArtifactRecord, string)>();
        public ModelArtifactRecord RegisterArtifact(string projectPath, string runId, string path, string kind = "adapter", string notes = "") => new ModelArtifactRecord();
        public Task<GateReport> GateArtifactAsync(string projectPath, string artifactId) => Task.FromResult(new GateReport());
        public Task<ModelArtifactRecord> PromoteArtifactAsync(string projectPath, string artifactId) => Task.FromResult(new ModelArtifactRecord());
        public ModelArtifactRecord UpdateArtifactStatus(string projectPath, string artifactId, string status) => new ModelArtifactRecord();
        public Task<ProjectIndexRebuildResult> RebuildProjectIndexAsync() => Task.FromResult(new ProjectIndexRebuildResult());
        public Task<System.Collections.Generic.IReadOnlyList<DatasetProjectListItem>> LoadProjectsFromIndexAsync() => Task.FromResult<System.Collections.Generic.IReadOnlyList<DatasetProjectListItem>>(new System.Collections.Generic.List<DatasetProjectListItem>());
        public Task SetGateThresholdsAsync(string projectPath, GateThresholds thresholds) => Task.CompletedTask;
        public Task<System.Collections.Generic.IReadOnlyList<ProviderPolicyItem>> GetProviderPoliciesAsync(string projectPath) => Task.FromResult<System.Collections.Generic.IReadOnlyList<ProviderPolicyItem>>(new System.Collections.Generic.List<ProviderPolicyItem>());
        public Task<QualityReport> BuildQualityReportAsync(string projectPath) => Task.FromResult(new QualityReport());
        public QualityHistoryEntry SaveQualityHistoryEntry(string projectPath, QualityReport report) => new QualityHistoryEntry();
        public System.Collections.Generic.IReadOnlyList<QualityHistoryEntry> LoadQualityHistory(string projectPath, int maxEntries = 5) => new System.Collections.Generic.List<QualityHistoryEntry>();
        public System.Collections.Generic.IReadOnlyList<TrainingRunRecord> LoadTrainingRunRecords(string projectPath) => new System.Collections.Generic.List<TrainingRunRecord>();
        public string? LinkAfterEvalToNewestRun(string projectPath, string afterEvalPath, string? afterEvalModel) => null;
        public Task<GateReport> RunTrainingRunGateAsync(string projectPath, string runId) => Task.FromResult(new GateReport());
        public Task<SplitReport> GenerateProjectSplitsAsync(string projectPath, string schemaId, double trainRatio, double validationRatio, int seed) => Task.FromResult(new SplitReport());
        public void SaveProjectSplitSettings(string projectPath, SplitSettings settings) { }
        public Task<BackendHealthReport> CheckBackendHealthAsync(string backend, string model, string? baseUrl, int timeoutSeconds) => Task.FromResult(new BackendHealthReport());
        public Task<BackendModelListReport> ListBackendModelsAsync(string backend, string? baseUrl, int timeoutSeconds) => Task.FromResult(new BackendModelListReport());
        public Task<EvaluationRunResult> RunEvaluationAsync(string projectPath, string schemaId, string backend, string model, string? baseUrl, int? limit, double scoreThreshold, int timeoutSeconds, string? judgeModel = null, string? judgeBackend = null, string? judgeBaseUrl = null) => Task.FromResult(new EvaluationRunResult(new EvaluationReport(), string.Empty, string.Empty));
        public System.Collections.Generic.IReadOnlyList<EvaluationReportHistoryItem> LoadEvaluationReportHistory(string projectPath, int maxReports = 20) => new System.Collections.Generic.List<EvaluationReportHistoryItem>();
        public System.Collections.Generic.IReadOnlyList<ReviewedFixRecord> ReconcileReviewedFixes(string projectPath, System.Collections.Generic.IReadOnlyList<EvaluationExampleResult> results) => new System.Collections.Generic.List<ReviewedFixRecord>();
        public Task<TrainingConfigExportResult> GenerateTrainingConfigAsync(string projectPath, string schemaId, string target, string baseModel, string datasetFormat, int sequenceLen, int loraR, int loraAlpha, int microBatchSize, int gradientAccumulationSteps, double learningRate) => Task.FromResult(new TrainingConfigExportResult());
        public Task<System.Collections.Generic.IReadOnlyList<DatasetVersionDisplayItem>> LoadDatasetVersionsAsync(string projectPath)
            => Task.FromResult<System.Collections.Generic.IReadOnlyList<DatasetVersionDisplayItem>>(new System.Collections.Generic.List<DatasetVersionDisplayItem>());
        public Task<DatasetVersionRecord> CreateDatasetVersionAsync(string projectPath, string label, string trigger) => Task.FromResult(new DatasetVersionRecord());
        public Task<string> GetDatasetVersionCardAsync(string projectPath, string versionId) => Task.FromResult("# card");
        public Task<string> GetDatasetVersionDiffAsync(string projectPath, string baseVersionId, string otherVersionId) => Task.FromResult("# diff");
        public Task<string> GetWeightCardAsync(string projectPath, string artifactId) => Task.FromResult("# weights");
        public Task<DatasetCardResult> GenerateDatasetCardAsync(string projectPath, string schemaId) => Task.FromResult(new DatasetCardResult());
        public Task<ValidationReport> ValidateDraftAsync(string draftText, string schemaId) => Task.FromResult(new ValidationReport { Valid = ValidateReturnsValid });
        public Task<TrainingCompatibilityResult> CheckTrainingCompatibilityAsync(string schemaId, string datasetFormat, string target) => Task.FromResult(new TrainingCompatibilityResult());
        public Task<System.Collections.Generic.IReadOnlyList<SuiteSummary>> ListSuitesAsync(string projectPath) => Task.FromResult((System.Collections.Generic.IReadOnlyList<SuiteSummary>)new System.Collections.Generic.List<SuiteSummary>());
        public Task<PreferenceExportResult> ExportPreferenceForTrainingAsync(string projectPath, string format) => Task.FromResult(new PreferenceExportResult());
    }

    private sealed class FakeDialogService : IDialogService
    {
        private readonly bool _confirm;
        public FakeDialogService(bool confirm) => _confirm = confirm;
        public Task<bool> ConfirmAsync(string message, string title, DialogButtons buttons = DialogButtons.YesNo, DialogSeverity severity = DialogSeverity.Question, bool defaultAffirmative = false) => Task.FromResult(_confirm);
        public Task ShowAsync(string message, string title, DialogSeverity severity = DialogSeverity.Information) => Task.CompletedTask;
    }

    private static MainWindowViewModel VmWith(IEngineService engine, IDialogService? dialogs = null) => new(
        new DebtViewModel(), new ArenaViewModel(), new SettingsViewModel(), new VersionsViewModel(),
        new ArtifactsViewModel(), new SuitesViewModel(), new SplitsViewModel(), new PreferenceReviewViewModel(),
        new QuarantineViewModel(), new ExamplesViewModel(), new WritingStudioViewModel(),
        new AiAssistRewriteBatchesViewModel(), new AiAssistConnectionViewModel(),
        new EvaluationConnectionViewModel(), new QualityViewModel(), engine, dialogs ?? new NullDialogService(),
        new NullFilePickerService());

    private static void SelectFakeProject(MainWindowViewModel vm) => vm.SelectProject(
        new DatasetProjectListItem(
            new DatasetProject("p", "P", "instruction", new DateTime(2026, 1, 1), new DateTime(2026, 1, 1)),
            @"C:\fake\project"),
        "instruction");

    [Fact]
    public async Task RunDatasetDebt_WithoutProject_SetsDebtError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "D", HasData = true }));

        await vm.RunDatasetDebtAsync();

        Assert.Contains("Create or select a dataset project", vm.Debt.DebtSummary);
    }

    [Fact]
    public async Task RunDatasetDebt_AppliesTheEngineReportToTheDebtTab()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "D", HasData = true, ExampleCount = 19 });
        var vm = VmWith(engine);
        SelectFakeProject(vm);

        await vm.RunDatasetDebtAsync();

        Assert.Equal("D", vm.Debt.DebtGrade);
        Assert.Equal(@"C:\fake\project", engine.LastProjectPath);
    }

    [Fact]
    public async Task RunGates_WithProject_InvokesTheEngineForThatProject()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" }, new GateReport());
        var vm = VmWith(engine);
        SelectFakeProject(vm);

        await vm.RunGatesAsync();

        Assert.Equal(@"C:\fake\project", engine.LastProjectPath);
    }

    [Fact]
    public async Task CaptureDatasetVersion_WithProject_CapturesAndClearsTheLabel()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm);
        vm.Versions.DatasetVersionLabel = "before training";

        await vm.CaptureDatasetVersionAsync();

        Assert.Equal(string.Empty, vm.Versions.DatasetVersionLabel); // cleared after a successful capture
    }

    [Fact]
    public async Task CaptureDatasetVersion_WithoutProject_SetsVersionError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.CaptureDatasetVersionAsync();

        Assert.Contains("Create or select a dataset project", vm.Versions.DatasetVersionSummary + vm.Versions.DatasetVersionDetail);
    }

    [Fact]
    public async Task GenerateTrainingConfig_WithoutProject_SetsError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.GenerateTrainingConfigAsync();

        Assert.Contains("Create or select a dataset project", vm.Training.TrainingSummary);
    }

    [Fact]
    public async Task GenerateTrainingConfig_WithMissingTarget_SetsValidationError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm);
        vm.Training.TrainingTarget = string.Empty;

        await vm.GenerateTrainingConfigAsync();

        Assert.Contains("target is required", vm.Training.TrainingSummary);
    }

    [Fact]
    public async Task RefreshDatasetVersions_WithoutProject_SetsVersionError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.RefreshDatasetVersionsAsync();

        Assert.Contains("Create or select a dataset project", vm.Versions.DatasetVersionSummary + vm.Versions.DatasetVersionDetail);
    }

    [Fact]
    public async Task RunGates_WithoutProject_SetsGateError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        await vm.RunGatesAsync();
        Assert.Contains("Create or select a dataset project", vm.GateSummary);
    }

    [Fact]
    public async Task RunBenchmark_WithoutProject_SetsBenchmarkError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        vm.BenchmarkModelsInput = "model-a";

        await vm.RunBenchmarkAsync();

        Assert.Contains("Create or select a dataset project", vm.BenchmarkSummary);
    }

    [Fact]
    public async Task RunBenchmark_WithoutModels_SetsBenchmarkError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm);
        vm.BenchmarkModelsInput = string.Empty;

        await vm.RunBenchmarkAsync();

        Assert.Contains("at least one model", vm.BenchmarkSummary);
    }

    [Fact]
    public async Task RunBenchmark_WithProjectAndModels_AppliesTheEngineReport()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm);
        vm.BenchmarkModelsInput = "model-a";

        await vm.RunBenchmarkAsync();

        Assert.Contains("Benchmarked", vm.BenchmarkSummary); // ApplyBenchmarkReport ran (no error branch)
    }

    [Fact]
    public async Task RunEvaluation_WithoutProject_SetsEvaluationError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.RunEvaluationAsync();

        Assert.Contains("Create or select a dataset project", vm.Evaluation.EvaluationSummary);
    }

    [Fact]
    public async Task RunEvaluation_WithUnreachableBackend_SetsPreflightError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm); // instruction schema + valid default connection options

        await vm.RunEvaluationAsync();

        // The fake's default BackendHealthReport is not reachable → preflight blocks the run.
        Assert.Contains("backend health check failed", vm.Evaluation.EvaluationSummary);
    }

    [Fact]
    public async Task RerunEvaluationReport_WithoutProject_SetsEvaluationError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.RerunEvaluationReportAsync();

        Assert.Contains("Create or select a dataset project", vm.Evaluation.EvaluationSummary);
    }

    [Fact]
    public async Task CheckEvaluationBackend_WithBlankBackend_SetsEvaluationError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        vm.EvaluationConnection.EvaluationBackend = "   ";

        await vm.CheckEvaluationBackendAsync();

        Assert.Contains("backend is required", vm.Evaluation.EvaluationSummary);
    }

    [Fact]
    public async Task RefreshEvaluationModels_WithBlankBackend_SetsModelListError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        vm.EvaluationConnection.EvaluationBackend = string.Empty;

        await vm.RefreshEvaluationModelsAsync();

        Assert.Contains("backend is required", vm.EvaluationConnection.EvaluationModelListSummary);
    }

    [Fact]
    public async Task CheckAiAssistBackend_WithBlankBackend_SetsAiAssistError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        vm.AiAssistConnection.AiAssistBackend = "   ";

        await vm.CheckAiAssistBackendAsync();

        Assert.Contains("backend is required", vm.AiAssist.AiAssistSummary);
    }

    [Fact]
    public async Task RefreshAiAssistModels_WithBlankBackend_SetsModelListError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        vm.AiAssistConnection.AiAssistBackend = string.Empty;

        await vm.RefreshAiAssistModelsAsync();

        Assert.Contains("backend is required", vm.AiAssistConnection.AiAssistModelListSummary);
    }

    [Fact]
    public async Task GenerateSplits_WithoutProject_SetsSplitError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.GenerateSplitsAsync();

        Assert.Contains("Create a dataset project", vm.Splits.SplitSummary);
    }

    [Fact]
    public async Task GenerateSplits_WithInvalidTrainPercent_SetsSplitError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm);
        vm.Splits.SplitTrainPercent = "not-a-number";

        await vm.GenerateSplitsAsync();

        Assert.Contains("Train split must be a number", vm.Splits.SplitSummary);
    }

    [Fact]
    public async Task GateTrainingRun_WithoutProject_SetsGateError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.GateTrainingRunAsync();

        Assert.Contains("Create or select a dataset project", vm.Training.TrainingRunGateSummary);
    }

    [Fact]
    public async Task GateTrainingRun_WithNoRecordedRun_SetsGateError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm); // fake has no runs + LinkAfterEvalToNewestRun returns null → runId is null

        await vm.GateTrainingRunAsync();

        Assert.Contains("No training run has been recorded", vm.Training.TrainingRunGateSummary);
    }

    [Fact]
    public async Task RunQuality_WithoutProject_SetsQualityError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.RefreshQualityAsync();

        Assert.Contains("Create or select a dataset project", vm.Quality.QualitySummary);
    }

    [Fact]
    public async Task SaveGateThresholds_WithoutProject_SetsError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.SaveGateThresholdsAsync();

        Assert.Contains("Create or select a dataset project", vm.Settings.GateThresholdsSummary);
    }

    [Fact]
    public async Task RefreshProviderPolicies_WithoutProject_SetsError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.RefreshProviderPoliciesAsync();

        Assert.Contains("Create or select a dataset project", vm.Settings.ProviderPolicySummary);
    }

    [Fact]
    public async Task RebuildProjectIndex_AppliesTheRebuiltResult()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.RebuildProjectIndexAsync();

        // Fake returns an empty rebuild result + empty project list → the flow completed end-to-end.
        Assert.Contains("No projects found to index", vm.ProjectIndexSummary);
        Assert.Empty(vm.Projects);
    }

    [Fact]
    public void RegisterArtifactFromRun_WithoutProject_SetsArtifactError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        vm.RegisterArtifactFromRun();

        Assert.Contains("Create or select a dataset project", vm.Artifacts.ArtifactSummary);
    }

    [Fact]
    public async Task KeepArtifact_WithoutSelection_SetsArtifactError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        SelectFakeProject(vm); // no artifact selected

        await vm.KeepArtifactAsync();

        Assert.Contains("Select an artifact first", vm.Artifacts.ArtifactSummary);
    }

    [Fact]
    public void RefreshArtifacts_WithoutProject_SetsArtifactError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        vm.RefreshArtifacts();

        Assert.Contains("Create or select a dataset project", vm.Artifacts.ArtifactSummary);
    }

    [Fact]
    public void ExportPreferenceRanking_WithoutProject_SetsError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        vm.ExportPreferenceRanking();

        Assert.Contains("Create or select a preference project", vm.PreferenceReview.PreferenceReviewSummary);
    }

    [Fact]
    public async Task RestoreDatasetVersion_WithoutSelection_SetsVersionError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));

        await vm.RestoreDatasetVersionAsync();

        Assert.Contains("Select a version to restore", vm.Versions.DatasetVersionSummary + vm.Versions.DatasetVersionDetail);
    }

    [Fact]
    public async Task RestoreDatasetVersion_WhenConfirmDeclined_DoesNotCallEngine()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" });
        var vm = VmWith(engine, new FakeDialogService(confirm: false)); // user clicks No
        SelectFakeProject(vm);
        vm.Versions.SelectedDatasetVersion = new DatasetVersionDisplayItem(new DatasetVersionRecord { VersionId = "v1" });

        await vm.RestoreDatasetVersionAsync();

        Assert.False(engine.RestoreCalled); // declining the confirm gates the destructive restore
    }

    [Fact]
    public async Task RestoreDatasetVersion_WhenConfirmed_CallsEngine()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" });
        var vm = VmWith(engine, new FakeDialogService(confirm: true)); // user clicks Yes
        SelectFakeProject(vm);
        vm.Versions.SelectedDatasetVersion = new DatasetVersionDisplayItem(new DatasetVersionRecord { VersionId = "v1" });

        await vm.RestoreDatasetVersionAsync();

        Assert.True(engine.RestoreCalled);
    }

    [Fact]
    public async Task SaveExample_WithValidDraft_AppendsAndMarksDraftClean()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" }) { ValidateReturnsValid = true };
        var vm = VmWith(engine);
        SelectFakeProject(vm);

        await vm.SaveExampleAsync();

        Assert.True(engine.AppendCalled);
        Assert.False(vm.WritingStudio.IsDraftDirty); // MarkDraftClean ran after persist
    }

    [Fact]
    public async Task SaveExample_WithInvalidDraft_DoesNotAppend()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" }) { ValidateReturnsValid = false };
        var vm = VmWith(engine);
        SelectFakeProject(vm);

        await vm.SaveExampleAsync();

        Assert.False(engine.AppendCalled); // invalid draft is not persisted
    }

    [Fact]
    public async Task PreviewAndImport_WhenConfirmed_CommitsTheImport()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" }) { PreviewAccepted = 3 };
        var vm = VmWith(engine, new FakeDialogService(confirm: true));
        SelectFakeProject(vm);

        await vm.PreviewAndImportJsonlAsync(@"C:\fake\import.jsonl");

        Assert.True(engine.CommitCalled);
    }

    [Fact]
    public async Task PreviewAndImport_WhenDeclined_DoesNotCommit()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" }) { PreviewAccepted = 3 };
        var vm = VmWith(engine, new FakeDialogService(confirm: false));
        SelectFakeProject(vm);

        await vm.PreviewAndImportJsonlAsync(@"C:\fake\import.jsonl");

        Assert.False(engine.CommitCalled); // declining the confirm gates the import
    }

    [Fact]
    public async Task PreviewAndImport_WhenNoImportableRows_DoesNotCommit()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" }); // 0 accepted / 0 rejected
        var vm = VmWith(engine, new FakeDialogService(confirm: true));
        SelectFakeProject(vm);

        await vm.PreviewAndImportJsonlAsync(@"C:\fake\import.jsonl");

        Assert.False(engine.CommitCalled);
    }
}
