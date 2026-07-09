using System.Collections.Generic;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>The engine operations a shared view-model needs, behind an interface so run-orchestration
/// can move out of the per-head code-behind into the VMs (as bindable async commands) and be faked in
/// tests. Implemented by <see cref="PythonEngineService"/>. Grown one method at a time as each engine
/// code-behind handler is converted — this is the first (dataset debt).</summary>
public interface IEngineService
{
    /// <summary>Compute the graded dataset-debt ledger for a project's examples.jsonl.</summary>
    Task<DebtReport> GetDatasetDebtAsync(string projectPath);

    /// <summary>Run the dataset gate suite (schema/quality/leakage/PII/eval) over a project.</summary>
    Task<GateReport> RunDatasetGatesAsync(string projectPath, string schemaId, bool exportScope = false);

    /// <summary>Run the chat conversation-structure gate over a chat project.</summary>
    Task<GateReport> RunChatGatesAsync(string projectPath);

    /// <summary>Run a prompt suite across several models (comparison artifacts, not trainable rows).</summary>
    Task<ArenaReport> RunArenaAsync(string promptsText, IReadOnlyList<string> models,
        string? judgeModel = null, string? projectPath = null);

    /// <summary>Run a named evaluation suite (live backend evaluations) and return its report.</summary>
    Task<SuiteReport> RunSuiteAsync(string projectPath, string suiteName);

    /// <summary>A suite's run history (oldest → newest) for the Suites-tab trend.</summary>
    Task<IReadOnlyList<SuiteHistoryEntry>> GetSuiteHistoryAsync(string projectPath, string suiteName);

    /// <summary>Preflight a backend/model (reachability + model availability) before a live run.</summary>
    Task<BackendHealthReport> CheckBackendHealthAsync(string backend, string model, string? baseUrl, int timeoutSeconds);

    /// <summary>List a backend's available models (for the connection panels' model pickers).</summary>
    Task<BackendModelListReport> ListBackendModelsAsync(string backend, string? baseUrl, int timeoutSeconds);

    /// <summary>Run a graded evaluation over the dataset (live backend calls); optional LLM-judge.</summary>
    Task<EvaluationRunResult> RunEvaluationAsync(
        string projectPath, string schemaId, string backend, string model, string? baseUrl,
        int? limit, double scoreThreshold, int timeoutSeconds, string? judgeModel = null,
        string? judgeBackend = null, string? judgeBaseUrl = null);

    /// <summary>The project's saved evaluation reports (newest first) for history/regression.</summary>
    IReadOnlyList<EvaluationReportHistoryItem> LoadEvaluationReportHistory(string projectPath, int maxReports = 20);

    /// <summary>Reconcile tracked reviewed-fixes against a fresh evaluation's per-example results.</summary>
    IReadOnlyList<ReviewedFixRecord> ReconcileReviewedFixes(string projectPath, IReadOnlyList<EvaluationExampleResult> results);

    /// <summary>Preview a JSONL import against a schema (accepted/rejected rows, no write).</summary>
    Task<ImportPreviewReport> PreviewImportAsync(string importPath, string schemaId);

    /// <summary>Commit a previewed JSONL import into the project's examples.jsonl (dedupe + quarantine).</summary>
    ImportCommitResult CommitJsonlImportToProjectExamples(string projectPath, string importPath, ImportPreviewReport report);

    /// <summary>Validate + append the draft to the project's examples.jsonl; returns the rows saved.</summary>
    int AppendDraftToProjectExamples(string projectPath, string draftText);

    /// <summary>Load the project's saved examples (examples.jsonl → display items).</summary>
    IReadOnlyList<SavedExampleItem> LoadExamples(string projectPath);

    /// <summary>Remove a resolved import-quarantine record (its on-disk file).</summary>
    void RemoveImportQuarantineItem(ImportQuarantineItem item);

    /// <summary>Load the project's import-quarantine items (rows that failed import validation).</summary>
    IReadOnlyList<ImportQuarantineItem> LoadImportQuarantineItems(string projectPath);

    /// <summary>Restore a dataset version in place (captures an undo version, verifies, atomically swaps).</summary>
    Task<RestoreResult> RestoreDatasetVersionInPlaceAsync(string projectPath, string versionId, string undoLabel);

    /// <summary>Export the visible preference-review ranking to a file; returns the output path.</summary>
    string ExportPreferenceRanking(string projectPath, IReadOnlyList<PreferenceReviewItem> items);

    /// <summary>Load the project's model artifacts (record + live integrity string).</summary>
    IReadOnlyList<(ModelArtifactRecord Record, string Integrity)> LoadArtifacts(
        string projectPath, System.Func<ModelArtifactRecord, string>? integrityOf = null);

    /// <summary>Register a training run's output directory as a model artifact.</summary>
    ModelArtifactRecord RegisterArtifact(
        string projectPath, string runId, string path, string kind = "adapter", string notes = "");

    /// <summary>Preview the promote gate for an artifact (verdict/reason; does not write).</summary>
    Task<GateReport> GateArtifactAsync(string projectPath, string artifactId);

    /// <summary>Promote an artifact to "kept" through the engine (re-enforces the promote gate).</summary>
    Task<ModelArtifactRecord> PromoteArtifactAsync(string projectPath, string artifactId);

    /// <summary>Set an artifact's status directly (candidate/rejected; "kept" must use PromoteArtifactAsync).</summary>
    ModelArtifactRecord UpdateArtifactStatus(string projectPath, string artifactId, string status);

    /// <summary>Rebuild the on-disk project index (rescans the project root).</summary>
    Task<ProjectIndexRebuildResult> RebuildProjectIndexAsync();

    /// <summary>Load the project list from the index (newest first).</summary>
    Task<IReadOnlyList<DatasetProjectListItem>> LoadProjectsFromIndexAsync();

    /// <summary>Persist the edited gate thresholds (engine validates + rejects out-of-range values).</summary>
    Task SetGateThresholdsAsync(string projectPath, GateThresholds thresholds);

    /// <summary>List the project's provider generation policies (approved model allow-list).</summary>
    Task<IReadOnlyList<ProviderPolicyItem>> GetProviderPoliciesAsync(string projectPath);

    /// <summary>Approve (or revoke) a provider/model for generating trainable rows.</summary>
    Task ApproveProviderGenerationAsync(string projectPath, string providerId, string modelId, bool revoke = false);

    /// <summary>Build the dataset quality report (heuristics, PII/secret detection, debt signals).</summary>
    Task<QualityReport> BuildQualityReportAsync(string projectPath);

    /// <summary>Append a quality-history entry for the debt-trend chart.</summary>
    QualityHistoryEntry SaveQualityHistoryEntry(string projectPath, QualityReport report);

    /// <summary>Load the project's quality history (newest first) for the trend chart.</summary>
    IReadOnlyList<QualityHistoryEntry> LoadQualityHistory(string projectPath, int maxEntries = 5);

    /// <summary>The project's recorded training runs (newest first).</summary>
    IReadOnlyList<TrainingRunRecord> LoadTrainingRunRecords(string projectPath);

    /// <summary>Link an after-training eval report to the newest run (returns that run's id, or null).</summary>
    string? LinkAfterEvalToNewestRun(string projectPath, string afterEvalPath, string? afterEvalModel);

    /// <summary>Run the training-run regression gate for a recorded run.</summary>
    Task<GateReport> RunTrainingRunGateAsync(string projectPath, string runId);

    /// <summary>Generate deterministic train/validation/test splits over the project's examples.</summary>
    Task<SplitReport> GenerateProjectSplitsAsync(string projectPath, string schemaId, double trainRatio, double validationRatio, int seed);

    /// <summary>Persist the project's split settings (ratios + seed) to project.json.</summary>
    void SaveProjectSplitSettings(string projectPath, SplitSettings settings);

    /// <summary>Run a multi-model benchmark over the dataset (live backend calls).</summary>
    Task<BenchmarkReport> RunBenchmarkAsync(
        string projectPath, string schemaId, string backend, IReadOnlyList<string> models,
        string? baseUrl, int? limit, double scoreThreshold, int timeoutSeconds);

    /// <summary>Generate a training config export from the Training tab's options.</summary>
    Task<TrainingConfigExportResult> GenerateTrainingConfigAsync(
        string projectPath, string schemaId, string target, string baseModel, string datasetFormat,
        int sequenceLen, int loraR, int loraAlpha, int microBatchSize, int gradientAccumulationSteps,
        double learningRate);

    /// <summary>List the project's dataset versions (newest first) with live integrity.</summary>
    Task<IReadOnlyList<DatasetVersionDisplayItem>> LoadDatasetVersionsAsync(string projectPath);

    /// <summary>Capture the current dataset as a new version (the engine computes the fingerprint).</summary>
    Task<DatasetVersionRecord> CreateDatasetVersionAsync(string projectPath, string label, string trigger);

    /// <summary>Render a saved dataset-version card as Markdown (lineage + live integrity).</summary>
    Task<string> GetDatasetVersionCardAsync(string projectPath, string versionId);

    /// <summary>Diff two saved dataset versions (added/removed/common rows) as Markdown.</summary>
    Task<string> GetDatasetVersionDiffAsync(string projectPath, string baseVersionId, string otherVersionId);

    /// <summary>Render a model-artifact weight card as Markdown (byte-exact integrity at decision time).</summary>
    Task<string> GetWeightCardAsync(string projectPath, string artifactId);

    /// <summary>Generate the dataset card (metadata/schema/splits/quality/eval summary).</summary>
    Task<DatasetCardResult> GenerateDatasetCardAsync(string projectPath, string schemaId);

    /// <summary>Validate a draft against a schema (required fields, types, enums, chat shape, …).</summary>
    Task<ValidationReport> ValidateDraftAsync(string draftText, string schemaId);

    /// <summary>Check training-config compatibility for a schema/format/target with warnings.</summary>
    Task<TrainingCompatibilityResult> CheckTrainingCompatibilityAsync(string schemaId, string datasetFormat, string target);

    /// <summary>Scaffold a new evaluation suite file (evaluation_suites/&lt;name&gt;.json).</summary>
    Task NewSuiteAsync(string projectPath, string name);

    /// <summary>List the project's registered evaluation suites (name/case-count/validity).</summary>
    Task<IReadOnlyList<SuiteSummary>> ListSuitesAsync(string projectPath);

    /// <summary>Export a preference project for training (DPO/KTO/reward) with a pair-integrity gate.</summary>
    Task<PreferenceExportResult> ExportPreferenceForTrainingAsync(string projectPath, string format);
}
