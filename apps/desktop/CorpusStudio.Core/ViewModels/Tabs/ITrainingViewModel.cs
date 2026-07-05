using System;
using System.Collections.Generic;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Training tab view-model (Avalonia Phase 2, slice 4). Owns the training-config export
/// inputs + preview + compatibility, the launch command / live run log / run lifecycle, the run registry
/// + regression gate, checkpoints/resume, and the before/after baseline comparison.
///
/// <para>It holds the shared Evaluation VM (for the baseline comparison's report formatter). The shell
/// keeps the launch-prep bridges that read the engine / Evaluation report history and feed
/// <see cref="SetTrainingBaseline"/> / <see cref="CompareTrainingBaseline"/>. A config-export failure
/// surfaces via <see cref="ErrorReported"/> (the shell forwards it to its error banner). Behind an
/// interface so the shell/tests/DI depend on the contract.</para></summary>
public interface ITrainingViewModel : INotifyPropertyChanged
{
    event Action<string>? ErrorReported;

    // Config-export inputs (two-way bound).
    string TrainingTarget { get; set; }
    string TrainingBaseModel { get; set; }
    string TrainingFormat { get; set; }
    string TrainingSequenceLen { get; set; }
    string TrainingLoraR { get; set; }
    string TrainingLoraAlpha { get; set; }
    string TrainingMicroBatchSize { get; set; }
    string TrainingGradientAccumulationSteps { get; set; }
    string TrainingLearningRate { get; set; }

    string TrainingSummary { get; }
    string TrainingConfigPreview { get; }
    string TrainingLaunchCommand { get; }
    string TrainingRunLog { get; }
    string TrainingRunStatus { get; }
    IReadOnlyList<string> TrainingLaunchArgv { get; }
    string TrainingLaunchWorkingDirectory { get; }
    string TrainingRunHistorySummary { get; }
    string TrainingRunGateSummary { get; }
    string TrainingOutputDirectory { get; }
    string TrainingConfigPath { get; }
    string TrainingCheckpointsSummary { get; }
    string TrainingResumeCommand { get; }
    IReadOnlyList<string> TrainingResumeArgv { get; }
    string TrainingComparisonSummary { get; }
    EvaluationReportHistoryItem? TrainingBaselineReport { get; }
    IReadOnlyList<string> TrainingCheckpointNames { get; }

    /// <summary>True while a launched run is active; gates the launch/resume affordances.</summary>
    bool IsTrainingRunning { get; }
    bool CanLaunchTraining { get; }
    bool CanResumeTraining { get; }

    int BeginTrainingRun();
    void AppendTrainingRunLog(string line);
    void AppendTrainingRunLogBatch(int runId, IReadOnlyList<string> lines);
    void CompleteTrainingRun(int exitCode);
    void SetTrainingRunCancelled();
    void SetTrainingRunError(string message);
    void SetTrainingRunGateError(string message);
    void ApplyTrainingRunGate(GateReport report);
    void SetTrainingRunHistoryError(string message);
    void ApplyTrainingRunHistory(IReadOnlyList<TrainingRunRecord> records);
    void SetTrainingBaseline(EvaluationReportHistoryItem? baseline);
    void CompareTrainingBaseline(IReadOnlyList<EvaluationReportHistoryItem> history);
    void ApplyTrainingCheckpoints(TrainingCheckpointsResult result);
    void SetTrainingConfigInProgress();
    void ApplyTrainingConfigExportResult(TrainingConfigExportResult result);
    void SetTrainingConfigError(string message);
    void ApplyTrainingCompatibility(TrainingCompatibilityResult result);

    /// <summary>Reset the config inputs/preview on a project switch (format follows the schema).</summary>
    void Reset(string schemaId);
}
