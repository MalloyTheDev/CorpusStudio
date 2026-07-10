using System;
using System.Collections.Generic;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Training tab core view-model (Avalonia Phase 2, slice 4). Behaviour moved verbatim
/// from the shell (<c>MainWindowViewModel</c>) — the config-export inputs + preview + compatibility, the
/// launch command / live run log / run lifecycle, the run registry + regression gate, checkpoints/resume,
/// and the before/after baseline comparison. Holds the shared Evaluation VM so the baseline comparison can
/// reuse its report formatter; a config-export failure surfaces via <see cref="ErrorReported"/> (the shell
/// forwards it to its error banner). Honesty invariants intact: the VRAM/token numbers stay heuristic, the
/// regression gate reflects the keyword-overlap delta (a lexical proxy), and launches show the exact argv.</summary>
public sealed class TrainingViewModel : ViewModelBase, ITrainingViewModel
{
    private readonly IEvaluationViewModel _evaluation;

    /// <summary>Raised when a training config export fails; the shell forwards it to its error banner.</summary>
    public event Action<string>? ErrorReported;

    public TrainingViewModel(IEvaluationViewModel evaluation)
    {
        _evaluation = evaluation;
    }

    private string _trainingTarget = "axolotl_yaml";

    private string _trainingBaseModel = "Qwen/Qwen2.5-Coder-7B-Instruct";

    private string _trainingFormat = "instruction";

    private string _trainingSequenceLen = "4096";

    private string _trainingLoraR = "16";

    private string _trainingLoraAlpha = "32";

    private string _trainingMicroBatchSize = "1";

    private string _trainingGradientAccumulationSteps = "8";

    private string _trainingLearningRate = "0.0002";

    private string _trainingSummary =
        "Generate a training config after validation, splits, and evaluation checks.";

    private string _trainingConfigPreview = "Training config preview appears here.";

    private string _trainingLaunchCommand = string.Empty;

    private IReadOnlyList<string> _trainingLaunchArgv = [];
    private bool _preflightCanLaunch = true;

    private string _trainingLaunchWorkingDirectory = string.Empty;

    private string _trainingOutputDirectory = string.Empty;

    private string _trainingConfigPath = string.Empty;

    private IReadOnlyList<string> _trainingCheckpointNames = [];

    private string _trainingRunHistorySummary = "Refresh to see past training runs recorded for this project.";

    private string _trainingRunGateSummary = "Gate a run to check for regression vs its baseline.";

    private string _trainingEvalHandoffSummary =
        "Finish a run to see how to evaluate the model it produced (close the train→eval loop).";

    private string _trainingCheckpointsSummary =
        "Checkpoints appear here after a training run writes them.";

    private IReadOnlyList<string> _trainingResumeArgv = [];

    private string _trainingResumeCommand = string.Empty;

    private EvaluationReportHistoryItem? _trainingBaselineReport;

    private string _trainingComparisonSummary =
        "Run an evaluation before training to capture a baseline for before/after comparison.";

    private readonly List<string> _trainingRunLines = [];

    private int _trainingRunId;

    private string _trainingRunLog = "Launch training after generating a config; live logs appear here.";

    private string _trainingRunStatus = "Idle";

    public string TrainingTarget
    {
        get => _trainingTarget;
        set
        {
            if (SetField(ref _trainingTarget, value))
            {
                OnPropertyChanged(nameof(IsFirstPartyTarget));
            }
        }
    }

    public string TrainingBaseModel
    {
        get => _trainingBaseModel;
        set => SetField(ref _trainingBaseModel, value);
    }

    public string TrainingFormat
    {
        get => _trainingFormat;
        set => SetField(ref _trainingFormat, value);
    }

    public string TrainingSequenceLen
    {
        get => _trainingSequenceLen;
        set => SetField(ref _trainingSequenceLen, value);
    }

    public string TrainingLoraR
    {
        get => _trainingLoraR;
        set => SetField(ref _trainingLoraR, value);
    }

    public string TrainingLoraAlpha
    {
        get => _trainingLoraAlpha;
        set => SetField(ref _trainingLoraAlpha, value);
    }

    public string TrainingMicroBatchSize
    {
        get => _trainingMicroBatchSize;
        set => SetField(ref _trainingMicroBatchSize, value);
    }

    public string TrainingGradientAccumulationSteps
    {
        get => _trainingGradientAccumulationSteps;
        set => SetField(ref _trainingGradientAccumulationSteps, value);
    }

    public string TrainingLearningRate
    {
        get => _trainingLearningRate;
        set => SetField(ref _trainingLearningRate, value);
    }

    public string TrainingSummary
    {
        get => _trainingSummary;
        private set => SetField(ref _trainingSummary, value);
    }

    public string TrainingConfigPreview
    {
        get => _trainingConfigPreview;
        private set => SetField(ref _trainingConfigPreview, value);
    }

    public string TrainingLaunchCommand
    {
        get => _trainingLaunchCommand;
        private set => SetField(ref _trainingLaunchCommand, value);
    }

    public string TrainingRunLog
    {
        get => _trainingRunLog;
        private set => SetField(ref _trainingRunLog, value);
    }

    public string TrainingRunStatus
    {
        get => _trainingRunStatus;
        private set => SetField(ref _trainingRunStatus, value);
    }

    public IReadOnlyList<string> TrainingLaunchArgv => _trainingLaunchArgv;

    public string TrainingLaunchWorkingDirectory => _trainingLaunchWorkingDirectory;

    public const int TrainingLogMaxLines = 2000;

    public int BeginTrainingRun()
    {
        _trainingRunLines.Clear();
        TrainingRunLog = string.Empty;
        TrainingRunStatus = "Running...";
        IsTrainingRunning = true;
        return ++_trainingRunId;
    }

    public void AppendTrainingRunLog(string line)
    {
        _trainingRunLines.Add(line);
        TrimAndPublishTrainingLog();
    }

    public void AppendTrainingRunLogBatch(int runId, IReadOnlyList<string> lines)
    {
        if (runId != _trainingRunId || lines.Count == 0)
        {
            return;
        }

        _trainingRunLines.AddRange(lines);
        TrimAndPublishTrainingLog();
    }

    private void TrimAndPublishTrainingLog()
    {
        if (_trainingRunLines.Count > TrainingLogMaxLines)
        {
            _trainingRunLines.RemoveRange(0, _trainingRunLines.Count - TrainingLogMaxLines);
        }

        TrainingRunLog = string.Join(Environment.NewLine, _trainingRunLines);
    }

    public void CompleteTrainingRun(int exitCode)
    {
        IsTrainingRunning = false;
        TrainingRunStatus = exitCode == 0
            ? "Completed (exit 0)"
            : $"Failed (exit {exitCode})";
    }

    public void SetTrainingRunCancelled()
    {
        IsTrainingRunning = false;
        TrainingRunStatus = "Cancelled";
    }

    public void SetTrainingRunError(string message)
    {
        IsTrainingRunning = false;
        TrainingRunStatus = "Error";
        AppendTrainingRunLog($"[error] {message}");
    }

    public string TrainingRunHistorySummary
    {
        get => _trainingRunHistorySummary;
        private set => SetField(ref _trainingRunHistorySummary, value);
    }

    public string TrainingRunGateSummary
    {
        get => _trainingRunGateSummary;
        private set => SetField(ref _trainingRunGateSummary, value);
    }

    public void SetTrainingRunGateError(string message)
    {
        TrainingRunGateSummary = $"Regression gate could not run.{Environment.NewLine}{message}";
    }

    public void ApplyTrainingRunGate(GateReport report)
    {
        var mark = report.OverallStatus switch
        {
            "block" => "⛔ BLOCK",
            "warn" => "⚠ WARN",
            _ => "✅ PASS",
        };
        var result = report.Results.Count > 0 ? report.Results[0] : null;
        TrainingRunGateSummary = result is null
            ? $"Regression gate: {mark}"
            : $"Regression gate: {mark} — {result.Message}";
    }

    public void SetTrainingRunHistoryError(string message)
    {
        TrainingRunHistorySummary = $"Run history could not load.{Environment.NewLine}{message}";
    }

    public string TrainingEvalHandoffSummary
    {
        get => _trainingEvalHandoffSummary;
        private set => SetField(ref _trainingEvalHandoffSummary, value);
    }

    public void SetEvalHandoffError(string message)
    {
        TrainingEvalHandoffSummary = $"The eval plan could not load.{Environment.NewLine}{message}";
    }

    /// <summary>Render the close-the-loop plan for a finished run. Not-ready runs (still
    /// running / failed) show the plan's honest note instead of steps. Pure + testable.</summary>
    public void ApplyEvalHandoff(EvalHandoffPlan plan)
    {
        if (!plan.Ready)
        {
            TrainingEvalHandoffSummary = string.IsNullOrWhiteSpace(plan.Note)
                ? "No finished run to evaluate yet."
                : plan.Note;
            return;
        }

        var lines = new List<string>
        {
            $"Evaluate the model from run {plan.RunId} — serving is external; the eval/link/gate commands are exact:",
            string.Empty,
        };
        var index = 1;
        foreach (var step in plan.Steps)
        {
            lines.Add($"{index}. {step.Title}");
            lines.Add($"   {step.Detail}");
            if (!string.IsNullOrWhiteSpace(step.Command))
            {
                lines.Add($"   $ {step.Command}");
            }
            lines.Add(string.Empty);
            index++;
        }

        TrainingEvalHandoffSummary = string.Join(Environment.NewLine, lines).TrimEnd();
    }

    public void ApplyTrainingRunHistory(IReadOnlyList<TrainingRunRecord> records)
    {
        if (records.Count == 0)
        {
            TrainingRunHistorySummary = "No training runs recorded yet.";
            return;
        }

        var lines = new List<string> { $"Training runs ({records.Count}, newest first):", string.Empty };
        foreach (var record in records)
        {
            lines.Add($"[{record.Status}] {record.RunId} — {record.BaseModel} ({record.Target})");
            var bits = new List<string>
            {
                $"{record.Checkpoints?.Count ?? 0} checkpoint(s)",
                string.IsNullOrWhiteSpace(record.BeforeEvalPath) ? "before-eval –" : "before-eval ✓",
                string.IsNullOrWhiteSpace(record.AfterEvalPath) ? "after-eval –" : "after-eval ✓",
            };
            if (record.ExitCode is { } exit)
            {
                bits.Add($"exit {exit}");
            }
            lines.Add("   " + string.Join("; ", bits));

            // Reproducibility manifest: the recipe (data + config + engine) behind this run.
            if (record.Provenance is { } prov)
            {
                var provBits = new List<string>();
                if (prov.DatasetFingerprint is { Length: > 0 } fingerprint)
                {
                    provBits.Add($"data {fingerprint[..Math.Min(12, fingerprint.Length)]} ({prov.DatasetRowCount} rows)");
                }
                if (prov.ConfigSha256 is { Length: > 0 } configHash)
                {
                    provBits.Add($"config {configHash[..Math.Min(12, configHash.Length)]}");
                }
                if (!string.IsNullOrWhiteSpace(prov.EngineVersion))
                {
                    provBits.Add($"engine {prov.EngineVersion}");
                }
                if (provBits.Count > 0)
                {
                    lines.Add("   provenance: " + string.Join("; ", provBits));
                }
            }
        }

        TrainingRunHistorySummary = string.Join(Environment.NewLine, lines);
    }

    public string TrainingOutputDirectory => _trainingOutputDirectory;

    public string TrainingConfigPath => _trainingConfigPath;

    public string TrainingCheckpointsSummary
    {
        get => _trainingCheckpointsSummary;
        private set => SetField(ref _trainingCheckpointsSummary, value);
    }

    public string TrainingResumeCommand => _trainingResumeCommand;

    public IReadOnlyList<string> TrainingResumeArgv => _trainingResumeArgv;

    public string TrainingComparisonSummary
    {
        get => _trainingComparisonSummary;
        private set => SetField(ref _trainingComparisonSummary, value);
    }

    public EvaluationReportHistoryItem? TrainingBaselineReport => _trainingBaselineReport;

    public void SetTrainingBaseline(EvaluationReportHistoryItem? baseline)
    {
        _trainingBaselineReport = baseline;
        TrainingComparisonSummary = baseline is null
            ? "No baseline: no evaluation report existed when this run started. "
              + "Run an evaluation before the next training run to enable before/after comparison."
            : $"Baseline captured: {baseline.DisplayName}{Environment.NewLine}"
              + "After training: load the trained adapter into your local backend, run an "
              + "evaluation against it, then click Compare vs baseline.";
    }

    public void CompareTrainingBaseline(IReadOnlyList<EvaluationReportHistoryItem> history)
    {
        if (_trainingBaselineReport is null)
        {
            TrainingComparisonSummary =
                "No baseline was captured for the last training run. Run an evaluation, "
                + "train, then evaluate the trained model to compare.";
            return;
        }

        var after = history.FirstOrDefault(item => !string.Equals(
            item.ReportPath,
            _trainingBaselineReport.ReportPath,
            StringComparison.OrdinalIgnoreCase
        ));
        if (after is null)
        {
            TrainingComparisonSummary =
                "No post-training evaluation found yet. Load the trained adapter into your "
                + "local backend and run an evaluation, then compare again.";
            return;
        }

        if (after.LastModified < _trainingBaselineReport.LastModified)
        {
            TrainingComparisonSummary =
                "The newest other report is older than the baseline. Run an evaluation of "
                + "the trained model first, then compare again.";
            return;
        }

        TrainingComparisonSummary =
            $"Before/after (after − before):{Environment.NewLine}"
            + _evaluation.BuildEvaluationReportComparison(after, _trainingBaselineReport);
    }

    public IReadOnlyList<string> TrainingCheckpointNames => _trainingCheckpointNames;

    public void ApplyTrainingCheckpoints(TrainingCheckpointsResult result)
    {
        _trainingCheckpointNames = result.Checkpoints.ToArray();
        if (result.Checkpoints.Count == 0)
        {
            TrainingCheckpointsSummary = "No checkpoints found yet.";
            _trainingResumeArgv = [];
            _trainingResumeCommand = string.Empty;
        }
        else
        {
            TrainingCheckpointsSummary =
                $"Checkpoints: {result.Checkpoints.Count} (latest {result.LatestCheckpoint})";
            var resumeReady = result.ResumeSupported == true
                && result.LatestCheckpoint is not null
                && result.ResumeArgv is { Count: > 0 };
            _trainingResumeArgv = resumeReady ? result.ResumeArgv!.ToArray() : [];
            _trainingResumeCommand = resumeReady ? result.ResumeCommand ?? string.Empty : string.Empty;
            if (!resumeReady && result.ResumeSupported == false)
            {
                TrainingCheckpointsSummary +=
                    " — resume is config-driven for this target; set the checkpoint in the config.";
            }
        }

        OnPropertyChanged(nameof(CanResumeTraining));
    }

    public void SetTrainingConfigInProgress()
    {
        TrainingSummary = string.Join(
            Environment.NewLine,
            [
                "Generating training config...",
                $"Target: {TrainingTarget}",
                $"Base model: {TrainingBaseModel}",
            ]
        );
        TrainingConfigPreview = "Waiting for config export.";
    }

    public void ApplyTrainingConfigExportResult(TrainingConfigExportResult result)
    {
        var launcherStatus = result.TrainingLauncherImplemented ? "implemented" : "not implemented";
        var lines = new List<string>
        {
            $"Target: {result.Target}",
            $"Config: {result.OutputPath}",
            $"Training launcher: {launcherStatus}",
        };

        if (result.TokenBudget is { } budget && budget.ExampleCount > 0)
        {
            lines.Add("");
            lines.Add(
                $"Token budget ({budget.Method}): ~{budget.EstimatedTokens:N0} tokens over "
                + $"{budget.ExampleCount} example(s), ~{budget.TokensPerEpoch:N0}/epoch at seq_len "
                + $"{budget.SequenceLen}");
            lines.Add(
                $"  mean ~{budget.MeanTokensPerExample:N0}, max ~{budget.MaxTokensInExample:N0} tokens; "
                + $"{budget.ExamplesOverSequenceLen} over seq_len");
        }

        if (result.VramEstimate is { } vram)
        {
            lines.Add("");
            if (vram.ParameterCountBillions is { } paramsB)
            {
                lines.Add(
                    $"VRAM (rough, {paramsB:0.#}B params): ~{vram.TotalGbFp16:0.#} GB fp16 / "
                    + $"~{vram.TotalGbInt8:0.#} GB 8-bit / ~{vram.TotalGbInt4:0.#} GB 4-bit");
            }
            else
            {
                lines.Add("VRAM: no estimate (model size not parseable from the name).");
            }
        }

        if (result.LoraRecommendation is { } lora)
        {
            lines.Add($"LoRA suggestion: r={lora.RecommendedR}, alpha={lora.RecommendedAlpha}");
            lines.AddRange(lora.Warnings.Select(warning => $"- {warning}"));
        }

        _trainingOutputDirectory = result.TrainingOutputDirectory;
        _trainingConfigPath = result.OutputPath;
        _trainingResumeArgv = [];
        _trainingResumeCommand = string.Empty;
        TrainingCheckpointsSummary = "Refresh checkpoints after a run writes them.";
        OnPropertyChanged(nameof(CanResumeTraining));
        OnPropertyChanged(nameof(CanMergeAdapter));

        if (result.Launch is { } launch && !string.IsNullOrWhiteSpace(launch.Command))
        {
            TrainingLaunchCommand = launch.Command;
            _trainingLaunchArgv = launch.Argv.ToArray();
            _trainingLaunchWorkingDirectory = string.IsNullOrWhiteSpace(result.OutputPath)
                ? string.Empty
                : (System.IO.Path.GetDirectoryName(result.OutputPath) ?? string.Empty);
            lines.Add("");
            lines.Add("Launch command (review before running):");
            lines.Add($"  {launch.Command}");
            if (launch.ResumeSupported)
            {
                lines.Add($"  resume: {launch.ResumeCommand}");
            }
            if (launch.Dependencies.Count > 0)
            {
                lines.Add($"  requires: {string.Join(", ", launch.Dependencies)}");
            }
        }
        else
        {
            TrainingLaunchCommand = string.Empty;
            _trainingLaunchArgv = [];
            _trainingLaunchWorkingDirectory = string.Empty;
        }

        // Pre-flight: cheap fail-fast checks; a BLOCK (missing config/data, empty dataset)
        // disables the Launch button so a certain-to-fail run can't be started.
        _preflightCanLaunch = result.Preflight?.CanLaunch ?? true;
        if (result.Preflight is { } preflight)
        {
            lines.Add("");
            lines.Add(preflight.Status switch
            {
                "block" => "Pre-flight: BLOCKED — fix the item(s) below before launching:",
                "warn" => "Pre-flight (warnings — review before launching):",
                _ => "Pre-flight: all checks passed.",
            });
            foreach (var check in preflight.Checks)
            {
                var mark = check.Status switch { "block" => "[x]", "warn" => "[!]", _ => "[ok]" };
                lines.Add($"  {mark} {check.Message}");
            }
            if (!preflight.CanLaunch)
            {
                lines.Add("  Launch is disabled until the blocking item(s) are resolved.");
            }
        }

        OnPropertyChanged(nameof(CanLaunchTraining));

        if (result.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(result.Warnings.Select(warning => $"- {warning}"));
        }

        TrainingSummary = string.Join(Environment.NewLine, lines);
        TrainingConfigPreview = result.ConfigText;
    }

    public void SetTrainingConfigError(string message)
    {
        TrainingSummary = $"Training config could not be generated.{Environment.NewLine}{message}";
        TrainingConfigPreview = "No training config was generated.";
        ErrorReported?.Invoke(message);
    }

    public void ApplyTrainingCompatibility(TrainingCompatibilityResult result)
    {
        if (result.Compatible)
        {
            TrainingSummary =
                $"Compatible: {result.Schema} / {result.Format} → {result.Target}. "
                + "No compatibility warnings — safe to generate.";
            return;
        }

        TrainingSummary = string.Join(
            Environment.NewLine,
            new[]
            {
                $"Compatibility warnings for {result.Schema} / {result.Format} → {result.Target}:",
            }.Concat(result.Warnings.Select(warning => $"- {warning}"))
        );
    }
    /// <summary>Reset the config inputs/preview on a project switch (the format follows the project's
    /// schema). Matches the shell's prior SelectProject behaviour verbatim.</summary>
    public void Reset(string schemaId)
    {
        TrainingFormat = schemaId;
        TrainingSummary = "Generate a training config after validation, splits, and evaluation checks.";
        TrainingConfigPreview = "Training config preview appears here.";
    }

    // ---- run-state gating (missed by the first pass; moved verbatim) ----

    private bool _isTrainingRunning;

    public bool IsTrainingRunning
    {
        get => _isTrainingRunning;
        private set
        {
            if (SetField(ref _isTrainingRunning, value))
            {
                OnPropertyChanged(nameof(CanLaunchTraining));
                OnPropertyChanged(nameof(CanResumeTraining));
                OnPropertyChanged(nameof(CanMergeAdapter));
            }
        }
    }

    public bool CanLaunchTraining => !_isTrainingRunning && _trainingLaunchArgv.Count > 0 && _preflightCanLaunch;

    public bool CanResumeTraining => !_isTrainingRunning && _trainingResumeArgv.Count > 0;

    // ---- First-party trainer (the opt-in [train] extra; runs in-process) ----

    private bool _cpuToyMode;

    private string _trainingRuntimeSummary =
        "Check the training runtime (train-check) to see whether a real GPU QLoRA — or only the CPU toy "
        + "smoke path — can run on this machine.";

    private string _trainingMergeSummary =
        "After a first-party run, merge the adapter into its base (train-merge). 'auto' falls back to a "
        + "CPU-offload merge or adapter-only serving when the GPU is too small (a 7B fp16 merge ≈14 GB).";

    /// <summary>True when the selected target is Corpus Studio's own trainer (accepts the aliases the
    /// engine normalizes: corpus_studio / corpus-studio / corpusstudio / corpus / first_party).</summary>
    public bool IsFirstPartyTarget
    {
        get
        {
            var normalized = _trainingTarget.Trim().Replace('-', '_').ToLowerInvariant();
            return normalized is "corpus_studio" or "corpusstudio" or "corpus" or "first_party" or "firstparty";
        }
    }

    public bool CpuToyMode
    {
        get => _cpuToyMode;
        set => SetField(ref _cpuToyMode, value);
    }

    public string TrainingRuntimeSummary
    {
        get => _trainingRuntimeSummary;
        private set => SetField(ref _trainingRuntimeSummary, value);
    }

    public string TrainingMergeSummary
    {
        get => _trainingMergeSummary;
        private set => SetField(ref _trainingMergeSummary, value);
    }

    /// <summary>The merge target is the run's output dir (it holds the adapter). Gate on it existing and
    /// no run being active; the engine reports a clear "no adapter yet" error if merged too early.</summary>
    public bool CanMergeAdapter => !_isTrainingRunning && !string.IsNullOrWhiteSpace(_trainingOutputDirectory);

    /// <summary>Render the train-check runtime report (verdict + per-dep presence + GPU + notes). Pure.</summary>
    public void ApplyTrainingRuntime(TrainingRuntimeReport report)
    {
        var verdict = report.Ready
            ? "READY (GPU QLoRA)"
            : report.CpuToyReady ? "CPU-TOY ONLY" : "NOT READY";
        var lines = new List<string> { $"Training runtime: {verdict}" };
        foreach (var (package, version) in report.Installed)
        {
            lines.Add($"  {(version is null ? "--" : "ok")} {package}: {version ?? "not installed"}");
        }
        lines.Add(report.Gpu.Available
            ? $"  GPU: {report.Gpu.Name} ({report.Gpu.TotalMemoryGb:0.#} GB, {report.Gpu.DeviceCount} device(s))"
            : "  GPU: none detected");
        lines.AddRange(report.Notes.Select(note => $"  • {note}"));
        if (!report.Ready && !report.CpuToyReady && !string.IsNullOrWhiteSpace(report.InstallHint))
        {
            lines.Add($"  {report.InstallHint}");
        }

        TrainingRuntimeSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetTrainingRuntimeError(string message)
    {
        TrainingRuntimeSummary = $"The training runtime check could not run.{Environment.NewLine}{message}";
    }

    public void SetMergeInProgress()
    {
        TrainingMergeSummary =
            "Merging the adapter into its base… (loading the base can take a while; 'auto' falls back on "
            + "a small GPU).";
    }

    /// <summary>Render the train-merge result. adapter-only (not merged) is a valid outcome on a small
    /// GPU — surface the serving note, not an error. Pure.</summary>
    public void ApplyMergeResult(MergeResult result)
    {
        var lines = new List<string>();
        if (result.Merged)
        {
            lines.Add($"Merged ({result.Strategy}) → {result.OutputPath}");
            if (!string.IsNullOrWhiteSpace(result.BaseModel))
            {
                lines.Add($"Base: {result.BaseModel}");
            }
        }
        else
        {
            lines.Add($"Not merged (strategy: {result.Strategy}) — serve the base model with the adapter applied:");
        }
        lines.AddRange(result.Notes);
        TrainingMergeSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetMergeError(string message)
    {
        TrainingMergeSummary = $"The adapter merge could not run.{Environment.NewLine}{message}";
    }

    // ---- Base-model download (model-fetch) + adapter model card (model-card) ----

    private string _modelFetchRepoId = string.Empty;
    private bool _isFetchingModel;

    private string _modelFetchSummary =
        "Download a base model from the Hugging Face Hub to train on. Prefer a permissive (MIT/Apache) "
        + "license — the base model's license governs what you may do with the trained result.";

    private string _modelCardText =
        "After a first-party run, view the trained adapter's model card here (base model + license, LoRA "
        + "config, honesty notes). train-run also writes MODEL_CARD.md next to the adapter.";

    public string ModelFetchRepoId
    {
        get => _modelFetchRepoId;
        set
        {
            if (SetField(ref _modelFetchRepoId, value))
            {
                OnPropertyChanged(nameof(CanFetchModel));
            }
        }
    }

    public string ModelFetchSummary
    {
        get => _modelFetchSummary;
        private set => SetField(ref _modelFetchSummary, value);
    }

    public bool IsFetchingModel
    {
        get => _isFetchingModel;
        private set
        {
            if (SetField(ref _isFetchingModel, value))
            {
                OnPropertyChanged(nameof(CanFetchModel));
            }
        }
    }

    public bool CanFetchModel => !_isFetchingModel && !string.IsNullOrWhiteSpace(_modelFetchRepoId);

    public string ModelCardText
    {
        get => _modelCardText;
        private set => SetField(ref _modelCardText, value);
    }

    public void SetModelFetchInProgress()
    {
        IsFetchingModel = true;
        ModelFetchSummary = string.IsNullOrWhiteSpace(_modelFetchRepoId)
            ? "Downloading… (resumable)"
            : $"Downloading {_modelFetchRepoId} (resumable — survives dropped connections)…";
    }

    public void AppendModelFetchProgress(string line)
    {
        if (!string.IsNullOrWhiteSpace(line))
        {
            ModelFetchSummary = line; // show the latest progress line while the download runs
        }
    }

    /// <summary>Render a completed model-fetch: repo, LICENSE (permissive or not — the honest, fail-closed
    /// classification), local path, size, and any warnings (non-permissive license, pickle-only weights).</summary>
    public void ApplyModelFetch(ModelFetchResult result)
    {
        IsFetchingModel = false;
        var licenseNote = result.LicensePermissive
            ? "permissive — OK to train on and redistribute the result"
            : "NOT clearly permissive — verify the repo's terms before training/redistributing";
        var lines = new List<string>
        {
            $"Downloaded {result.RepoId}" + (string.IsNullOrWhiteSpace(result.Revision) ? "" : $"@{result.Revision}"),
            $"License: {result.License ?? "unknown"} — {licenseNote}",
            $"Path: {result.LocalPath}",
            $"{result.WeightFiles.Count} weight file(s), ~{result.TotalSizeMb:N1} MB",
        };
        lines.AddRange(result.Warnings.Select(warning => $"⚠ {warning}"));
        ModelFetchSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetModelFetchError(string message)
    {
        IsFetchingModel = false;
        ModelFetchSummary = $"The model download failed.{Environment.NewLine}{message}";
    }

    public void SetModelCardInProgress()
    {
        ModelCardText = "Generating the model card…";
    }

    public void ApplyModelCard(string cardMarkdown)
    {
        ModelCardText = string.IsNullOrWhiteSpace(cardMarkdown) ? "(the model card was empty)" : cardMarkdown;
    }

    public void SetModelCardError(string message)
    {
        ModelCardText = $"The model card could not be generated.{Environment.NewLine}{message}";
    }

    /// <summary>The launch decision for the first-party trainer, given a fresh train-check report and
    /// whether CPU-toy was requested. PURE + unit-tested — encodes the honesty rules: a real GPU run
    /// needs <c>ready</c>; the CPU smoke path needs <c>cpu_toy_ready</c>; otherwise block with the
    /// install hint (never silently downgrade a real run to a toy run).</summary>
    public static FirstPartyLaunchDecision DecideFirstPartyLaunch(TrainingRuntimeReport report, bool cpuToyRequested)
    {
        if (cpuToyRequested)
        {
            return report.CpuToyReady
                ? new FirstPartyLaunchDecision(true, true,
                    "CPU toy smoke test: a tiny model + a few steps. It proves the training pipeline runs — "
                    + "it does NOT train a usable model.")
                : new FirstPartyLaunchDecision(false, false,
                    "The CPU toy path needs torch + transformers + trl + peft + datasets + accelerate. "
                    + report.InstallHint);
        }

        if (report.Ready)
        {
            return new FirstPartyLaunchDecision(true, false, "Ready: a 4-bit QLoRA GPU run is possible.");
        }

        if (report.CpuToyReady)
        {
            return new FirstPartyLaunchDecision(false, false,
                "No GPU QLoRA runtime detected (only the CPU toy path is available). Tick 'CPU toy' to run "
                + "the smoke test, or install a CUDA torch build + bitsandbytes for a real run.");
        }

        return new FirstPartyLaunchDecision(false, false,
            "The first-party training runtime isn't installed. " + report.InstallHint);
    }
}

/// <summary>The outcome of the first-party launch gate: whether to launch, whether it is the CPU-toy
/// smoke path (so the argv gets <c>--cpu-toy</c>), and the message to show the user.</summary>
public readonly record struct FirstPartyLaunchDecision(bool CanLaunch, bool CpuToy, string Message);
