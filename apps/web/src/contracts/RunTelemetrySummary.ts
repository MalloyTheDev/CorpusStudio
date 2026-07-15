/* GENERATED from docs/contracts/RunTelemetrySummary.schema.json — do not edit. Run: npm run gen:contracts */

export type DegradedSampleCount = number;
export type MissingRequiredPaperFields = string[];
export type Reason = string;
export type ScientificallyComplete = boolean;
export type TelemetryDegraded = boolean;
export type ContractVersion = "1.0.0";
export type BoundaryClipped = boolean;
export type CoverageFraction = number | null;
export type EnergyPer1000NonpaddingTokens = number | null;
export type JoulesPerMeasuredOptimizerStep = number | null;
export type MaxPowerWatts = number | null;
export type MeasuredWindowJoules = number | null;
export type MedianMeasuredPowerWatts = number | null;
export type Method = "trapezoidal-power-over-monotonic-time-v1";
export type PowerSampleCount = number;
export type RunJoules = number | null;
export type TimeWeightedMeanPowerWatts = number | null;
export type GeneratedAt = string;
export type DeviceIndex = number | null;
export type DeviceUuid = string | null;
export type Count = number;
export type Maximum = number | null;
export type Mean = number | null;
export type Median = number | null;
export type Minimum = number | null;
export type SampleStandardDeviation = number | null;
export type Unit = string;
export type MaxRunTemperatureC = number | null;
export type CudaDeviceFreeBytes = number | null;
export type CudaDeviceUsedBytes = number | null;
export type DedicatedGpuBytes = number | null;
export type ProcessRssBytes = number | null;
export type SharedGpuBytes = number | null;
export type SystemRamUsedBytes = number | null;
export type TorchAllocatedBytes = number | null;
export type TorchPeakAllocatedBytes = number | null;
export type TorchPeakReservedBytes = number | null;
export type TorchReservedBytes = number | null;
export type ObservedPerformanceStates = string[];
export type ObservedThrottleReasons = string[];
export type StartingTemperatureC = number | null;
export type DiskReadDeltaBytes = number | null;
export type DiskWriteDeltaBytes = number | null;
export type SwapUsedDeltaBytes = number | null;
export type AmendmentId = string | null;
export type CapabilityReportHash = string | null;
export type CellId = string | null;
export type ChatTemplateSha256 = string | null;
export type DatasetFingerprint = string | null;
export type EffectiveMatrixSha256 = string | null;
export type EnvironmentLockHash = string | null;
export type ExecutionConfigurationHash = string | null;
export type ExecutionProbeHash = string | null;
export type ModelRef = string | null;
export type PlanHash = string | null;
export type PlanId = string | null;
export type ProtocolVersion = string | null;
export type RepositoryCommit = string | null;
export type RunId = string | null;
export type SequenceView = number | null;
export type StudyId = string | null;
export type TokenizerRef = string | null;
export type TrialId = string | null;
export type WorkerWheelSha256 = string | null;
/**
 * Ordered lifecycle stage of a run, launch → export. A RunEvent carries the stage it belongs to
 * so a consumer can render a precise progress spine and localize a failure to the exact stage.
 */
export type StageMarker =
  | "process_start"
  | "dataset_verification"
  | "execution_config_verified"
  | "env_loaded"
  | "cuda_init"
  | "tokenizer_load"
  | "dataset_formatting"
  | "truncation_analysis"
  | "attention_policy_applied"
  | "model_load"
  | "placement_verified"
  | "placement_deviation"
  | "model_loaded"
  | "quantized"
  | "adapter_attached"
  | "optimizer_created"
  | "batch_materialized"
  | "forward"
  | "loss"
  | "backward"
  | "optimizer_step"
  | "checkpoint"
  | "reload"
  | "evaluation"
  | "export";
/**
 * Terminal outcome category. ``PASS`` is included so the same enum classifies a completed
 * probe/run, not only failures. Grounded in the exact hazards the engine documents: the sm_120
 * fused-attention deadlock (KERNEL_STALL), the WDDM silent spill (ACCIDENTAL_SPILL vs a clean
 * OOM), and env/dependency mismatches (ENVIRONMENT_FAILURE).
 */
export type FailureTaxonomy =
  | "PASS"
  | "FAIL"
  | "OOM"
  | "TIMEOUT"
  | "KERNEL_STALL"
  | "NUMERICAL_FAILURE"
  | "GRADIENT_FAILURE"
  | "LOSS_EVIDENCE_FAILURE"
  | "OPTIMIZER_FAILURE"
  | "UPDATE_FAILURE"
  | "ARTIFACT_FAILURE"
  | "CHECKPOINT_FAILURE"
  | "ENVIRONMENT_FAILURE"
  | "UNSUPPORTED_CONFIGURATION"
  | "ACCIDENTAL_SPILL"
  | "CONTROLLED_OFFLOAD";
/**
 * ``math``/``eager`` is forced on native-Windows/WDDM Blackwell sm_120 because the fused flash
 * kernel deadlocks there. Other platforms require their own functional capability result; WSL
 * evidence is not bare-Linux proof.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
/**
 * The fit verdict. ``NATIVE_*`` = fully resident. ``CONTROLLED_*`` = a deliberate, planned
 * offload (acceptable, slower). ``ACCIDENTAL_*`` / ``THRASHING`` = an unplanned spill the platform
 * did silently (the failure mode the engine warns about). ``FAIL`` = will not run.
 */
export type FitClass =
  | "PLANNED_UNPROVEN"
  | "NATIVE_SAFE"
  | "NATIVE_TIGHT"
  | "NATIVE_UNPROVEN"
  | "MARGINAL"
  | "CONTROLLED_ACTIVATION_OFFLOAD"
  | "CONTROLLED_OPTIMIZER_OFFLOAD"
  | "CONTROLLED_PARAMETER_OFFLOAD"
  | "ACCIDENTAL_UNIFIED_MEMORY_PAGING"
  | "ACCIDENTAL_WDDM_SPILL"
  | "THRASHING"
  | "FAIL";
export type ContractVersion1 = "1.0.0";
export type DeviceCapacityBytes = number | null;
export type EstimatedPeakBytes = number | null;
export type HeadroomBytes = number | null;
export type Rationale = string;
export type State = "prepared" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";
export type TrainingSuccess = boolean;
export type Method1 = "cumulative-sampler-busy-time-v1";
export type OverheadFractionOfWall = number | null;
export type PerSampleMeanSeconds = number | null;
export type TotalSamplerSeconds = number | null;
export type WallSeconds = number | null;
export type LineCount = number;
export type Path = string;
export type RecordKind = "run_events" | "telemetry_samples" | "run_manifest";
export type Sha256 = string;
export type RawRecords = RawRecordBinding[];
export type RunId1 = string;
export type ObservedMaxIntervalMs = number | null;
export type ObservedMedianIntervalMs = number | null;
export type ObservedMinIntervalMs = number | null;
export type RequestedIntervalMs = number | null;
export type SampleCount = number;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type SchemaKind = "run-telemetry-summary-v1";
export type ChangedAdapterTensorCount = number | null;
export type CompletedOptimizerSteps = number;
export type FirstLoss = number | null;
export type GradientObservedTensorCount = number | null;
export type LastLoss = number | null;
export type MeasuredOptimizerSteps = number[];
export type MinLoss = number | null;
export type OptimizerStepsPerMinute = number | null;
export type Loss = number;
export type OptimizerStep = number;
export type StepLosses = OptimizerStepLossEvidence[];
export type TrainableStateAfterSha256 = string | null;
export type TrainableStateBeforeSha256 = string | null;
export type WarmupOptimizerSteps = number[];

/**
 * The single, paper-facing summary of one run's measurement. Every field is DERIVED from the raw
 * records named in ``raw_records`` (durable ``RunEvent`` stream + ``TelemetrySample`` series +
 * ``RunManifest``); CSV, tables, and plot series all render from this same derived object so they
 * can never disagree with each other or with the raw source.
 */
export interface RunTelemetrySummary {
  completeness: ScientificCompleteness;
  contract_version?: ContractVersion;
  energy: EnergyIntegration;
  generated_at: GeneratedAt;
  gpu?: GpuTelemetrySummary | null;
  host?: HostTelemetrySummary | null;
  identity: TelemetryIdentity;
  outcome: RunOutcomeSummary;
  overhead: MeasurementOverhead;
  raw_records?: RawRecords;
  run_id: RunId1;
  sampling: SamplingCadence;
  schema_kind?: SchemaKind;
  step: StepTelemetrySummary;
}
/**
 * Whether the run captured every field the paper requires. A telemetry gap NEVER converts a
 * workload success into paper data: ``scientifically_complete`` may be False even when the run
 * succeeded, and it does not alter the run's terminal state.
 */
export interface ScientificCompleteness {
  degraded_sample_count?: DegradedSampleCount;
  missing_required_paper_fields?: MissingRequiredPaperFields;
  reason?: Reason;
  scientifically_complete: ScientificallyComplete;
  telemetry_degraded?: TelemetryDegraded;
}
/**
 * GPU energy from the trapezoidal rule over adjacent power samples
 * ``E = sum(0.5*(P_i+P_{i+1})*(t_{i+1}-t_i))`` (research ``METRICS.md``). Intervals crossing the
 * measured-window boundary are linearly clipped. All power in watts, time from the monotonic clock;
 * joules are null when no power sample exists (never zero).
 */
export interface EnergyIntegration {
  boundary_clipped?: BoundaryClipped;
  coverage_fraction?: CoverageFraction;
  energy_per_1000_nonpadding_tokens?: EnergyPer1000NonpaddingTokens;
  joules_per_measured_optimizer_step?: JoulesPerMeasuredOptimizerStep;
  max_power_watts?: MaxPowerWatts;
  measured_window_joules?: MeasuredWindowJoules;
  median_measured_power_watts?: MedianMeasuredPowerWatts;
  method?: Method;
  power_sample_count?: PowerSampleCount;
  run_joules?: RunJoules;
  time_weighted_mean_power_watts?: TimeWeightedMeanPowerWatts;
}
/**
 * Derived GPU aggregates over the MEASURED window (plus explicit whole-run temperature bounds).
 */
export interface GpuTelemetrySummary {
  device_index?: DeviceIndex;
  device_uuid?: DeviceUuid;
  graphics_clock_mhz?: MetricSummary | null;
  max_run_temperature_c?: MaxRunTemperatureC;
  memory?: MemoryWindowSummary | null;
  memory_clock_mhz?: MetricSummary | null;
  memory_controller_utilization_percent?: MetricSummary | null;
  observed_performance_states?: ObservedPerformanceStates;
  observed_throttle_reasons?: ObservedThrottleReasons;
  power_watts?: MetricSummary | null;
  starting_temperature_c?: StartingTemperatureC;
  temperature_c?: MetricSummary | null;
  utilization_percent?: MetricSummary | null;
}
/**
 * Descriptive statistics for one metric over a value set (measured steps/samples for a single
 * run, or trial values for a cross-trial set). ``sample_standard_deviation`` uses the ``n-1``
 * denominator and is null for ``count < 2``. Inputs are always the unrounded raw values.
 */
export interface MetricSummary {
  count: Count;
  maximum?: Maximum;
  mean?: Mean;
  median?: Median;
  minimum?: Minimum;
  sample_standard_deviation?: SampleStandardDeviation;
  unit: Unit;
}
/**
 * Baseline, measured-window maximum, and whole-run maximum of the memory signature, kept
 * separate so a model-load peak is never confused with steady-state residency (``METRICS.md``).
 */
export interface MemoryWindowSummary {
  baseline?: MemoryMetrics | null;
  measured_window_max?: MemoryMetrics | null;
  whole_run_max?: MemoryMetrics | null;
}
/**
 * The full memory-signature block sampled during a run. Distinguishes PyTorch's allocator view,
 * raw CUDA device memory, and OS-level residency (``dedicated`` vs ``shared`` GPU memory) so a
 * Windows/WDDM spill to shared memory is VISIBLE rather than hidden inside 'used VRAM'. Grounded in
 * gpu_probe.GpuMemory + the estimators note that torch.max_memory_allocated counts the WDDM spill.
 */
export interface MemoryMetrics {
  cuda_device_free_bytes?: CudaDeviceFreeBytes;
  cuda_device_used_bytes?: CudaDeviceUsedBytes;
  dedicated_gpu_bytes?: DedicatedGpuBytes;
  process_rss_bytes?: ProcessRssBytes;
  shared_gpu_bytes?: SharedGpuBytes;
  system_ram_used_bytes?: SystemRamUsedBytes;
  torch_allocated_bytes?: TorchAllocatedBytes;
  torch_peak_allocated_bytes?: TorchPeakAllocatedBytes;
  torch_peak_reserved_bytes?: TorchPeakReservedBytes;
  torch_reserved_bytes?: TorchReservedBytes;
}
/**
 * Derived host aggregates over the MEASURED window. Disk deltas are end-minus-start of the
 * cumulative counters across the run interval.
 */
export interface HostTelemetrySummary {
  cpu_utilization_percent?: MetricSummary | null;
  disk_read_delta_bytes?: DiskReadDeltaBytes;
  disk_write_delta_bytes?: DiskWriteDeltaBytes;
  process_tree_rss?: MemoryWindowSummary | null;
  process_tree_rss_bytes?: MetricSummary | null;
  swap_used_bytes?: MetricSummary | null;
  swap_used_delta_bytes?: SwapUsedDeltaBytes;
  system_ram_used_bytes?: MetricSummary | null;
}
/**
 * The complete lineage a metric record must link to be paper-valid (research ``METRICS.md``).
 * All fields are optional strings; the harness fills what the plan/manifest expose, and
 * :class:`ScientificCompleteness` decides whether the required subset is present.
 */
export interface TelemetryIdentity {
  amendment_id?: AmendmentId;
  capability_report_hash?: CapabilityReportHash;
  cell_id?: CellId;
  chat_template_sha256?: ChatTemplateSha256;
  dataset_fingerprint?: DatasetFingerprint;
  effective_matrix_sha256?: EffectiveMatrixSha256;
  environment_lock_hash?: EnvironmentLockHash;
  execution_configuration_hash?: ExecutionConfigurationHash;
  execution_probe_hash?: ExecutionProbeHash;
  model_ref?: ModelRef;
  plan_hash?: PlanHash;
  plan_id?: PlanId;
  protocol_version?: ProtocolVersion;
  repository_commit?: RepositoryCommit;
  run_id?: RunId;
  sequence_view?: SequenceView;
  study_id?: StudyId;
  tokenizer_ref?: TokenizerRef;
  trial_id?: TrialId;
  worker_wheel_sha256?: WorkerWheelSha256;
}
/**
 * The terminal outcome, copied verbatim from the authoritative RunManifest - never re-derived so
 * a summary can never disagree with the run's own terminal truth.
 */
export interface RunOutcomeSummary {
  failure_stage?: StageMarker | null;
  failure_taxonomy?: FailureTaxonomy | null;
  measured_fit?: FitClassification | null;
  state: State;
  training_success?: TrainingSuccess;
}
/**
 * The planner/calibrator verdict on whether a resolved RunPlan fits the target environment, and
 * HOW: a native fit, a deliberately-offloaded fit, or an ACCIDENTAL spill (the silent WDDM/unified
 * paging that looks frozen but crawls at 10-25x). NEW — the engine emits only a coarse warn/pass
 * VRAM band (preflight.gpu_memory, _VRAM_SAFETY_MARGIN_GB).
 */
export interface FitClassification {
  attention_path?: AttentionImpl | null;
  classification: FitClass;
  contract_version?: ContractVersion1;
  device_capacity_bytes?: DeviceCapacityBytes;
  estimated_peak_bytes?: EstimatedPeakBytes;
  headroom_bytes?: HeadroomBytes;
  rationale?: Rationale;
}
/**
 * The sampler's own cost, so telemetry overhead is quantified rather than assumed negligible.
 */
export interface MeasurementOverhead {
  method?: Method1;
  overhead_fraction_of_wall?: OverheadFractionOfWall;
  per_sample_mean_seconds?: PerSampleMeanSeconds;
  total_sampler_seconds?: TotalSamplerSeconds;
  wall_seconds?: WallSeconds;
}
/**
 * The exact raw file a summary was derived from - path, sha256, and line count - so a summary is
 * provably a function of its raw source and cannot silently drift from it.
 */
export interface RawRecordBinding {
  line_count: LineCount;
  path: Path;
  record_kind: RecordKind;
  sha256: Sha256;
}
/**
 * The observed inter-sample cadence, so a claimed 200 ms rate is checked against reality and no
 * precision beyond the true cadence is invented.
 */
export interface SamplingCadence {
  observed_max_interval_ms?: ObservedMaxIntervalMs;
  observed_median_interval_ms?: ObservedMedianIntervalMs;
  observed_min_interval_ms?: ObservedMinIntervalMs;
  requested_interval_ms?: RequestedIntervalMs;
  sample_count?: SampleCount;
  source?: OperatingSystem | null;
}
/**
 * Derived per-step training aggregates. Warm-up (steps 1-2) and measured (steps 3-12) are kept
 * partitioned; loss is recorded for EVERY completed step, warm-up included.
 */
export interface StepTelemetrySummary {
  changed_adapter_tensor_count?: ChangedAdapterTensorCount;
  completed_optimizer_steps?: CompletedOptimizerSteps;
  first_loss?: FirstLoss;
  gradient_observed_tensor_count?: GradientObservedTensorCount;
  last_loss?: LastLoss;
  measured_optimizer_steps?: MeasuredOptimizerSteps;
  min_loss?: MinLoss;
  nonpadding_tokens_per_second?: MetricSummary | null;
  optimizer_steps_per_minute?: OptimizerStepsPerMinute;
  samples_per_second?: MetricSummary | null;
  step_losses?: StepLosses;
  step_time_seconds?: MetricSummary | null;
  supervised_tokens_per_second?: MetricSummary | null;
  trainable_state_after_sha256?: TrainableStateAfterSha256;
  trainable_state_before_sha256?: TrainableStateBeforeSha256;
  warmup_optimizer_steps?: WarmupOptimizerSteps;
}
/**
 * One finite loss bound to exactly one completed optimizer step.
 */
export interface OptimizerStepLossEvidence {
  loss: Loss;
  optimizer_step: OptimizerStep;
}
