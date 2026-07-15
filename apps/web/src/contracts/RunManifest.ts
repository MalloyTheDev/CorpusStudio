/* GENERATED from docs/contracts/RunManifest.schema.json — do not edit. Run: npm run gen:contracts */

export type ArtifactIds = string[];
export type BaseModel = string;
export type Checkpoints = string[];
export type ContractVersion = "1.0.0";
export type CreatedAt = string;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type AfterEvalModel = string | null;
export type AfterEvalRef = string | null;
export type BeforeEvalRef = string | null;
export type ContractVersion1 = "1.0.0";
export type Detail = string | null;
export type DetectedAt = string | null;
export type ExceptionType = string | null;
export type ExitCode = number | null;
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
export type ContractVersion2 = "1.0.0";
export type DeviceCapacityBytes = number | null;
export type EstimatedPeakBytes = number | null;
export type HeadroomBytes = number | null;
export type Rationale = string;
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
export type Message = string;
export type Reconciled = boolean;
export type Remediation = string | null;
export type RunId = string | null;
export type Signal = string | null;
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
export type FinishedAt = string | null;
export type Notes = string;
export type OutputDir = string;
export type ParameterAccountingRefs = Ref[];
export type Argv = string[];
export type ExitCode1 = number | null;
export type Pid = number | null;
export type ProcessStartedAt = string | null;
export type ConfigSha256 = string | null;
export type DatasetFingerprint = string | null;
export type DatasetRowCount = number;
export type EngineVersion = string;
export type Platform = string;
export type PythonVersion = string;
export type RunId1 = string;
export type StartedAt = string | null;
export type State = "prepared" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";
export type Target = string;
export type AdapterBytesVerified = true;
export type AdapterConfigSha256 = string;
export type AdapterSafetensorsSha256 = string;
export type ArtifactIntegrityVerified = true;
export type AdapterConfigSemanticSha256 = string;
export type AfterSha256 = string;
export type BeforeSha256 = string;
export type ChangedTensorCount = number;
/**
 * @minItems 1
 */
export type ChangedTensorNames = [string, ...string[]];
export type HashAlgorithm = "sha256-safetensors-tensor-state-v1";
export type TensorCount = number;
/**
 * @minItems 1
 */
export type TensorNames = [string, ...string[]];
export type CompletedOptimizerSteps = number;
export type EligibleTensorCount = number;
/**
 * @minItems 1
 */
export type EligibleTensorNames = [string, ...string[]];
export type ObservedTensorCount = number;
/**
 * @minItems 1
 */
export type ObservedTensorNames = [string, ...string[]];
export type OptimizerCreated = true;
/**
 * @minItems 1
 */
export type StepLosses = [OptimizerStepLossEvidence, ...OptimizerStepLossEvidence[]];
export type Loss = number;
export type OptimizerStep = number;
export type AfterSha2561 = string;
export type BeforeSha2561 = string;
export type ChangedTensorCount1 = number;
/**
 * @minItems 1
 */
export type ChangedTensorNames1 = [string, ...string[]];
export type HashAlgorithm1 = "sha256-trainable-state-v1";
export type TrainableTensorCount = number;
/**
 * @minItems 1
 */
export type TrainableTensorNames = [string, ...string[]];
export type OutputPathVerified = true;
export type UpdatedAt = string;

/**
 * A single run INSTANCE: the crash-safe durable record of one execution of a RunPlan.
 * Formalizes run_registry.TrainingRunRecord almost field-for-field + its state machine (terminal =
 * {succeeded, failed, cancelled, interrupted}; a dead-pid 'running' record reconciles to
 * interrupted).
 */
export interface RunManifest {
  artifact_ids?: ArtifactIds;
  base_model?: BaseModel;
  checkpoints?: Checkpoints;
  contract_version?: ContractVersion;
  created_at: CreatedAt;
  dataset_ref?: Ref | null;
  environment_ref?: Ref | null;
  evaluation?: RunEvaluationLink | null;
  failure?: FailureRecord | null;
  final_fit?: FitClassification | null;
  finished_at?: FinishedAt;
  notes?: Notes;
  output_dir?: OutputDir;
  parameter_accounting_refs?: ParameterAccountingRefs;
  plan_ref: Ref;
  process?: RunProcessInfo | null;
  reproducibility?: RunReproducibility | null;
  run_id: RunId1;
  started_at?: StartedAt;
  state?: State;
  target?: Target;
  training_success_evidence?: TrainingSuccessEvidence | null;
  updated_at: UpdatedAt;
}
/**
 * A stable reference to another contract instance by id, optionally pinned to a content hash so
 * the reference cannot silently re-point.
 */
export interface Ref {
  hash?: HashRef | null;
  id: Id;
}
/**
 * An algorithm-tagged digest. The engine emits sha256 today; the algo tag makes a future
 * migration additive (cf. versions/version_registry.FINGERPRINT_ALGO).
 */
export interface HashRef {
  algo?: Algo;
  value?: Value;
}
export interface RunEvaluationLink {
  after_eval_model?: AfterEvalModel;
  after_eval_ref?: AfterEvalRef;
  before_eval_ref?: BeforeEvalRef;
}
/**
 * A structured, classified terminal outcome for a run, capability probe, or export. The
 * taxonomy turns 'it died' into an actionable category — a real OOM vs a KERNEL_STALL (the sm_120
 * fused-attention deadlock) vs an ACCIDENTAL_SPILL vs a CONTROLLED_OFFLOAD. NEW.
 */
export interface FailureRecord {
  contract_version?: ContractVersion1;
  detail?: Detail;
  detected_at?: DetectedAt;
  exception_type?: ExceptionType;
  exit_code?: ExitCode;
  fit_at_failure?: FitClassification | null;
  memory_at_failure?: MemoryMetrics | null;
  message: Message;
  reconciled?: Reconciled;
  remediation?: Remediation;
  run_id?: RunId;
  signal?: Signal;
  stage?: StageMarker | null;
  taxonomy: FailureTaxonomy;
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
  contract_version?: ContractVersion2;
  device_capacity_bytes?: DeviceCapacityBytes;
  estimated_peak_bytes?: EstimatedPeakBytes;
  headroom_bytes?: HeadroomBytes;
  rationale?: Rationale;
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
 * Process identity so a recycled pid is never mistaken for a live run. A 'running' record whose
 * pid is not alive reconciles to 'interrupted' (run_registry.reconcile_running_records).
 */
export interface RunProcessInfo {
  argv?: Argv;
  exit_code?: ExitCode1;
  pid?: Pid;
  process_started_at?: ProcessStartedAt;
}
/**
 * Embedded reproducibility manifest (provenance.RunProvenance) for a self-contained audit.
 */
export interface RunReproducibility {
  config_sha256?: ConfigSha256;
  dataset_fingerprint?: DatasetFingerprint;
  dataset_row_count?: DatasetRowCount;
  engine_version?: EngineVersion;
  platform?: Platform;
  python_version?: PythonVersion;
}
/**
 * All gates required before a resolved run or measured fit may be called successful.
 */
export interface TrainingSuccessEvidence {
  adapter_bytes_verified: AdapterBytesVerified;
  adapter_config_sha256: AdapterConfigSha256;
  adapter_safetensors_sha256: AdapterSafetensorsSha256;
  artifact_integrity_verified: ArtifactIntegrityVerified;
  execution: TrainingExecutionEvidence;
  measured_peak?: MemoryMetrics | null;
  output_path_verified: OutputPathVerified;
}
/**
 * Trainer-side proof produced before adapter export is admitted as a success.
 */
export interface TrainingExecutionEvidence {
  adapter_export_state: AdapterExportStateEvidence;
  completed_optimizer_steps: CompletedOptimizerSteps;
  gradient_coverage: GradientCoverageEvidence;
  optimizer_created: OptimizerCreated;
  step_losses: StepLosses;
  trainable_state: TrainableStateChangeEvidence;
}
/**
 * Canonical identity for the exact PEFT state expected in adapter_model.safetensors.
 */
export interface AdapterExportStateEvidence {
  adapter_config_semantic_sha256: AdapterConfigSemanticSha256;
  after_sha256: AfterSha256;
  before_sha256: BeforeSha256;
  changed_tensor_count: ChangedTensorCount;
  changed_tensor_names: ChangedTensorNames;
  hash_algorithm?: HashAlgorithm;
  tensor_count: TensorCount;
  tensor_names: TensorNames;
}
/**
 * Observed materialized adapter gradients without claiming unused tensors had gradients.
 */
export interface GradientCoverageEvidence {
  eligible_tensor_count: EligibleTensorCount;
  eligible_tensor_names: EligibleTensorNames;
  observed_tensor_count: ObservedTensorCount;
  observed_tensor_names: ObservedTensorNames;
}
/**
 * One finite loss bound to exactly one completed optimizer step.
 */
export interface OptimizerStepLossEvidence {
  loss: Loss;
  optimizer_step: OptimizerStep;
}
/**
 * Canonical before/after identity for the complete trainable adapter state.
 */
export interface TrainableStateChangeEvidence {
  after_sha256: AfterSha2561;
  before_sha256: BeforeSha2561;
  changed_tensor_count: ChangedTensorCount1;
  changed_tensor_names: ChangedTensorNames1;
  hash_algorithm?: HashAlgorithm1;
  trainable_tensor_count: TrainableTensorCount;
  trainable_tensor_names: TrainableTensorNames;
}
