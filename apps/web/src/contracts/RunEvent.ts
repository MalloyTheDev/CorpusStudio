/* GENERATED from docs/contracts/RunEvent.schema.json — do not edit. Run: npm run gen:contracts */

export type ContractVersion = "1.0.0";
export type EmittedAt = string;
export type Epoch = number | null;
export type EventType =
  | "stage"
  | "metric"
  | "log"
  | "warning"
  | "checkpoint_written"
  | "eval_result"
  | "artifact_produced"
  | "heartbeat"
  | "terminal";
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
export type Message = string | null;
export type GpuUtilization = number | null;
export type GradNorm = number | null;
export type LearningRate = number | null;
export type Loss = number | null;
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
export type MemoryControllerUtilization = number | null;
export type Assumptions = string[];
export type ParameterObservationCoverage = "complete" | "sampled" | "partial";
export type Definition = string;
export type Evidence = "measured" | "estimated" | "declared";
export type CountHandling =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling1 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling2 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling3 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling4 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling5 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type CountHandling6 =
  "included" | "excluded" | "deduplicated" | "represented_separately" | "not_applicable" | "unknown";
export type ParameterIdentityBasis =
  | "independent_coordinates"
  | "stored_tensor_elements"
  | "optimizer_addressable_coordinates"
  | "runtime_identity_set"
  | "topology_formula"
  | "declared_definition"
  | "unknown";
/**
 * Distinct parameter quantities required for dense-safe and MoE-safe accounting.
 */
export type ParameterCountKind =
  | "logical"
  | "active_token"
  | "active_sequence"
  | "touched_window"
  | "resident"
  | "updated_window"
  | "exposed_window"
  | "effective";
export type Notes = string;
export type ObservationId = string;
export type ComponentIds = string[];
export type CoordinateUniverseId = string;
export type CoordinateUniverseSha256 = string | null;
export type Definition1 = string;
export type DeviceId = string | null;
export type ExpertIds = string[];
export type ParameterScopeKind =
  | "model"
  | "component_set"
  | "shared"
  | "router"
  | "expert_group"
  | "expert_set"
  | "adapter"
  | "embedding"
  | "output_head"
  | "device_residency"
  | "custom";
/**
 * A physical state tier. A RunPlan names the intended tier; only runtime evidence may claim
 * actual residency there.
 */
export type MemoryTier = "gpu" | "pinned_ram" | "pageable_ram" | "nvme" | "sata" | "remote" | "unknown";
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ScopeId = string;
export type CapturedAt = string | null;
export type ParameterEvidenceSourceKind =
  | "model_config"
  | "model_descriptor"
  | "safetensors_header"
  | "planner"
  | "backend_worker"
  | "checkpoint_inventory"
  | "evaluation_runtime"
  | "user_supplied";
export type Method = string;
export type Producer = string;
export type ProducerVersion = string;
export type Unit = "coordinates" | "elements" | "parameters";
export type Value1 = number;
export type ParameterValueRelation = "exact" | "estimate" | "lower_bound" | "upper_bound";
export type CapturedAt1 = string | null;
export type Definition2 = string;
export type EventSeqEnd = number | null;
export type EventSeqStart = number | null;
export type ParameterWindowKind =
  "static_snapshot" | "token" | "sequence" | "instant" | "microbatch" | "optimizer_window" | "run";
export type MicrostepEnd = number | null;
export type MicrostepStart = number | null;
export type OptimizerStepEnd = number | null;
export type OptimizerStepStart = number | null;
export type SequenceId = string | null;
export type TokenIndex = number | null;
export type WindowId = string;
export type ParameterObservations = ParameterObservation[];
export type PcieRxBytesPerSec = number | null;
export type PcieTxBytesPerSec = number | null;
export type PowerWatts = number | null;
export type StepTimeSeconds = number | null;
export type SupervisedTokensPerSec = number | null;
export type TemperatureC = number | null;
export type TokensPerSec = number | null;
export type Microstep = number | null;
export type OptimizerStep = number | null;
export type Payload = {
  [k: string]: unknown;
} | null;
export type RunId = string;
export type Seq = number;
/**
 * Ordered lifecycle stage of a run, launch → export. A RunEvent carries the stage it belongs to
 * so a consumer can render a precise progress spine and localize a failure to the exact stage.
 */
export type StageMarker =
  | "process_start"
  | "env_loaded"
  | "cuda_init"
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
 * One envelope in the structured telemetry stream a worker emits for a run — the RunEvent half
 * of the immutable-RunPlan-in / RunEvent-stream-out worker protocol. NEW; the engine has no
 * streaming telemetry today (run_registry is a durable per-run record, not an event stream).
 */
export interface RunEvent {
  contract_version?: ContractVersion;
  emitted_at: EmittedAt;
  epoch?: Epoch;
  event_type: EventType;
  fit?: FitClassification | null;
  message?: Message;
  metrics?: EventMetrics | null;
  microstep?: Microstep;
  optimizer_step?: OptimizerStep;
  payload?: Payload;
  run_id: RunId;
  seq: Seq;
  stage?: StageMarker | null;
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
 * Present on metric/heartbeat events. All optional — a worker emits what it can sample. The
 * memory block + step_time make the WDDM spill (10-25x slowdown, non-zero shared bytes) visible.
 */
export interface EventMetrics {
  gpu_utilization?: GpuUtilization;
  grad_norm?: GradNorm;
  learning_rate?: LearningRate;
  loss?: Loss;
  memory?: MemoryMetrics | null;
  memory_controller_utilization?: MemoryControllerUtilization;
  parameter_observations?: ParameterObservations;
  pcie_rx_bytes_per_sec?: PcieRxBytesPerSec;
  pcie_tx_bytes_per_sec?: PcieTxBytesPerSec;
  power_watts?: PowerWatts;
  step_time_seconds?: StepTimeSeconds;
  supervised_tokens_per_sec?: SupervisedTokensPerSec;
  temperature_c?: TemperatureC;
  tokens_per_sec?: TokensPerSec;
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
 * One evidence-bearing parameter count. Unknown evidence is represented as a gap, never zero.
 */
export interface ParameterObservation {
  assumptions?: Assumptions;
  coverage: ParameterObservationCoverage;
  definition: Definition;
  evidence: Evidence;
  handling: ParameterCountHandling;
  identity_basis: ParameterIdentityBasis;
  kind: ParameterCountKind;
  notes?: Notes;
  observation_id: ObservationId;
  scope: ParameterScope;
  source: ParameterEvidenceSource;
  unit?: Unit;
  value: Value1;
  value_relation: ParameterValueRelation;
  window: ParameterWindow;
}
export interface ParameterCountHandling {
  decompressed_caches?: CountHandling;
  generated?: CountHandling1;
  optimizer_shadows?: CountHandling2;
  quantized?: CountHandling3;
  replicated?: CountHandling4;
  shared?: CountHandling5;
  tied?: CountHandling6;
}
/**
 * Stable coordinate universe for an authoritative parameter observation.
 *
 * Runtime addresses are never identities. Sparse scopes carry stable expert IDs, and every scope
 * is tied to one exact model reference plus a named coordinate universe.
 */
export interface ParameterScope {
  component_ids?: ComponentIds;
  coordinate_universe_id: CoordinateUniverseId;
  coordinate_universe_sha256?: CoordinateUniverseSha256;
  definition: Definition1;
  device_id?: DeviceId;
  expert_ids?: ExpertIds;
  kind: ParameterScopeKind;
  memory_tier?: MemoryTier | null;
  model_ref: Ref;
  scope_id: ScopeId;
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
export interface ParameterEvidenceSource {
  backend_ref?: Ref | null;
  captured_at?: CapturedAt;
  environment_ref?: Ref | null;
  kind: ParameterEvidenceSourceKind;
  method: Method;
  producer: Producer;
  producer_version: ProducerVersion;
  source_ref: Ref;
}
/**
 * The exact computation or scheduling window a count describes.
 */
export interface ParameterWindow {
  captured_at?: CapturedAt1;
  definition: Definition2;
  event_seq_end?: EventSeqEnd;
  event_seq_start?: EventSeqStart;
  kind: ParameterWindowKind;
  microstep_end?: MicrostepEnd;
  microstep_start?: MicrostepStart;
  optimizer_step_end?: OptimizerStepEnd;
  optimizer_step_start?: OptimizerStepStart;
  plan_ref?: Ref | null;
  run_ref?: Ref | null;
  sequence_id?: SequenceId;
  token_index?: TokenIndex;
  window_id: WindowId;
}
