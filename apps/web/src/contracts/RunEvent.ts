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
 * ``math``/``eager`` is forced on Blackwell sm_120 — the fused flash/mem-efficient kernels
 * deadlock on the first backward (training/environment.py, estimators.py) — at a large activation
 * VRAM cost.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
/**
 * The fit verdict. ``NATIVE_*`` = fully resident. ``CONTROLLED_*`` = a deliberate, planned
 * offload (acceptable, slower). ``ACCIDENTAL_*`` / ``THRASHING`` = an unplanned spill the platform
 * did silently (the failure mode the engine warns about). ``FAIL`` = will not run.
 */
export type FitClass =
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
