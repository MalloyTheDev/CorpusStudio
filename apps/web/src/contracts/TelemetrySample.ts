/* GENERATED from docs/contracts/TelemetrySample.schema.json — do not edit. Run: npm run gen:contracts */

export type ContractVersion = "1.0.0";
export type DeviceIndex = number | null;
export type DeviceUuid = string | null;
export type GraphicsClockMhz = number | null;
export type MemoryClockMhz = number | null;
export type MemoryControllerUtilizationPercent = number | null;
export type PerformanceState = string | null;
export type PowerWatts = number | null;
export type TemperatureC = number | null;
export type ThrottleReasons = string[];
export type UtilizationPercent = number | null;
export type CpuUtilizationPercent = number | null;
export type DiskReadBytes = number | null;
export type DiskWriteBytes = number | null;
export type ProcessTreeRssBytes = number | null;
export type SwapUsedBytes = number | null;
export type SystemRamAvailableBytes = number | null;
export type SystemRamUsedBytes = number | null;
export type CudaDeviceFreeBytes = number | null;
export type CudaDeviceUsedBytes = number | null;
export type DedicatedGpuBytes = number | null;
export type ProcessRssBytes = number | null;
export type SharedGpuBytes = number | null;
export type SystemRamUsedBytes1 = number | null;
export type TorchAllocatedBytes = number | null;
export type TorchPeakAllocatedBytes = number | null;
export type TorchPeakReservedBytes = number | null;
export type TorchReservedBytes = number | null;
export type MonotonicNs = number;
export type OptimizerStep = number | null;
export type Phase = "baseline" | "setup" | "warmup" | "measured" | "teardown";
export type ProbeUnavailable = string[];
export type RunId = string;
export type SampleSeq = number;
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
export type WallUtc = string;

/**
 * One raw environmental sample (the harness's per-tick record). Written append-only, one JSON
 * object per line, to ``<run-dir>/TelemetrySamples.jsonl`` as it is taken - the authoritative raw
 * series the summary is derived from. ``monotonic_ns`` drives interval math; ``wall_utc`` is
 * lineage only. ``sample_source`` keeps native-Linux, WSL, and Windows samples DISTINCT and is
 * never collapsed into one category.
 */
export interface TelemetrySample {
  contract_version?: ContractVersion;
  gpu?: GpuTelemetrySample | null;
  host?: HostTelemetrySample | null;
  memory?: MemoryMetrics | null;
  monotonic_ns: MonotonicNs;
  optimizer_step?: OptimizerStep;
  phase: Phase;
  probe_unavailable?: ProbeUnavailable;
  run_id: RunId;
  sample_seq: SampleSeq;
  sample_source: OperatingSystem;
  wall_utc: WallUtc;
}
/**
 * One raw GPU environmental reading bound to an exact device index + UUID. Every field is
 * optional: a driver that does not expose a field leaves it ``null`` and names the probe on the
 * parent sample's ``probe_unavailable`` list. Values are never zero-filled. GPU/host *memory* lives
 * in the sample's shared :class:`MemoryMetrics`, not here.
 */
export interface GpuTelemetrySample {
  device_index?: DeviceIndex;
  device_uuid?: DeviceUuid;
  graphics_clock_mhz?: GraphicsClockMhz;
  memory_clock_mhz?: MemoryClockMhz;
  memory_controller_utilization_percent?: MemoryControllerUtilizationPercent;
  performance_state?: PerformanceState;
  power_watts?: PowerWatts;
  temperature_c?: TemperatureC;
  throttle_reasons?: ThrottleReasons;
  utilization_percent?: UtilizationPercent;
}
/**
 * One raw host environmental reading. ``process_tree_rss_bytes`` is the worker *process tree*
 * RSS (distinct from the single-process ``MemoryMetrics.process_rss_bytes``). Disk counters are
 * cumulative byte totals; the aggregator differences them across the window.
 */
export interface HostTelemetrySample {
  cpu_utilization_percent?: CpuUtilizationPercent;
  disk_read_bytes?: DiskReadBytes;
  disk_write_bytes?: DiskWriteBytes;
  process_tree_rss_bytes?: ProcessTreeRssBytes;
  swap_used_bytes?: SwapUsedBytes;
  system_ram_available_bytes?: SystemRamAvailableBytes;
  system_ram_used_bytes?: SystemRamUsedBytes;
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
  system_ram_used_bytes?: SystemRamUsedBytes1;
  torch_allocated_bytes?: TorchAllocatedBytes;
  torch_peak_allocated_bytes?: TorchPeakAllocatedBytes;
  torch_peak_reserved_bytes?: TorchPeakReservedBytes;
  torch_reserved_bytes?: TorchReservedBytes;
}
