/* GENERATED from docs/contracts/CapabilityReport.schema.json — do not edit. Run: npm run gen:contracts */

export type BackendId = string;
export type BackendVersion = string | null;
export type BitsandbytesOk = boolean;
export type ContractVersion = "1.0.0";
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type AdapterMethods = AdapterMethod[];
/**
 * ``math``/``eager`` is forced on native-Windows/WDDM Blackwell sm_120 because the fused flash
 * kernel deadlocks there. Other platforms require their own functional capability result; WSL
 * evidence is not bare-Linux proof.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
export type AttentionImpls = AttentionImpl[];
/**
 * The exact attention implementation an execution policy permits at runtime.
 */
export type AttentionKernel =
  | "eager"
  | "torch_sdpa_math"
  | "torch_sdpa_flash"
  | "torch_sdpa_mem_efficient"
  | "flash_attention_2"
  | "flash_attention_3"
  | "xformers";
export type AttentionKernels = AttentionKernel[];
export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type CheckpointImpls = CheckpointImpl[];
export type CommunicationBackend = "none" | "nccl" | "gloo" | "mpi" | "ucc" | "backend_native";
export type CommunicationBackends = CommunicationBackend[];
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
export type ExecutionContractVersion = string;
export type ExportFormat =
  "adapter_peft" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
export type LossImpl = "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt";
export type Optimizer =
  | "adamw_torch"
  | "adamw_torch_fused"
  | "adamw_8bit"
  | "adamw_bnb_8bit"
  | "paged_adamw_8bit"
  | "paged_adamw_32bit"
  | "adafactor"
  | "lion"
  | "sgd";
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type Probe = string;
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type RuntimeMode = "training" | "cpu_toy";
export type ExecutionCombinations = ExecutionCapabilityCombination[];
export type ExecutionContractVersions = string[];
export type LossImpls = LossImpl[];
export type ObjectiveCapabilities = string[];
/**
 * The ``controlled_*`` values are the deliberate, planned counterparts of the accidental spills
 * in :class:`FitClass`.
 */
export type OffloadStrategy =
  | "none"
  | "controlled_activation_offload"
  | "controlled_optimizer_offload"
  | "controlled_parameter_offload"
  | "cpu_offload"
  | "disk_offload"
  | "deepspeed_zero2"
  | "deepspeed_zero3";
export type OffloadStrategies = OffloadStrategy[];
export type Optimizers = Optimizer[];
export type ParallelismKind = "data" | "tensor" | "pipeline" | "expert" | "sequence" | "context";
export type ParallelismKinds = ParallelismKind[];
export type PlacementMode =
  "single_resource" | "identity_scoped" | "replicated" | "sharded" | "tiered" | "expert_scoped";
export type PlacementModes = PlacementMode[];
/**
 * A physical state tier. A RunPlan names the intended tier; only runtime evidence may claim
 * actual residency there.
 */
export type MemoryTier = "gpu" | "pinned_ram" | "pageable_ram" | "nvme" | "sata" | "remote" | "unknown";
export type PlacementTiers = MemoryTier[];
export type PrecisionModes = PrecisionMode[];
export type QuantizationModes = QuantizationMode[];
export type TrainerFields = string[];
export type TrainerInitFields = string[];
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type GeneratedAt = string | null;
export type Artifact = string | null;
export type Dependencies = string[];
export type Direct = boolean | null;
export type DirectUrl = string | null;
export type Editable = boolean | null;
export type InstalledFileCount = number | null;
export type Installer = string | null;
export type Name = string;
export type NormalizedName = string;
export type RecordEntries = number | null;
export type RecordFailedEntries = string[];
export type RecordIntegrity = "verified" | "failed" | "missing" | "unknown";
export type RecordVerifiedEntries = number | null;
export type Requested = boolean | null;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type SourceEvidenceReason = string | null;
export type SourceIndexUrl = string | null;
export type VcsCommit = string | null;
export type VcsRepository = string | null;
export type Version = string | null;
export type InstalledPackages = PackageLock[];
export type MissingPackages = string[];
export type Notes = string[];
export type Detail = string | null;
export type ExecutionCombinations1 = ExecutionCapabilityCombination[];
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
  | "CHECKPOINT_FAILURE"
  | "ENVIRONMENT_FAILURE"
  | "UNSUPPORTED_CONFIGURATION"
  | "ACCIDENTAL_SPILL"
  | "CONTROLLED_OFFLOAD";
export type Probe1 = string;
export type ProbeResults = ProbeResult[];
export type Readiness = "ready" | "cpu_toy_only" | "not_ready";

/**
 * The DYNAMIC, measured counterpart of a BackendManifest: probe results against a specific
 * EnvironmentProfile. Formalizes environment.probe_training_runtime (ready/cpu_toy_ready/
 * bitsandbytes_ok/notes) and generalizes it to per-probe outcomes tagged with FailureTaxonomy.
 */
export interface CapabilityReport {
  backend_id: BackendId;
  backend_version?: BackendVersion;
  bitsandbytes_ok?: BitsandbytesOk;
  contract_version?: ContractVersion;
  effective_capabilities?: EffectiveCapabilities | null;
  environment_ref: Ref;
  generated_at?: GeneratedAt;
  installed_packages?: InstalledPackages;
  missing_packages?: MissingPackages;
  notes?: Notes;
  probe_results?: ProbeResults;
  readiness: Readiness;
}
/**
 * The intersection of what a backend DECLARES and what PROVED to work on this host. The planner
 * resolves a RunPlan against this, not the raw BackendManifest.
 */
export interface EffectiveCapabilities {
  adapter_methods?: AdapterMethods;
  attention_impls?: AttentionImpls;
  attention_kernels?: AttentionKernels;
  checkpoint_impls?: CheckpointImpls;
  communication_backends?: CommunicationBackends;
  execution_combinations?: ExecutionCombinations;
  execution_contract_versions?: ExecutionContractVersions;
  loss_impls?: LossImpls;
  objective_capabilities?: ObjectiveCapabilities;
  offload_strategies?: OffloadStrategies;
  optimizers?: Optimizers;
  parallelism_kinds?: ParallelismKinds;
  placement_modes?: PlacementModes;
  placement_tiers?: PlacementTiers;
  precision_modes?: PrecisionModes;
  quantization_modes?: QuantizationModes;
  trainer_fields?: TrainerFields;
  trainer_init_fields?: TrainerInitFields;
}
/**
 * One execution tuple demonstrated together by a bounded functional probe.
 *
 * Independent successes on precision, quantization, adapter, optimizer, loss, attention, and
 * checkpoint axes are diagnostic only. The planner may seal a run only from one of these complete
 * tuples, preventing a union of unrelated probes from becoming a fictional capability.
 */
export interface ExecutionCapabilityCombination {
  adapter_method: AdapterMethod;
  attention_impl: AttentionImpl;
  attention_kernel: AttentionKernel;
  checkpoint_impl: CheckpointImpl;
  device: DeviceKind;
  execution_contract_version: ExecutionContractVersion;
  export_format: ExportFormat;
  loss_impl: LossImpl;
  optimizer: Optimizer;
  precision: PrecisionMode;
  probe: Probe;
  quantization: QuantizationMode;
  runtime_mode: RuntimeMode;
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
/**
 * A resolved dependency and its install provenance.
 *
 * ``hash`` seals the installed distribution's RECORD metadata when that evidence is available; it
 * is not mislabelled as the original wheel hash. ``direct_url`` and ``artifact`` preserve the
 * stronger source identity pip exposes for direct/VCS/local installs. ``dependencies`` is the
 * installed metadata dependency graph, not a second resolver.
 */
export interface PackageLock {
  artifact?: Artifact;
  artifact_hash?: HashRef | null;
  dependencies?: Dependencies;
  direct?: Direct;
  direct_url?: DirectUrl;
  editable?: Editable;
  hash?: HashRef | null;
  installed_file_count?: InstalledFileCount;
  installed_files_hash?: HashRef | null;
  installer?: Installer;
  name: Name;
  normalized_name?: NormalizedName;
  record_entries?: RecordEntries;
  record_failed_entries?: RecordFailedEntries;
  record_integrity?: RecordIntegrity;
  record_verified_entries?: RecordVerifiedEntries;
  requested?: Requested;
  source?: Source;
  source_evidence_reason?: SourceEvidenceReason;
  source_index_url?: SourceIndexUrl;
  vcs_commit?: VcsCommit;
  vcs_repository?: VcsRepository;
  version?: Version;
}
export interface ProbeResult {
  detail?: Detail;
  execution_combinations?: ExecutionCombinations1;
  measured?: Measured;
  outcome: FailureTaxonomy;
  probe: Probe1;
  proves?: Proves;
}
export interface Measured {
  [k: string]: unknown;
}
export interface Proves {
  [k: string]: string[];
}
