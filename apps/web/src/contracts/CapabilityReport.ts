/* GENERATED from docs/contracts/CapabilityReport.schema.json — do not edit. Run: npm run gen:contracts */

export type BackendId = string;
export type BackendVersion = string | null;
export type BitsandbytesOk = boolean;
export type ContractVersion = "1.0.0";
export type AdapterMethod =
  "none" | "lora" | "qlora" | "dora" | "ia3" | "full_finetune" | "prompt_tuning" | "prefix_tuning";
export type AdapterMethods = AdapterMethod[];
/**
 * ``math``/``eager`` is forced on Blackwell sm_120 — the fused flash/mem-efficient kernels
 * deadlock on the first backward (training/environment.py, estimators.py) — at a large activation
 * VRAM cost.
 */
export type AttentionImpl =
  "math" | "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "mem_efficient" | "xformers";
export type AttentionImpls = AttentionImpl[];
export type CommunicationBackend = "none" | "nccl" | "gloo" | "mpi" | "ucc" | "backend_native";
export type CommunicationBackends = CommunicationBackend[];
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
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type PrecisionModes = PrecisionMode[];
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type QuantizationModes = QuantizationMode[];
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type GeneratedAt = string | null;
export type Artifact = string | null;
export type Dependencies = string[];
export type DirectUrl = string | null;
export type Installer = string | null;
export type Name = string;
export type Requested = boolean | null;
export type Source = "pypi" | "wheel" | "sdist" | "conda" | "vcs" | "local" | "unknown";
export type Version = string | null;
export type InstalledPackages = PackageLock[];
export type MissingPackages = string[];
export type Notes = string[];
export type Detail = string | null;
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
export type Probe = string;
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
  communication_backends?: CommunicationBackends;
  objective_capabilities?: ObjectiveCapabilities;
  offload_strategies?: OffloadStrategies;
  parallelism_kinds?: ParallelismKinds;
  placement_modes?: PlacementModes;
  placement_tiers?: PlacementTiers;
  precision_modes?: PrecisionModes;
  quantization_modes?: QuantizationModes;
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
  dependencies?: Dependencies;
  direct_url?: DirectUrl;
  hash?: HashRef | null;
  installer?: Installer;
  name: Name;
  requested?: Requested;
  source?: Source;
  version?: Version;
}
export interface ProbeResult {
  detail?: Detail;
  measured?: Measured;
  outcome: FailureTaxonomy;
  probe: Probe;
}
export interface Measured {
  [k: string]: unknown;
}
