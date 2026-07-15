/* GENERATED from docs/contracts/WorkerMessage.schema.json — do not edit. Run: npm run gen:contracts */

export type Body =
  | HelloBody
  | CapabilityProbeRequestBody
  | CapabilityReport
  | RunDispatchBody
  | RunAcceptedBody
  | FailureRecord
  | RunControlBody
  | RunEvent
  | HeartbeatBody
  | TerminalResultBody;
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
export type BackendId = string;
export type BackendVersion = string;
export type CapabilityProbes = string[];
export type CheckpointImpl = "full_state" | "adapter_only" | "sharded" | "distcp" | "safetensors";
export type CheckpointImpls = CheckpointImpl[];
export type Contains = (
  "adapter_weights" | "base_weights" | "optimizer_state" | "lr_scheduler" | "rng_state" | "trainer_state"
)[];
export type ReloadVerifiable = boolean;
export type Resumable = boolean;
export type CommunicationBackend = "none" | "nccl" | "gloo" | "mpi" | "ucc" | "backend_native";
export type CommunicationBackends = CommunicationBackend[];
export type CompileMode = "none" | "eager" | "reduce_overhead" | "max_autotune" | "aot_inductor";
export type CompileModes = CompileMode[];
export type ContractVersion = "1.0.0";
export type Condition = string;
/**
 * @minItems 1
 */
export type Packages = [string, ...string[]];
export type Severity = "block" | "warn";
export type DependencyConflicts = DependencyConflict[];
export type Name = string;
export type Optional = boolean;
export type Reason = string | null;
export type Specifier = string | null;
export type DependencyRequirements = DependencyRequirement[];
export type DisplayName = string;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ExecutionContractVersions = string[];
export type ExportFormat =
  "adapter_peft" | "merged_safetensors" | "merged_fp16" | "gguf" | "onnx" | "awq" | "gptq" | "mlx";
export type ServesIn = string[];
export type ExportCompatibility = ExportCompatibilityEntry[];
export type ExportFormats = ExportFormat[];
export type Condition1 = string;
export type Description = string;
export type Mitigation = string | null;
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
export type KnownFailureModes = KnownFailureMode[];
export type LossImpl = "cross_entropy" | "liger_fused_ce" | "chunked_ce" | "dpo" | "orpo" | "kto" | "ipo" | "reward_bt";
export type LossImpls = LossImpl[];
export type ModelFamilies = string[];
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
export type PrecisionMode = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type PrecisionModes = PrecisionMode[];
export type QuantizationMode = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type QuantizationModes = QuantizationMode[];
export type RequiredComputeCapability = string | null;
/**
 * @minItems 1
 */
export type SupportedDevices = [DeviceKind, ...DeviceKind[]];
export type DeviceKind = "cuda" | "rocm" | "mps" | "xpu" | "cpu";
/**
 * @minItems 1
 */
export type SupportedOs = [OperatingSystem, ...OperatingSystem[]];
export type OperatingSystem = "windows" | "wsl" | "linux" | "macos" | "unknown";
/**
 * @minItems 1
 */
export type TaskTypes = [TaskType, ...TaskType[]];
export type TaskType =
  | "sft"
  | "pretraining"
  | "preference"
  | "reward"
  | "classification"
  | "embedding"
  | "multimodal"
  | "evaluation"
  | "distillation"
  | "grpo";
export type Metrics = string[];
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
export type TelemetryHooks = TelemetryHook[];
export type TrainerFields = string[];
export type TrainerInitFields = string[];
/**
 * Verbatim from config_templates.TrainingConfigTarget.
 */
export type TrainerTarget =
  "corpus_studio" | "axolotl_yaml" | "trl_config" | "unsloth_script" | "huggingface_trainer" | "llama_factory";
export type CudaDriverVersion = string | null;
export type CudaRuntimeVersion = string | null;
export type DriverVersion = string | null;
export type MpsAvailable = boolean | null;
export type NvidiaSmiAvailable = boolean;
export type RocmVersion = string | null;
export type CapturedAt = string | null;
export type ContractVersion1 = "1.0.0";
export type InstructionSets = string[];
export type LogicalCores = number | null;
export type Model = string;
export type PhysicalCores = number | null;
export type EngineVersion = string;
export type EnvironmentSignature = string;
export type ComputeCapability = string | null;
export type ComputeCapabilityMajor = number | null;
export type Index = number;
export type Name1 = string;
export type Gen = number | null;
export type Width = number | null;
export type SupportedDtypes = PrecisionMode[];
export type VramFreeBytes = number | null;
export type VramTotalBytes = number | null;
export type Gpus = GpuDevice[];
export type HostnameHash = string | null;
/**
 * How the platform maps device memory. ``wddm`` (Windows) silently spills overflow to shared
 * system RAM and thrashes over PCIe; ``linux_dedicated`` models a hard-OOM boundary rather than a
 * WDDM spill; ``unified_memory`` is Apple MPS / integrated shared memory. This is a planning model,
 * not a measured-fit claim.
 */
export type MemoryResidencyModel = "wddm" | "linux_dedicated" | "unified_memory" | "unknown";
export type OsDetail = string;
export type PythonVersion = string;
export type Notes = string[];
export type Artifact = string | null;
export type Dependencies = string[];
export type Direct = boolean | null;
export type DirectUrl = string | null;
export type Editable = boolean | null;
export type InstalledFileCount = number | null;
export type Installer = string | null;
export type Name2 = string;
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
export type Packages1 = PackageLock[];
export type AvailableBytes = number | null;
export type TotalBytes = number | null;
export type FreeBytes = number | null;
export type Kind = "nvme" | "ssd" | "hdd" | "network" | "unknown";
export type ScratchPath = string | null;
export type WorkerId = string;
export type Probes = string[];
export type BackendId1 = string;
export type BackendVersion1 = string | null;
export type BitsandbytesOk = boolean;
export type ContractVersion2 = "1.0.0";
export type AdapterMethods1 = AdapterMethod[];
export type AttentionImpls1 = AttentionImpl[];
export type AttentionKernels1 = AttentionKernel[];
export type CheckpointImpls1 = CheckpointImpl[];
export type CommunicationBackends1 = CommunicationBackend[];
export type ExecutionContractVersion = string;
export type Probe = string;
export type RuntimeMode = "training" | "cpu_toy";
export type ExecutionCombinations = ExecutionCapabilityCombination[];
export type ExecutionContractVersions1 = string[];
export type LossImpls1 = LossImpl[];
export type ObjectiveCapabilities1 = string[];
export type OffloadStrategies1 = OffloadStrategy[];
export type Optimizers1 = Optimizer[];
export type ParallelismKinds1 = ParallelismKind[];
export type PlacementModes1 = PlacementMode[];
export type PlacementTiers1 = MemoryTier[];
export type PrecisionModes1 = PrecisionMode[];
export type QuantizationModes1 = QuantizationMode[];
export type TrainerFields1 = string[];
export type TrainerInitFields1 = string[];
export type GeneratedAt = string | null;
export type InstalledPackages = PackageLock[];
export type MissingPackages = string[];
export type Notes1 = string[];
export type Detail = string | null;
export type ExecutionCombinations1 = ExecutionCapabilityCombination[];
export type Probe1 = string;
export type ProbeResults = ProbeResult[];
export type Readiness = "ready" | "cpu_toy_only" | "not_ready";
export type HeartbeatIntervalSeconds = number;
export type Bias = ("none" | "all" | "lora_only") | null;
export type LoraAlpha = number | null;
export type LoraDropout = number | null;
export type LoraR = number | null;
export type TargetModules = string[] | null;
export type AllocatorPolicy = "default" | "expandable_segments" | "max_split_size" | "garbage_collection";
export type BaseModel = string;
export type FallbackGradAccumulationSteps = number | null;
export type MicroBatchSize = number;
export type SupervisedTokenAccumulationTarget = number;
export type CadenceOptimizerSteps = number | null;
export type CadenceSeconds = number | null;
export type KeepLast = number | null;
export type ReloadVerify = boolean;
export type CompileMode1 = "none" | "eager" | "reduce_overhead" | "max_autotune" | "aot_inductor";
export type ContractVersion3 = "1.0.0";
export type CreatedAt = string | null;
export type AfterRun = boolean;
export type BeforeRun = boolean;
export type EveryOptimizerSteps = number | null;
export type SuiteRef = string | null;
export type OutputDir = string;
export type GradientCheckpointing = boolean;
/**
 * The ``controlled_*`` values are the deliberate, planned counterparts of the accidental spills
 * in :class:`FitClass`.
 */
export type OffloadStrategy1 =
  | "none"
  | "controlled_activation_offload"
  | "controlled_optimizer_offload"
  | "controlled_parameter_offload"
  | "cpu_offload"
  | "disk_offload"
  | "deepspeed_zero2"
  | "deepspeed_zero3";
export type AdamBeta1 = number;
export type AdamBeta2 = number;
export type AdamEpsilon = number;
export type LearningRate = number;
export type LrScheduler = string | null;
export type MaxGradNorm = number;
export type WarmupRatio = number | null;
export type WeightDecay = number | null;
export type EvidenceStatus = "planned_not_measured";
export type EvictionPolicy = "none" | "lru" | "lfu" | "layer_window" | "heat_based";
export type OffloadMechanism = "cpu_copy" | "cuda_unified_memory" | "nvme_io" | "backend_native";
export type PrefetchPolicy = "none" | "static" | "layer_window" | "route_prediction" | "heat_based";
/**
 * What the physical scheduler does when requested state is not ready. Semantic fallback is
 * never implicit: it requires a separately pinned learned-policy reference.
 */
export type RouteMissAction = "wait" | "defer" | "fail" | "semantic_fallback";
export type RuleId = string;
export type ComponentIds = string[];
export type ExpertIds = string[];
export type ParameterScopeIds = string[];
export type WholeModel = boolean;
export type SourceResourceId = string;
export type PhysicalStateKind = "parameters" | "gradients" | "optimizer_state" | "activations";
export type TargetResourceId = string;
export type OffloadTrigger = "static" | "memory_pressure" | "ahead_of_use" | "after_use";
export type OffloadRules = OffloadRule[];
export type GroupId = string;
export type ParameterScopeIds1 = string[];
/**
 * @minItems 2
 */
export type Ranks = [number, number, ...number[]];
export type Groups = ParallelGroup[];
/**
 * @minItems 1
 */
export type Ranks1 = [RankBinding, ...RankBinding[]];
export type LocalRank = number;
export type NodeId = string;
export type Rank = number;
export type ResourceId = string;
export type WorldSize = number;
/**
 * @minItems 1
 */
export type Placements = [StatePlacement, ...StatePlacement[]];
export type PlacementId = string;
export type ResourceId1 = string;
export type PlacementRole = "authoritative" | "shard" | "replica" | "cache";
export type ShardCount = number | null;
export type ShardGroupId = string | null;
export type ShardIndex = number | null;
export type SourcePlacementId = string | null;
/**
 * @minItems 1
 */
export type Resources = [PhysicalResource, ...PhysicalResource[]];
export type DeviceId = string | null;
export type ResourceId2 = string;
/**
 * The per-role verdict for a candidate path. ``unsuitable`` is a hard no (data-loss or
 * thrash-to-a-halt risk); ``marginal`` will work but degrade (e.g. an HDD for offload); ``unknown``
 * when detection couldn't characterize the device (honest, never a false ``suitable``).
 */
export type StorageSuitability = "suitable" | "marginal" | "unsuitable" | "unknown";
export type DeviceMountPoint = string | null;
export type FreeBytes1 = number | null;
/**
 * How a storage device attaches. The interface — not just free space — decides whether a device
 * can sustain the heavy sequential + random writes of optimizer/parameter offload and checkpointing.
 * A USB bridge or a network mount will thrash under sustained offload even with terabytes free.
 */
export type StorageInterface = "nvme_pcie" | "sata_ssd" | "hdd" | "usb" | "network" | "virtual" | "unknown";
export type Path = string;
export type Reasons = string[];
export type RequiredFreeBytes = number | null;
/**
 * The role a path plays in a run. Roles differ in access pattern: ``optimizer_offload`` /
 * ``parameter_offload`` / ``scratch`` / ``checkpoints`` are WRITE-heavy; ``model_cache`` /
 * ``dataset_cache`` are read-LATENCY-sensitive during load; ``source_repo`` / ``python_env`` are
 * thousands of SMALL files touched on every process start (an import over a USB bridge or a WSL
 * ``/mnt`` mount stalls); ``archive`` just wants capacity. A path's suitability is judged PER ROLE (a
 * USB SSD is fine for ``archive``, poor for ``model_cache``, unfit for ``optimizer_offload``).
 */
export type StorageRole =
  | "os"
  | "source_repo"
  | "python_env"
  | "model_cache"
  | "dataset_cache"
  | "checkpoints"
  | "scratch"
  | "optimizer_offload"
  | "parameter_offload"
  | "artifacts"
  | "archive"
  | "logs";
/**
 * The per-role verdict for a candidate path. ``unsuitable`` is a hard no (data-loss or
 * thrash-to-a-halt risk); ``marginal`` will work but degrade (e.g. an HDD for offload); ``unknown``
 * when detection couldn't characterize the device (honest, never a false ``suitable``).
 */
export type StorageSuitability1 = "suitable" | "marginal" | "unsuitable" | "unknown";
export type Path1 = string;
export type RouteFidelity = "preserve_or_fail" | "declared_semantic_fallback";
export type PlanHash = string;
export type PlanId = string;
export type AdapterTaskType = "CAUSAL_LM";
export type EvidenceKind = "functional_probe" | "cpu_reference";
export type FallbackPolicy = "refuse";
export type FlashSdpEnabled = boolean;
export type MathSdpEnabled = boolean;
export type MemEfficientSdpEnabled = boolean;
/**
 * The model-loader API selected before execution.
 *
 * This is deliberately separate from :class:`AttentionKernel`: ``sdpa`` is an API that can
 * dispatch to several materially different PyTorch kernels.
 */
export type ModelAttentionApi = "eager" | "sdpa" | "flash_attention_2" | "flash_attention_3" | "xformers";
export type SafetyMandate = string | null;
/**
 * Whether the planner may seal a capability that lacks functional evidence.
 */
export type ExecutionVerificationRequirement = "require_verified" | "allow_unverified";
export type Bnb4BitUseDoubleQuant = boolean;
export type ConfigurationHash = string;
export type ConfigurationId = string;
export type ContractVersion4 = "1.0.0";
export type ChatTemplateSha256 = string | null;
export type DatasetFormat = "instruction" | "chat" | "trace";
export type DatasetTextField = string;
export type FormatterId = string;
export type FormatterSha256 = string;
export type Packing = boolean;
export type TruncationAnalysis = "full_pinned_dataset";
export type TruncationPolicy = "refuse" | "allow";
export type DataSeed = number;
/**
 * @minItems 1
 */
export type DeviceMap = [DeviceMapEntry, ...DeviceMapEntry[]];
export type Device = string;
export type Module = string;
export type EnvironmentBinding = "profile_snapshot" | "managed_lock";
export type GradientCheckpointing1 = boolean;
export type ContentSha256 = string | null;
export type Kind1 = "dataset" | "model" | "tokenizer";
export type Location = string;
export type ResolvedRevision = string | null;
export type Source1 = "local_file" | "local_directory" | "huggingface";
export type OutputDir1 = string;
export type OutputLayout = "run_scoped_v1";
export type PrecisionMode1 = "fp32" | "tf32" | "fp16" | "bf16" | "fp8" | "mixed_bf16" | "mixed_fp16";
export type OptimizerStateDtype = PrecisionMode | QuantizationMode;
export type QuantizationMode1 = "none" | "int8" | "int4" | "nf4" | "fp4" | "gptq" | "awq" | "hqq";
export type RuntimeMode1 = "training" | "cpu_toy";
export type SaveStrategy = "steps";
export type MaxSteps = number | null;
export type NumTrainEpochs = number | null;
export type Seed = number;
export type Buckets = number[];
export type MaxSequenceLen = number;
export type Packing1 = boolean;
export type TruncationAllowed = boolean;
export type DisableTqdm = boolean;
export type LoggingSteps = number;
/**
 * @minItems 1
 */
export type PackageVersions = [PackageLock, ...PackageLock[]];
export type ReportTo = string[];
/**
 * @minItems 1
 */
export type RequiredSftConfigFields = [string, ...string[]];
export type SequenceLengthField = "max_seq_length" | "max_length";
export type TokenizerParameter = "tokenizer" | "processing_class";
export type TrustRemoteCode = false;
export type UseSafetensors = true;
export type Seed1 = number;
export type RunId = string;
export type ExecutionConfigurationHash = string | null;
export type Pid = number | null;
export type ProcessStartedAt = string | null;
export type RunId1 = string;
export type ContractVersion5 = "1.0.0";
export type Detail1 = string | null;
export type DetectedAt = string | null;
export type ExceptionType = string | null;
export type ExitCode = number | null;
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
export type ContractVersion6 = "1.0.0";
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
export type RunId2 = string | null;
export type Signal = string | null;
export type Action = "cancel" | "pause" | "resume" | "checkpoint_now";
export type RunId3 = string;
export type ContractVersion7 = "1.0.0";
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
export type Message1 = string | null;
export type GpuUtilization = number | null;
export type GradNorm = number | null;
export type LearningRate1 = number | null;
export type Loss = number | null;
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
export type Notes2 = string;
export type ObservationId = string;
export type ComponentIds1 = string[];
export type CoordinateUniverseId = string;
export type CoordinateUniverseSha256 = string | null;
export type Definition1 = string;
export type DeviceId1 = string | null;
export type ExpertIds1 = string[];
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
export type ScopeId = string;
export type CapturedAt1 = string | null;
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
export type CapturedAt2 = string | null;
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
export type RunId4 = string;
export type Seq = number;
export type OptimizerStep1 = number | null;
export type PidAlive = boolean;
export type RunId5 = string;
export type ArtifactId = string;
export type BaseModel1 = string | null;
export type ContractVersion8 = "1.0.0";
export type CreatedAt1 = string | null;
export type CheapFingerprint = string | null;
export type ContentHash = string | null;
export type CurrentIntegrity = "ok" | "missing" | "modified" | "unknown";
export type Kind2 = "adapter" | "checkpoint" | "merged_model" | "gguf" | "onnx" | "quantized" | "other";
export type Notes3 = string;
export type Path2 = string;
export type ReloadVerified = boolean;
export type Status = "candidate" | "kept" | "rejected";
export type UpdatedAt = string | null;
export type Artifacts = ArtifactManifest[];
export type AdapterApplied = boolean | null;
export type Backend = string | null;
export type ChatTemplateApplied = boolean | null;
export type ContractVersion9 = "1.0.0";
export type DatasetFingerprint = string | null;
export type Name3 = string;
export type VersionRef = string | null;
export type EvalId = string;
export type BlockCount = number;
export type MaxRegressionScoreDrop = number | null;
export type MinEvalAverageScore = number | null;
export type MinEvalPassRate = number | null;
export type OverallStatus = "pass" | "warn" | "block";
export type PassCount = number;
export type WarnCount = number;
export type GeneratedAt1 = string | null;
export type JudgeModel = string | null;
export type Measures = string;
export type Name4 = "keyword_overlap" | "llm_judge" | "exact_match" | "pass_rate" | "custom";
export type ScoreThreshold = number | null;
export type ProvenanceCaveat = string | null;
export type ReportRef = string | null;
export type AverageManualScore = number | null;
export type AverageScore = number;
export type ExamplesTested = number;
export type FailedExamples = number;
export type PassRate = number | null;
export type WeakTags = string[];
export type ArtifactRef = string | null;
export type Model1 = string;
export type Phase = "before" | "after" | "standalone";
export type RunRef = string | null;
export type RunId6 = string;
export type ArtifactIds = string[];
export type BaseModel2 = string;
export type Checkpoints = string[];
export type ContractVersion10 = "1.0.0";
export type CreatedAt2 = string;
export type AfterEvalModel = string | null;
export type AfterEvalRef = string | null;
export type BeforeEvalRef = string | null;
export type FinishedAt = string | null;
export type Notes4 = string;
export type OutputDir2 = string;
export type ParameterAccountingRefs = Ref[];
export type Argv = string[];
export type ExitCode1 = number | null;
export type Pid1 = number | null;
export type ProcessStartedAt1 = string | null;
export type ConfigSha256 = string | null;
export type DatasetFingerprint1 = string | null;
export type DatasetRowCount = number;
export type EngineVersion1 = string;
export type Platform = string;
export type PythonVersion1 = string;
export type RunId7 = string;
export type StartedAt = string | null;
export type State = "prepared" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";
export type Target = string;
export type UpdatedAt1 = string;
export type CorrelationId = string | null;
export type Direction = "core_to_worker" | "worker_to_core";
export type MessageId = string;
export type ProtocolVersion = "2.0.0";
export type SentAt = string | null;
export type Type =
  | "hello"
  | "capability_probe_request"
  | "capability_report"
  | "run_dispatch"
  | "run_accepted"
  | "run_rejected"
  | "run_control"
  | "event"
  | "heartbeat"
  | "terminal_result"
  | "failure";

/**
 * The versioned envelope for the core↔worker channel — realizes the 'immutable RunPlan IN,
 * structured RunEvent stream OUT' boundary. Protocol 2.0 uses a mandatory worker-first identity
 * handshake. ``protocol_version`` evolves independently of any single contract's version. The body
 * union is language-neutral and must match ``type`` (see :data:`WORKER_BODY_BY_TYPE`).
 */
export interface WorkerMessage {
  body: Body;
  correlation_id?: CorrelationId;
  direction: Direction;
  message_id: MessageId;
  protocol_version: ProtocolVersion;
  sent_at?: SentAt;
  type: Type;
}
/**
 * Worker→core handshake: who I am + what I can do.
 */
export interface HelloBody {
  backend: BackendManifest;
  environment?: EnvironmentProfile | null;
  environment_ref: Ref;
  worker_id: WorkerId;
}
/**
 * A backend worker's STATIC self-declaration of everything it can do — the core reads this to
 * decide which backend can even attempt a RunPlan, BEFORE dispatch. Mostly NEW; generalizes the
 * inference model_backends.base.ModelBackend Protocol + training/compatibility into a declarable
 * manifest for a TRAINING backend.
 */
export interface BackendManifest {
  adapter_methods?: AdapterMethods;
  attention_impls?: AttentionImpls;
  attention_kernels?: AttentionKernels;
  backend_id: BackendId;
  backend_version: BackendVersion;
  capability_probes?: CapabilityProbes;
  checkpoint_impls?: CheckpointImpls;
  checkpoint_semantics?: CheckpointSemantics | null;
  communication_backends?: CommunicationBackends;
  compile_modes?: CompileModes;
  contract_version?: ContractVersion;
  dependency_conflicts?: DependencyConflicts;
  dependency_requirements?: DependencyRequirements;
  display_name?: DisplayName;
  environment_lock_ref?: Ref | null;
  execution_contract_versions?: ExecutionContractVersions;
  export_compatibility?: ExportCompatibility;
  export_formats?: ExportFormats;
  known_failure_modes?: KnownFailureModes;
  loss_impls?: LossImpls;
  model_families?: ModelFamilies;
  objective_capabilities?: ObjectiveCapabilities;
  offload_strategies?: OffloadStrategies;
  optimizers?: Optimizers;
  parallelism_kinds?: ParallelismKinds;
  placement_modes?: PlacementModes;
  placement_tiers?: PlacementTiers;
  precision_modes?: PrecisionModes;
  quantization_modes?: QuantizationModes;
  required_compute_capability?: RequiredComputeCapability;
  supported_devices: SupportedDevices;
  supported_os: SupportedOs;
  task_types: TaskTypes;
  telemetry_hooks?: TelemetryHooks;
  trainer_fields?: TrainerFields;
  trainer_init_fields?: TrainerInitFields;
  trainer_target?: TrainerTarget | null;
}
export interface CheckpointSemantics {
  contains?: Contains;
  reload_verifiable?: ReloadVerifiable;
  resumable?: Resumable;
}
export interface DependencyConflict {
  condition: Condition;
  packages: Packages;
  severity?: Severity;
}
export interface DependencyRequirement {
  name: Name;
  optional?: Optional;
  reason?: Reason;
  specifier?: Specifier;
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
export interface ExportCompatibilityEntry {
  format: ExportFormat;
  serves_in?: ServesIn;
}
/**
 * Pre-declared hazards tagged with a taxonomy so the core can warn/refuse up front. The
 * canonical example: fused attention deadlocks on sm_120.
 */
export interface KnownFailureMode {
  condition: Condition1;
  description?: Description;
  mitigation?: Mitigation;
  taxonomy: FailureTaxonomy;
}
export interface TelemetryHook {
  metrics?: Metrics;
  stage: StageMarker;
}
/**
 * The full, hashable SIGNATURE of a host + software environment. Formalizes + greatly extends
 * environment.probe_training_runtime (package versions + GpuInfo), gpu_probe.probe_gpu_memory, and
 * provenance.RunProvenance. A RunManifest/RunPlan references a profile by ``environment_signature``
 * so a result is always tied to the exact environment that produced it.
 */
export interface EnvironmentProfile {
  accelerator_runtime?: AcceleratorRuntime | null;
  captured_at?: CapturedAt;
  contract_version?: ContractVersion1;
  cpu?: EnvCpu | null;
  engine_version?: EngineVersion;
  environment_signature: EnvironmentSignature;
  gpus?: Gpus;
  host: EnvHost;
  notes?: Notes;
  packages?: Packages1;
  ram?: EnvRam | null;
  storage?: EnvStorage | null;
}
export interface AcceleratorRuntime {
  cuda_driver_version?: CudaDriverVersion;
  cuda_runtime_version?: CudaRuntimeVersion;
  driver_version?: DriverVersion;
  kind?: DeviceKind | null;
  mps_available?: MpsAvailable;
  nvidia_smi_available?: NvidiaSmiAvailable;
  rocm_version?: RocmVersion;
}
export interface EnvCpu {
  instruction_sets?: InstructionSets;
  logical_cores?: LogicalCores;
  model?: Model;
  physical_cores?: PhysicalCores;
}
/**
 * One accelerator. Grounded in environment.GpuInfo + gpu_probe.GpuMemory.
 */
export interface GpuDevice {
  compute_capability?: ComputeCapability;
  compute_capability_major?: ComputeCapabilityMajor;
  index: Index;
  kind: DeviceKind;
  name: Name1;
  pcie?: GpuPcie | null;
  supported_dtypes?: SupportedDtypes;
  vram_free_bytes?: VramFreeBytes;
  vram_total_bytes?: VramTotalBytes;
}
export interface GpuPcie {
  gen?: Gen;
  width?: Width;
}
export interface EnvHost {
  hostname_hash?: HostnameHash;
  memory_residency_model?: MemoryResidencyModel;
  os: OperatingSystem;
  os_detail?: OsDetail;
  python_version?: PythonVersion;
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
  name: Name2;
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
export interface EnvRam {
  available_bytes?: AvailableBytes;
  total_bytes?: TotalBytes;
}
export interface EnvStorage {
  free_bytes?: FreeBytes;
  kind?: Kind;
  scratch_path?: ScratchPath;
}
/**
 * Core→worker: run these probes on this host and reply with a CapabilityReport.
 */
export interface CapabilityProbeRequestBody {
  environment_ref?: Ref | null;
  probes?: Probes;
}
/**
 * The DYNAMIC, measured counterpart of a BackendManifest: probe results against a specific
 * EnvironmentProfile. Formalizes environment.probe_training_runtime (ready/cpu_toy_ready/
 * bitsandbytes_ok/notes) and generalizes it to per-probe outcomes tagged with FailureTaxonomy.
 */
export interface CapabilityReport {
  backend_id: BackendId1;
  backend_version?: BackendVersion1;
  bitsandbytes_ok?: BitsandbytesOk;
  contract_version?: ContractVersion2;
  effective_capabilities?: EffectiveCapabilities | null;
  environment_ref: Ref;
  generated_at?: GeneratedAt;
  installed_packages?: InstalledPackages;
  missing_packages?: MissingPackages;
  notes?: Notes1;
  probe_results?: ProbeResults;
  readiness: Readiness;
}
/**
 * The intersection of what a backend DECLARES and what PROVED to work on this host. The planner
 * resolves a RunPlan against this, not the raw BackendManifest.
 */
export interface EffectiveCapabilities {
  adapter_methods?: AdapterMethods1;
  attention_impls?: AttentionImpls1;
  attention_kernels?: AttentionKernels1;
  checkpoint_impls?: CheckpointImpls1;
  communication_backends?: CommunicationBackends1;
  execution_combinations?: ExecutionCombinations;
  execution_contract_versions?: ExecutionContractVersions1;
  loss_impls?: LossImpls1;
  objective_capabilities?: ObjectiveCapabilities1;
  offload_strategies?: OffloadStrategies1;
  optimizers?: Optimizers1;
  parallelism_kinds?: ParallelismKinds1;
  placement_modes?: PlacementModes1;
  placement_tiers?: PlacementTiers1;
  precision_modes?: PrecisionModes1;
  quantization_modes?: QuantizationModes1;
  trainer_fields?: TrainerFields1;
  trainer_init_fields?: TrainerInitFields1;
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
/**
 * Core→worker: execute this immutable plan (passed by value so the worker needs no shared
 * state).
 */
export interface RunDispatchBody {
  heartbeat_interval_seconds?: HeartbeatIntervalSeconds;
  plan: RunPlan;
  run_id: RunId;
}
/**
 * The IMMUTABLE, fully-resolved execution plan the core dispatches to a worker: no ambiguity is
 * left for the worker to decide. Formalizes + hardens config_templates.TrainingConfigTemplate. Key
 * upgrades: attention_backend defaults to math on Blackwell; the accumulation target is in
 * SUPERVISED TOKENS; ``plan_hash`` seals immutability (a change means a NEW plan).
 */
export interface RunPlan {
  adapter: AdapterSpec;
  allocator_policy?: AllocatorPolicy;
  attention_backend: AttentionImpl;
  backend_ref: Ref;
  base_model: BaseModel;
  batching: BatchingSpec;
  checkpoint_policy: CheckpointPolicy;
  compile_mode?: CompileMode1;
  contract_version?: ContractVersion3;
  created_at?: CreatedAt;
  dataset_ref: Ref;
  environment_ref: Ref;
  eval_schedule?: EvalSchedule;
  export: ExportSpec;
  gradient_checkpointing?: GradientCheckpointing;
  loss_impl: LossImpl;
  offload_strategy?: OffloadStrategy1;
  optimizer: OptimizerSpec;
  parameter_accounting_ref?: Ref | null;
  physical_execution?: PhysicalExecutionSpec | null;
  plan_hash: PlanHash;
  plan_id: PlanId;
  precision: PrecisionMode;
  quantization: QuantizationMode;
  resolved_execution?: ResolvedExecutionConfiguration | null;
  seed?: Seed1;
  sequence: SequenceSpec;
  task_type: TaskType;
  training_config_snapshot?: TrainingConfigSnapshot;
}
export interface AdapterSpec {
  bias?: Bias;
  lora_alpha?: LoraAlpha;
  lora_dropout?: LoraDropout;
  lora_r?: LoraR;
  method: AdapterMethod;
  target_modules?: TargetModules;
}
/**
 * The accumulation TARGET is expressed in SUPERVISED TOKENS, not microbatch count. A
 * token-target-CAPABLE backend accumulates until summed supervised tokens >= target and normalizes
 * the loss by supervised tokens across the window, so the effective batch is invariant to sequence
 * length + padding. The first-party ``corpus_studio`` reference trainer is NOT token-target-capable
 * today — it honors ``fallback_grad_accumulation_steps`` (a fixed microbatch count); the token
 * target is the contract a future token-aware backend would satisfy.
 */
export interface BatchingSpec {
  fallback_grad_accumulation_steps?: FallbackGradAccumulationSteps;
  micro_batch_size?: MicroBatchSize;
  supervised_token_accumulation_target: SupervisedTokenAccumulationTarget;
}
export interface CheckpointPolicy {
  cadence_optimizer_steps?: CadenceOptimizerSteps;
  cadence_seconds?: CadenceSeconds;
  impl: CheckpointImpl;
  keep_last?: KeepLast;
  reload_verify?: ReloadVerify;
}
export interface EvalSchedule {
  after_run?: AfterRun;
  before_run?: BeforeRun;
  every_optimizer_steps?: EveryOptimizerSteps;
  suite_ref?: SuiteRef;
}
export interface ExportSpec {
  format: ExportFormat;
  output_dir?: OutputDir;
}
export interface OptimizerSpec {
  adam_beta1?: AdamBeta1;
  adam_beta2?: AdamBeta2;
  adam_epsilon?: AdamEpsilon;
  impl: Optimizer;
  learning_rate: LearningRate;
  lr_scheduler?: LrScheduler;
  max_grad_norm?: MaxGradNorm;
  warmup_ratio?: WarmupRatio;
  weight_decay?: WeightDecay;
}
/**
 * The physical scheduler input, kept separate from learned semantic routing. Every field is
 * planned intent sealed by RunPlan; it is not runtime residency or fit evidence.
 */
export interface PhysicalExecutionSpec {
  evidence_status?: EvidenceStatus;
  offload_rules?: OffloadRules;
  parallelism: ParallelismSpec;
  placements: Placements;
  resources: Resources;
  route_fidelity?: RouteFidelity;
  semantic_fallback_policy_ref?: Ref | null;
  storage_profile_ref?: Ref | null;
}
export interface OffloadRule {
  eviction_policy?: EvictionPolicy;
  mechanism: OffloadMechanism;
  prefetch_policy?: PrefetchPolicy;
  route_miss_action?: RouteMissAction;
  rule_id: RuleId;
  selector: PhysicalScopeSelector;
  source_resource_id: SourceResourceId;
  state: PhysicalStateKind;
  target_resource_id: TargetResourceId;
  trigger: OffloadTrigger;
}
/**
 * Select planned state by stable logical identity. Empty identity lists mean nothing, never an
 * inferred dense model. ``whole_model`` is the explicit dense-safe fallback for unknown topology.
 */
export interface PhysicalScopeSelector {
  component_ids?: ComponentIds;
  expert_ids?: ExpertIds;
  parameter_scope_ids?: ParameterScopeIds;
  whole_model?: WholeModel;
}
/**
 * Explicit rank/group topology. Groups may overlap across axes, so the contract never assumes
 * that data x tensor x pipeline x expert degrees form one universal product.
 */
export interface ParallelismSpec {
  groups?: Groups;
  ranks: Ranks1;
  world_size?: WorldSize;
}
export interface ParallelGroup {
  communication_backend: CommunicationBackend;
  group_id: GroupId;
  kind: ParallelismKind;
  parameter_scope_ids?: ParameterScopeIds1;
  ranks: Ranks;
}
export interface RankBinding {
  local_rank?: LocalRank;
  node_id?: NodeId;
  rank: Rank;
  resource_id: ResourceId;
}
export interface StatePlacement {
  placement_id: PlacementId;
  resource_id: ResourceId1;
  role: PlacementRole;
  selector: PhysicalScopeSelector;
  shard_count?: ShardCount;
  shard_group_id?: ShardGroupId;
  shard_index?: ShardIndex;
  source_placement_id?: SourcePlacementId;
  state: PhysicalStateKind;
}
/**
 * One planned physical tier/device. This is scheduling intent, never measured residency.
 */
export interface PhysicalResource {
  device_id?: DeviceId;
  device_kind?: DeviceKind | null;
  resource_id: ResourceId2;
  storage?: PlannedStorageBinding | null;
  tier: MemoryTier;
}
/**
 * The exact StorageProfile assessment accepted by a plan. ``marginal``/``unknown`` are usable
 * only when that same verdict is explicitly recorded in ``accepted_suitability``; ``unsuitable`` is
 * always refused.
 */
export interface PlannedStorageBinding {
  accepted_suitability?: StorageSuitability;
  assessment: StorageRoleAssessment;
  path: Path1;
  role: StorageRole;
}
/**
 * The PER-ROLE verdict for a candidate path: can it play this role, and if not, WHY. The reasons
 * are the safe-spill guardrail's human-readable justification (USB bridge / synced folder / free-space
 * margin / inside the source repo / rotational disk).
 */
export interface StorageRoleAssessment {
  device_mount_point?: DeviceMountPoint;
  free_bytes?: FreeBytes1;
  interface?: StorageInterface;
  path: Path;
  reasons?: Reasons;
  required_free_bytes?: RequiredFreeBytes;
  role: StorageRole;
  suitability: StorageSuitability1;
}
/**
 * The hash-sealed configuration consumed directly by an isolated training worker.
 *
 * It contains every execution-affecting default. Workers may refuse it, but may not fill in,
 * filter, reinterpret, or override semantic fields after this configuration is sealed.
 */
export interface ResolvedExecutionConfiguration {
  adapter: AdapterSpec;
  adapter_task_type?: AdapterTaskType;
  attention: AttentionExecutionPolicy;
  backend_ref: Ref;
  batching: BatchingSpec;
  bnb_4bit_use_double_quant: Bnb4BitUseDoubleQuant;
  capability_report_ref: Ref;
  checkpoint_policy: CheckpointPolicy;
  configuration_hash: ConfigurationHash;
  configuration_id: ConfigurationId;
  contract_version?: ContractVersion4;
  data: TrainingDataPolicy;
  data_seed?: DataSeed;
  device_map: DeviceMap;
  environment_binding: EnvironmentBinding;
  environment_ref: Ref;
  export_format: ExportFormat;
  gradient_checkpointing?: GradientCheckpointing1;
  inputs: ExecutionInputs;
  loss_impl: LossImpl;
  objective_ref: Ref;
  optimizer: OptimizerSpec;
  output_dir: OutputDir1;
  output_layout?: OutputLayout;
  precision: PrecisionExecutionPolicy;
  runtime_mode: RuntimeMode1;
  save_strategy?: SaveStrategy;
  schedule: TrainingSchedule;
  seed?: Seed;
  sequence: SequenceSpec;
  trainer_interface: TrainerInterfacePolicy;
  trust_remote_code?: TrustRemoteCode;
  use_safetensors?: UseSafetensors;
}
/**
 * Exact model attention API plus the one runtime kernel that is permitted.
 */
export interface AttentionExecutionPolicy {
  effective_backend_required: AttentionKernel;
  evidence_kind: EvidenceKind;
  fallback_policy?: FallbackPolicy;
  flash_attention_package?: PackageLock | null;
  flash_sdp_enabled: FlashSdpEnabled;
  kernel_probe_ref: Ref;
  math_sdp_enabled: MathSdpEnabled;
  mem_efficient_sdp_enabled: MemEfficientSdpEnabled;
  model_attention_api: ModelAttentionApi;
  safety_mandate?: SafetyMandate;
  verification_requirement?: ExecutionVerificationRequirement;
}
export interface TrainingDataPolicy {
  chat_template_sha256?: ChatTemplateSha256;
  dataset_format: DatasetFormat;
  dataset_text_field?: DatasetTextField;
  formatter_id: FormatterId;
  formatter_sha256: FormatterSha256;
  packing?: Packing;
  truncation_analysis?: TruncationAnalysis;
  truncation_policy?: TruncationPolicy;
}
export interface DeviceMapEntry {
  device: Device;
  module: Module;
}
export interface ExecutionInputs {
  dataset: ExecutionInputBinding;
  model: ExecutionInputBinding;
  tokenizer: ExecutionInputBinding;
}
/**
 * One immutable input consumed by the worker.
 *
 * Local inputs pin the exact bytes (a stable file or directory digest). Hugging Face inputs pin an
 * immutable repository commit; a branch or tag is never sufficient execution identity.
 */
export interface ExecutionInputBinding {
  content_sha256?: ContentSha256;
  kind: Kind1;
  location: Location;
  ref: Ref;
  resolved_revision?: ResolvedRevision;
  source: Source1;
}
/**
 * The numerical representation of each material training state.
 *
 * ``weight_storage_dtype`` describes an unquantized frozen base; quantized bases use
 * ``quantized_storage_format`` instead. ``master_weight_dtype`` describes the trainable adapter
 * parameters. An 8-bit optimizer may use quantized primary state plus FP32 auxiliary tensors.
 */
export interface PrecisionExecutionPolicy {
  dequantization_dtype: PrecisionMode;
  forward_compute_dtype: PrecisionMode;
  gradient_dtype: PrecisionMode;
  master_weight_dtype?: PrecisionMode | null;
  optimizer_auxiliary_dtype?: PrecisionMode1;
  optimizer_state_dtype: OptimizerStateDtype;
  quantized_storage_format?: QuantizationMode1;
  weight_storage_dtype?: PrecisionMode | null;
}
export interface TrainingSchedule {
  max_steps?: MaxSteps;
  num_train_epochs?: NumTrainEpochs;
}
/**
 * Sequence handling. Buckets let variable-length rows batch efficiently; the max bucket ==
 * the trainer's sequence_len (config_templates.sequence_len, default 4096).
 */
export interface SequenceSpec {
  buckets?: Buckets;
  max_sequence_len?: MaxSequenceLen;
  packing?: Packing1;
  truncation_allowed?: TruncationAllowed;
}
/**
 * Version- and field-exact adapter to the installed TRL/Transformers surface.
 */
export interface TrainerInterfacePolicy {
  disable_tqdm?: DisableTqdm;
  logging_steps?: LoggingSteps;
  package_versions: PackageVersions;
  report_to?: ReportTo;
  required_sft_config_fields: RequiredSftConfigFields;
  sequence_length_field: SequenceLengthField;
  tokenizer_parameter: TokenizerParameter;
}
export interface TrainingConfigSnapshot {
  [k: string]: unknown;
}
export interface RunAcceptedBody {
  execution_configuration_hash?: ExecutionConfigurationHash;
  pid?: Pid;
  process_started_at?: ProcessStartedAt;
  run_id: RunId1;
}
/**
 * A structured, classified terminal outcome for a run, capability probe, or export. The
 * taxonomy turns 'it died' into an actionable category — a real OOM vs a KERNEL_STALL (the sm_120
 * fused-attention deadlock) vs an ACCIDENTAL_SPILL vs a CONTROLLED_OFFLOAD. NEW.
 */
export interface FailureRecord {
  contract_version?: ContractVersion5;
  detail?: Detail1;
  detected_at?: DetectedAt;
  exception_type?: ExceptionType;
  exit_code?: ExitCode;
  fit_at_failure?: FitClassification | null;
  memory_at_failure?: MemoryMetrics | null;
  message: Message;
  reconciled?: Reconciled;
  remediation?: Remediation;
  run_id?: RunId2;
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
  contract_version?: ContractVersion6;
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
export interface RunControlBody {
  action: Action;
  run_id: RunId3;
}
/**
 * One envelope in the structured telemetry stream a worker emits for a run — the RunEvent half
 * of the immutable-RunPlan-in / RunEvent-stream-out worker protocol. NEW; the engine has no
 * streaming telemetry today (run_registry is a durable per-run record, not an event stream).
 */
export interface RunEvent {
  contract_version?: ContractVersion7;
  emitted_at: EmittedAt;
  epoch?: Epoch;
  event_type: EventType;
  fit?: FitClassification | null;
  message?: Message1;
  metrics?: EventMetrics | null;
  microstep?: Microstep;
  optimizer_step?: OptimizerStep;
  payload?: Payload;
  run_id: RunId4;
  seq: Seq;
  stage?: StageMarker | null;
}
/**
 * Present on metric/heartbeat events. All optional — a worker emits what it can sample. The
 * memory block + step_time make the WDDM spill (10-25x slowdown, non-zero shared bytes) visible.
 */
export interface EventMetrics {
  gpu_utilization?: GpuUtilization;
  grad_norm?: GradNorm;
  learning_rate?: LearningRate1;
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
  notes?: Notes2;
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
  component_ids?: ComponentIds1;
  coordinate_universe_id: CoordinateUniverseId;
  coordinate_universe_sha256?: CoordinateUniverseSha256;
  definition: Definition1;
  device_id?: DeviceId1;
  expert_ids?: ExpertIds1;
  kind: ParameterScopeKind;
  memory_tier?: MemoryTier | null;
  model_ref: Ref;
  scope_id: ScopeId;
}
export interface ParameterEvidenceSource {
  backend_ref?: Ref | null;
  captured_at?: CapturedAt1;
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
  captured_at?: CapturedAt2;
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
export interface HeartbeatBody {
  optimizer_step?: OptimizerStep1;
  pid_alive?: PidAlive;
  run_id: RunId5;
  stage?: StageMarker | null;
}
/**
 * Worker→core: the run ended. A FailureRecord is present iff the outcome was not PASS.
 */
export interface TerminalResultBody {
  artifacts?: Artifacts;
  failure?: FailureRecord | null;
  final_eval?: EvaluationResult | null;
  outcome: FailureTaxonomy;
  run_id: RunId6;
  run_manifest: RunManifest;
}
/**
 * A first-class record of a weight artifact a run produced. Formalizes
 * artifact_registry.ModelArtifactRecord + its two-tier integrity model. The platform NEVER
 * moves/copies/deletes the underlying weights — the manifest only references + re-checks them.
 */
export interface ArtifactManifest {
  artifact_id: ArtifactId;
  base_model?: BaseModel1;
  contract_version?: ContractVersion8;
  created_at?: CreatedAt1;
  integrity?: ArtifactIntegrity | null;
  kind?: Kind2;
  notes?: Notes3;
  parameter_accounting_ref?: Ref | null;
  path: Path2;
  producer_run_ref: Ref;
  reload_verified?: ReloadVerified;
  status?: Status;
  updated_at?: UpdatedAt;
}
/**
 * Two-tier integrity: cheap size+mtime fingerprint powers the fast LIST; content_hash (sha256
 * over weight bytes) powers the promote GATE. ``current_integrity`` is computed LIVE at read.
 */
export interface ArtifactIntegrity {
  cheap_fingerprint?: CheapFingerprint;
  content_hash?: ContentHash;
  current_integrity?: CurrentIntegrity;
}
/**
 * The outcome of evaluating a model/dataset, with an explicit as-served vs raw distinction so a
 * number is never presented as a quality signal without saying what produced it. Formalizes
 * evaluation/reports.EvaluationReport + gates/models.GateReport.
 */
export interface EvaluationResult {
  as_served?: AsServed | null;
  contract_version?: ContractVersion9;
  dataset?: EvalDataset | null;
  eval_id: EvalId;
  gate?: EvalGate | null;
  generated_at?: GeneratedAt1;
  metric: EvalMetric;
  parameter_accounting_ref?: Ref | null;
  provenance_caveat?: ProvenanceCaveat;
  report_ref?: ReportRef;
  summary: EvalSummary;
  target: EvalTarget;
}
/**
 * How the model was actually served — the RAW-vs-AS-SERVED axis. Two evals of the 'same' model
 * differ if quantization/adapter/template/decoding differ.
 */
export interface AsServed {
  adapter_applied?: AdapterApplied;
  backend?: Backend;
  chat_template_applied?: ChatTemplateApplied;
  decoding?: Decoding;
  precision?: PrecisionMode | null;
  quantization?: QuantizationMode | null;
}
export interface Decoding {
  [k: string]: unknown;
}
export interface EvalDataset {
  dataset_fingerprint?: DatasetFingerprint;
  name?: Name3;
  version_ref?: VersionRef;
}
/**
 * Grounded in gates/models.GateReport/GateStatus (pass/warn/block, counts, effective
 * thresholds behind the verdict for reproducibility).
 */
export interface EvalGate {
  block_count?: BlockCount;
  max_regression_score_drop?: MaxRegressionScoreDrop;
  min_eval_average_score?: MinEvalAverageScore;
  min_eval_pass_rate?: MinEvalPassRate;
  overall_status: OverallStatus;
  pass_count?: PassCount;
  warn_count?: WarnCount;
}
/**
 * The scorer AND what it measures — so a number is never treated as quality without
 * qualification (reports.EvaluationReport.metric honesty note).
 */
export interface EvalMetric {
  judge_model?: JudgeModel;
  measures?: Measures;
  name?: Name4;
  score_threshold?: ScoreThreshold;
}
export interface EvalSummary {
  average_manual_score?: AverageManualScore;
  average_score: AverageScore;
  examples_tested: ExamplesTested;
  failed_examples?: FailedExamples;
  pass_rate?: PassRate;
  weak_tags?: WeakTags;
}
export interface EvalTarget {
  artifact_ref?: ArtifactRef;
  model: Model1;
  phase?: Phase;
  run_ref?: RunRef;
}
/**
 * A single run INSTANCE: the crash-safe durable record of one execution of a RunPlan.
 * Formalizes run_registry.TrainingRunRecord almost field-for-field + its state machine (terminal =
 * {succeeded, failed, cancelled, interrupted}; a dead-pid 'running' record reconciles to
 * interrupted).
 */
export interface RunManifest {
  artifact_ids?: ArtifactIds;
  base_model?: BaseModel2;
  checkpoints?: Checkpoints;
  contract_version?: ContractVersion10;
  created_at: CreatedAt2;
  dataset_ref?: Ref | null;
  environment_ref?: Ref | null;
  evaluation?: RunEvaluationLink | null;
  failure?: FailureRecord | null;
  final_fit?: FitClassification | null;
  finished_at?: FinishedAt;
  notes?: Notes4;
  output_dir?: OutputDir2;
  parameter_accounting_refs?: ParameterAccountingRefs;
  plan_ref: Ref;
  process?: RunProcessInfo | null;
  reproducibility?: RunReproducibility | null;
  run_id: RunId7;
  started_at?: StartedAt;
  state?: State;
  target?: Target;
  updated_at: UpdatedAt1;
}
export interface RunEvaluationLink {
  after_eval_model?: AfterEvalModel;
  after_eval_ref?: AfterEvalRef;
  before_eval_ref?: BeforeEvalRef;
}
/**
 * Process identity so a recycled pid is never mistaken for a live run. A 'running' record whose
 * pid is not alive reconciles to 'interrupted' (run_registry.reconcile_running_records).
 */
export interface RunProcessInfo {
  argv?: Argv;
  exit_code?: ExitCode1;
  pid?: Pid1;
  process_started_at?: ProcessStartedAt1;
}
/**
 * Embedded reproducibility manifest (provenance.RunProvenance) for a self-contained audit.
 */
export interface RunReproducibility {
  config_sha256?: ConfigSha256;
  dataset_fingerprint?: DatasetFingerprint1;
  dataset_row_count?: DatasetRowCount;
  engine_version?: EngineVersion1;
  platform?: Platform;
  python_version?: PythonVersion1;
}
