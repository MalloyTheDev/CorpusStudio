"""Shared controlled vocabularies for the platform contracts.

These are the language-neutral token sets a backend's declared capabilities and a plan's resolved
choices both draw from — a RunPlan can never name a precision/quant/adapter mode a BackendManifest
cannot describe. Grounded where the engine already fixes vocabulary (training/config_templates,
training/compatibility, model_backends, and the Blackwell math-SDPA path in training/environment).

Pure stdlib — no heavy imports. `import corpus_studio.platform` pulls no torch.
"""

from __future__ import annotations

from enum import Enum


class OperatingSystem(str, Enum):
    windows = "windows"
    # WSL is its OWN platform, not Windows and not bare Linux: it runs a Linux CUDA userspace, so the
    # fused FLASH SDPA kernel that DEADLOCKS on native Windows (WDDM) runs fine here — yet its GPU
    # memory still spills to shared system RAM via the host WDDM driver (wddm residency), so it
    # degrades to slow-but-training instead of hard-OOMing like bare Linux. Flash-safe like Linux,
    # spill-prone like Windows. Verified on a real RTX 5070 under WSL2.
    wsl = "wsl"
    linux = "linux"
    macos = "macos"
    unknown = "unknown"


class MemoryResidencyModel(str, Enum):
    """How the platform maps device memory. ``wddm`` (Windows) silently spills overflow to shared
    system RAM and thrashes over PCIe; ``linux_dedicated`` hard-OOMs instead; ``unified_memory`` is
    Apple MPS / integrated shared memory. The single most decisive field for spill-vs-OOM."""

    wddm = "wddm"
    linux_dedicated = "linux_dedicated"
    unified_memory = "unified_memory"
    unknown = "unknown"


class DeviceKind(str, Enum):
    cuda = "cuda"
    rocm = "rocm"
    mps = "mps"
    xpu = "xpu"
    cpu = "cpu"


class TaskType(str, Enum):
    sft = "sft"
    pretraining = "pretraining"
    preference = "preference"
    reward = "reward"
    classification = "classification"
    embedding = "embedding"
    multimodal = "multimodal"
    evaluation = "evaluation"
    distillation = "distillation"
    grpo = "grpo"


class ModelSourceKind(str, Enum):
    """Where a model/tokenizer identity originated. A local snapshot may still carry a repository
    and revision, but network retrieval is never implied by this value."""

    local = "local"
    huggingface = "huggingface"
    ollama = "ollama"
    artifact = "artifact"
    generated = "generated"
    external = "external"
    unknown = "unknown"


class ModelFormat(str, Enum):
    safetensors = "safetensors"
    pytorch_pickle = "pytorch_pickle"
    gguf = "gguf"
    onnx = "onnx"
    torchscript = "torchscript"
    numpy = "numpy"
    other = "other"
    unknown = "unknown"


class ModelTaskClass(str, Enum):
    causal_lm = "causal_lm"
    masked_lm = "masked_lm"
    seq2seq_lm = "seq2seq_lm"
    classification = "classification"
    embedding = "embedding"
    reranker = "reranker"
    reward_model = "reward_model"
    vision = "vision"
    speech = "speech"
    multimodal = "multimodal"
    custom = "custom"
    unknown = "unknown"


class ParameterCountKind(str, Enum):
    """Distinct parameter quantities required for dense-safe and MoE-safe accounting."""

    logical = "logical"
    active_token = "active_token"
    active_sequence = "active_sequence"
    touched_window = "touched_window"
    resident = "resident"
    updated_window = "updated_window"
    exposed_window = "exposed_window"
    effective = "effective"


class EvidenceKind(str, Enum):
    measured = "measured"
    estimated = "estimated"
    declared = "declared"
    unknown = "unknown"


class CountHandling(str, Enum):
    included = "included"
    excluded = "excluded"
    deduplicated = "deduplicated"
    represented_separately = "represented_separately"
    not_applicable = "not_applicable"
    unknown = "unknown"


class ModelExecutionKind(str, Enum):
    dense = "dense"
    sparse = "sparse"
    mixture_of_experts = "mixture_of_experts"
    conditional = "conditional"
    hybrid = "hybrid"
    unknown = "unknown"


class ModelAttentionType(str, Enum):
    full = "full"
    sliding_window = "sliding_window"
    block_sparse = "block_sparse"
    linear = "linear"
    state_space = "state_space"
    hybrid = "hybrid"
    custom = "custom"
    unknown = "unknown"


class PositionalEncoding(str, Enum):
    rope = "rope"
    alibi = "alibi"
    absolute = "absolute"
    relative = "relative"
    none = "none"
    custom = "custom"
    unknown = "unknown"


class VerificationOutcome(str, Enum):
    """One independent descriptor evidence axis. Integrity, compatibility, functional behavior,
    and hardware support must never be collapsed into a misleading linear level."""

    not_checked = "not_checked"
    passed = "passed"
    failed = "failed"
    partial = "partial"
    not_applicable = "not_applicable"


class CompatibilityStatus(str, Enum):
    compatible = "compatible"
    resize_required = "resize_required"
    incompatible = "incompatible"
    unverified = "unverified"


class TokenizerFormat(str, Enum):
    tokenizers_json = "tokenizers_json"
    sentencepiece = "sentencepiece"
    tiktoken = "tiktoken"
    custom = "custom"
    unknown = "unknown"


class DescriptorFileRole(str, Enum):
    config = "config"
    weights = "weights"
    weight_index = "weight_index"
    tokenizer = "tokenizer"
    tokenizer_config = "tokenizer_config"
    special_tokens = "special_tokens"
    generation_config = "generation_config"
    model_card = "model_card"
    license = "license"
    custom_code = "custom_code"
    other = "other"


class PrecisionMode(str, Enum):
    fp32 = "fp32"
    tf32 = "tf32"
    fp16 = "fp16"
    bf16 = "bf16"
    fp8 = "fp8"
    mixed_bf16 = "mixed_bf16"
    mixed_fp16 = "mixed_fp16"


class QuantizationMode(str, Enum):
    none = "none"
    int8 = "int8"
    int4 = "int4"
    nf4 = "nf4"
    fp4 = "fp4"
    gptq = "gptq"
    awq = "awq"
    hqq = "hqq"


class AdapterMethod(str, Enum):
    none = "none"
    lora = "lora"
    qlora = "qlora"
    dora = "dora"
    ia3 = "ia3"
    full_finetune = "full_finetune"
    prompt_tuning = "prompt_tuning"
    prefix_tuning = "prefix_tuning"


class AttentionImpl(str, Enum):
    """``math``/``eager`` is forced on Blackwell sm_120 — the fused flash/mem-efficient kernels
    deadlock on the first backward (training/environment.py, estimators.py) — at a large activation
    VRAM cost."""

    math = "math"
    eager = "eager"
    sdpa = "sdpa"
    flash_attention_2 = "flash_attention_2"
    flash_attention_3 = "flash_attention_3"
    mem_efficient = "mem_efficient"
    xformers = "xformers"


class LossImpl(str, Enum):
    cross_entropy = "cross_entropy"
    liger_fused_ce = "liger_fused_ce"
    chunked_ce = "chunked_ce"
    dpo = "dpo"
    orpo = "orpo"
    kto = "kto"
    ipo = "ipo"
    reward_bt = "reward_bt"


class Optimizer(str, Enum):
    adamw_torch = "adamw_torch"
    adamw_torch_fused = "adamw_torch_fused"
    adamw_8bit = "adamw_8bit"
    adamw_bnb_8bit = "adamw_bnb_8bit"
    paged_adamw_8bit = "paged_adamw_8bit"
    paged_adamw_32bit = "paged_adamw_32bit"
    adafactor = "adafactor"
    lion = "lion"
    sgd = "sgd"


class CheckpointImpl(str, Enum):
    full_state = "full_state"
    adapter_only = "adapter_only"
    sharded = "sharded"
    distcp = "distcp"
    safetensors = "safetensors"


class OffloadStrategy(str, Enum):
    """The ``controlled_*`` values are the deliberate, planned counterparts of the accidental spills
    in :class:`FitClass`."""

    none = "none"
    controlled_activation_offload = "controlled_activation_offload"
    controlled_optimizer_offload = "controlled_optimizer_offload"
    controlled_parameter_offload = "controlled_parameter_offload"
    cpu_offload = "cpu_offload"
    disk_offload = "disk_offload"
    deepspeed_zero2 = "deepspeed_zero2"
    deepspeed_zero3 = "deepspeed_zero3"


class DependencyLayer(str, Enum):
    """The three dependency layers. The CONTROL PLANE stays lightweight + always installable (opening
    CorpusStudio must never require CUDA/DeepSpeed/an ML framework); CAPABILITY profiles are opt-in
    feature stacks added to the core process with graceful fallback; BACKEND_WORKER environments are
    isolated per-framework runtimes (heavy frameworks pin conflicting torch/CUDA/xformers builds and
    cannot coexist — they talk to the core via the WorkerMessage protocol, never by import)."""

    control_plane = "control_plane"
    capability = "capability"
    backend_worker = "backend_worker"


class EnvironmentState(str, Enum):
    """The lifecycle state of a managed environment. The escalation is deliberate — "installed" is
    NEVER "supported": a package importing (IMPORTABLE) is not proof a kernel runs
    (FUNCTIONAL_PROBE_PASSED), which is not proof the hardware supports it (HARDWARE_VERIFIED). Only
    HARDWARE_VERIFIED earns "supported". The terminal-degraded states record WHY an env is unusable."""

    not_installed = "NOT_INSTALLED"
    installing = "INSTALLING"
    installed_unchecked = "INSTALLED_UNCHECKED"
    importable = "IMPORTABLE"
    dependency_probe_passed = "DEPENDENCY_PROBE_PASSED"
    functional_probe_passed = "FUNCTIONAL_PROBE_PASSED"
    hardware_verified = "HARDWARE_VERIFIED"
    degraded = "DEGRADED"
    incompatible = "INCOMPATIBLE"
    drifted = "DRIFTED"
    broken = "BROKEN"


class RecipeVerification(str, Enum):
    """How far a recipe has been proven — the recipe-level twin of EnvironmentState. A recipe is a
    DECLARATION of what to install; this says whether that declaration has ever produced a working
    environment, and at what level. ``declared`` = we can render the install plan but have not built +
    verified it; higher tiers require actual evidence (a real install / probe / hardware run)."""

    declared = "declared"
    build_verified = "build_verified"
    functional_verified = "functional_verified"
    hardware_verified = "hardware_verified"


class StorageInterface(str, Enum):
    """How a storage device attaches. The interface — not just free space — decides whether a device
    can sustain the heavy sequential + random writes of optimizer/parameter offload and checkpointing.
    A USB bridge or a network mount will thrash under sustained offload even with terabytes free."""

    nvme_pcie = "nvme_pcie"  # internal PCIe NVMe — the only interface fit for sustained offload
    sata_ssd = "sata_ssd"
    hdd = "hdd"  # rotational — random I/O is far too slow for offload
    usb = "usb"  # removable/USB bridge — unfit for sustained offload
    network = "network"  # SMB/NFS/network mount — latency + reliability unfit for offload
    virtual = "virtual"  # ramdisk / overlay / container layer
    unknown = "unknown"


class StorageRole(str, Enum):
    """The role a path plays in a run. Roles differ in access pattern: ``optimizer_offload`` /
    ``parameter_offload`` / ``scratch`` / ``checkpoints`` are WRITE-heavy; ``model_cache`` /
    ``dataset_cache`` are read-LATENCY-sensitive during load; ``source_repo`` / ``python_env`` are
    thousands of SMALL files touched on every process start (an import over a USB bridge or a WSL
    ``/mnt`` mount stalls); ``archive`` just wants capacity. A path's suitability is judged PER ROLE (a
    USB SSD is fine for ``archive``, poor for ``model_cache``, unfit for ``optimizer_offload``)."""

    os = "os"
    source_repo = "source_repo"
    # The Python virtual environment — thousands of small files imported at every process start;
    # over USB or a WSL /mnt mount this thrashes (NTFS translation + latency + small-file overhead).
    python_env = "python_env"
    model_cache = "model_cache"
    dataset_cache = "dataset_cache"
    checkpoints = "checkpoints"
    scratch = "scratch"
    optimizer_offload = "optimizer_offload"
    parameter_offload = "parameter_offload"
    artifacts = "artifacts"
    archive = "archive"
    logs = "logs"


class StorageSuitability(str, Enum):
    """The per-role verdict for a candidate path. ``unsuitable`` is a hard no (data-loss or
    thrash-to-a-halt risk); ``marginal`` will work but degrade (e.g. an HDD for offload); ``unknown``
    when detection couldn't characterize the device (honest, never a false ``suitable``)."""

    suitable = "suitable"
    marginal = "marginal"
    unsuitable = "unsuitable"
    unknown = "unknown"


class AllocatorPolicy(str, Enum):
    default = "default"
    expandable_segments = "expandable_segments"
    max_split_size = "max_split_size"
    garbage_collection = "garbage_collection"


class CompileMode(str, Enum):
    none = "none"
    eager = "eager"
    reduce_overhead = "reduce_overhead"
    max_autotune = "max_autotune"
    aot_inductor = "aot_inductor"


class ExportFormat(str, Enum):
    adapter_peft = "adapter_peft"
    merged_safetensors = "merged_safetensors"
    merged_fp16 = "merged_fp16"
    gguf = "gguf"
    onnx = "onnx"
    awq = "awq"
    gptq = "gptq"
    mlx = "mlx"


class TrainerTarget(str, Enum):
    """Verbatim from config_templates.TrainingConfigTarget."""

    corpus_studio = "corpus_studio"
    axolotl_yaml = "axolotl_yaml"
    trl_config = "trl_config"
    unsloth_script = "unsloth_script"
    huggingface_trainer = "huggingface_trainer"
    llama_factory = "llama_factory"


class StageMarker(str, Enum):
    """Ordered lifecycle stage of a run, launch → export. A RunEvent carries the stage it belongs to
    so a consumer can render a precise progress spine and localize a failure to the exact stage."""

    process_start = "process_start"
    env_loaded = "env_loaded"
    cuda_init = "cuda_init"
    model_loaded = "model_loaded"
    quantized = "quantized"
    adapter_attached = "adapter_attached"
    optimizer_created = "optimizer_created"
    batch_materialized = "batch_materialized"
    forward = "forward"
    loss = "loss"
    backward = "backward"
    optimizer_step = "optimizer_step"
    checkpoint = "checkpoint"
    reload = "reload"
    evaluation = "evaluation"
    export = "export"


class FailureTaxonomy(str, Enum):
    """Terminal outcome category. ``PASS`` is included so the same enum classifies a completed
    probe/run, not only failures. Grounded in the exact hazards the engine documents: the sm_120
    fused-attention deadlock (KERNEL_STALL), the WDDM silent spill (ACCIDENTAL_SPILL vs a clean
    OOM), and env/dependency mismatches (ENVIRONMENT_FAILURE)."""

    PASS = "PASS"
    FAIL = "FAIL"
    OOM = "OOM"
    TIMEOUT = "TIMEOUT"
    KERNEL_STALL = "KERNEL_STALL"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    UNSUPPORTED_CONFIGURATION = "UNSUPPORTED_CONFIGURATION"
    ACCIDENTAL_SPILL = "ACCIDENTAL_SPILL"
    CONTROLLED_OFFLOAD = "CONTROLLED_OFFLOAD"


class FitClass(str, Enum):
    """The fit verdict. ``NATIVE_*`` = fully resident. ``CONTROLLED_*`` = a deliberate, planned
    offload (acceptable, slower). ``ACCIDENTAL_*`` / ``THRASHING`` = an unplanned spill the platform
    did silently (the failure mode the engine warns about). ``FAIL`` = will not run."""

    NATIVE_SAFE = "NATIVE_SAFE"
    NATIVE_TIGHT = "NATIVE_TIGHT"
    NATIVE_UNPROVEN = "NATIVE_UNPROVEN"
    MARGINAL = "MARGINAL"
    CONTROLLED_ACTIVATION_OFFLOAD = "CONTROLLED_ACTIVATION_OFFLOAD"
    CONTROLLED_OPTIMIZER_OFFLOAD = "CONTROLLED_OPTIMIZER_OFFLOAD"
    CONTROLLED_PARAMETER_OFFLOAD = "CONTROLLED_PARAMETER_OFFLOAD"
    ACCIDENTAL_UNIFIED_MEMORY_PAGING = "ACCIDENTAL_UNIFIED_MEMORY_PAGING"
    ACCIDENTAL_WDDM_SPILL = "ACCIDENTAL_WDDM_SPILL"
    THRASHING = "THRASHING"
    FAIL = "FAIL"
