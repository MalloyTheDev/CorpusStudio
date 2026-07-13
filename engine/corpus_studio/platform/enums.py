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
    """The role a path plays in a run. Roles differ in write intensity + durability needs: ``os`` and
    ``source_repo`` want reliability; ``optimizer_offload`` / ``parameter_offload`` / ``scratch`` want
    sustained high-throughput internal storage; ``archive`` just wants capacity. A path's suitability
    is judged PER ROLE (a USB drive is fine for ``archive``, unfit for ``optimizer_offload``)."""

    os = "os"
    source_repo = "source_repo"
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
