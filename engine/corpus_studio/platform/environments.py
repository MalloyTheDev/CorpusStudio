"""The Environment Manager substrate — recipes + the install-PREVIEW resolver (Phase 2, slice 1).

"Dependency-light" describes the CONTROL PLANE, not the whole product. This module models the
three-layer dependency architecture (see ``docs/IMPLEMENTATION_PLAN.md``):

* the **control plane** — the always-installable core (this engine), no CUDA / ML framework;
* **capability profiles** — opt-in feature stacks added to the core process with graceful fallback;
* **backend worker environments** — ISOLATED per-framework runtimes (heavy frameworks pin conflicting
  torch/CUDA/xformers builds and cannot coexist in one env).

The declarative half is a registry of built-in :class:`EnvironmentRecipe` instances
(grounded in the engine's real optional extras) and a PURE :func:`resolve_dependencies` that renders
the exact, argv-structured (never shell) install plan + a CUDA-aware wheel-index choice + rough
disk/network estimates, for explicit confirmation BEFORE anything is installed. The side-effectful
manager lives in :mod:`corpus_studio.platform.environment_manager`; this module stays torch-free and
mutates nothing.
"""

from __future__ import annotations

import hashlib
import json
import re

from .common import HashRef, Ref
from .contracts import (
    DependencyConflict,
    DependencyRequirement,
    DependencyResolution,
    EnvironmentRecipe,
    ExecutionCapabilityCombination,
    InstallStep,
    PythonRuntime,
    QloraExecutionProbeSpec,
)
from .enums import DependencyLayer, OperatingSystem, RecipeVerification

# The PyTorch wheel indices, by accelerator tag (grounded in the download.pytorch.org layout). torch
# is installed from its OWN index; the rest resolve from PyPI — so a CUDA build is selected without
# breaking every other package (the recommended pytorch.org install pattern).
PYTORCH_INDEX_URLS: dict[str, str] = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "rocm6.2": "https://download.pytorch.org/whl/rocm6.2",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
PYPI_INDEX_URL = "https://pypi.org/simple"
READINESS_V2_RECIPE_ID = "backend-corpus-studio-readiness-v2"
READINESS_FLASH_V1_RECIPE_ID = "backend-corpus-studio-readiness-flash-v1"
READINESS_LIGER_V1_RECIPE_ID = "backend-corpus-studio-readiness-liger-v1"
READINESS_FLASH_LIGER_V1_RECIPE_ID = "backend-corpus-studio-readiness-flash-liger-v1"
READINESS_FLASH_LIGER_PAGED_V1_RECIPE_ID = "backend-corpus-studio-readiness-flash-liger-paged-v1"

# The exact reviewed per-lineage source floor is NOT baked into a recipe or a module constant: it is a
# changing per-amendment value supplied at plan time (env-plan --required-git-ancestor) and sealed into
# the DependencyResolution. Overloading one constant with both "the permanent historical minimum" and
# "this lineage's exact floor" is exactly the conflation this design removes. A separate, broad
# historical-minimum check (if ever needed) must use a distinct name and never be string-equated to an
# exact lineage floor.
_CANONICAL_GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")

# The distributions PyTorch ships from its own index (so they must NOT be requested from PyPI when a
# CUDA build is wanted).
_TORCH_DISTRIBUTIONS: frozenset[str] = frozenset({"torch", "torchvision", "torchaudio"})

# PyTorch's cu128 index currently republishes these torch 2.11 runtime dependencies without
# artifact hashes in the index links. Pip consequently emits ``archive_info: {}`` for them even
# though the wheel install succeeds. Readiness locks must not waive archive identity, so the exact
# known-compatible wheels are installed first, without dependency resolution, from PyPI (whose
# links carry SHA-256 fragments). The resulting argv and source are part of the resolution hash;
# this is not an index fallback or a retry policy.
_HASHED_PYTORCH_BOOTSTRAP_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "torch==2.11.0+cu128": (
        "cuda-pathfinder==1.2.2",
        "setuptools==78.1.0",
        "typing-extensions==4.15.0",
        "jinja2==3.1.6",
        "markupsafe==3.0.3",
    )
}

# Rough, EXPLICITLY-HEURISTIC download sizes in MB. torch dominates and depends on the accelerator, so
# it is special-cased below; everything else uses this table (unknown → a conservative default).
_DOWNLOAD_MB: dict[str, int] = {
    "transformers": 50,
    "peft": 5,
    "trl": 12,
    "accelerate": 15,
    "datasets": 45,
    "bitsandbytes": 120,
    "unsloth": 70,
    "tiktoken": 6,
    "tokenizers": 10,
    "pyarrow": 45,
    "pydantic": 8,
    "typer": 4,
    "deepspeed": 60,
    "xformers": 120,
}
_DEFAULT_PACKAGE_MB = 20
_TORCH_CUDA_MB = 2600  # a CUDA torch wheel is ~2.5-2.8 GB
_TORCH_CPU_MB = 220
_MB = 1_000_000
# Installed on-disk footprint is larger than the compressed download (wheels + unpacked).
_DISK_FOOTPRINT_MULTIPLIER = 2.3


def builtin_recipes() -> list[EnvironmentRecipe]:
    """The built-in environment recipes. Only recipes grounded in the engine's REAL optional extras
    (``[train]`` / ``[parquet]`` / ``[tokenizer]`` / ``[model-tokenizer]``) are declared here; heavier
    backends (DeepSpeed/FSDP/Axolotl/LLaMA-Factory/MoE) arrive with their own backend slices so a
    recipe is never claimed before it can be built + probed."""
    all_os = [OperatingSystem.windows, OperatingSystem.wsl, OperatingSystem.linux, OperatingSystem.macos]
    return [
        EnvironmentRecipe(
            recipe_id="control-plane",
            display_name="Control plane (CorpusStudio core)",
            layer=DependencyLayer.control_plane,
            description="The always-installable dependency-light core - no CUDA, no ML framework. "
            "Opening CorpusStudio requires only this.",
            target="corpus_studio_engine",
            python_requires=">=3.11",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier=">=2", reason="contracts"),
                DependencyRequirement(name="typer", specifier=">=0.12", reason="CLI"),
            ],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
            notes=["Installed as `pip install -e .` (the engine); usually already present."],
        ),
        EnvironmentRecipe(
            recipe_id="capability-tokenization",
            display_name="Exact tokenization (tiktoken)",
            layer=DependencyLayer.capability,
            description="Adds exact GPT-family BPE token counts to the core (else a Unicode heuristic).",
            target="tokenization",
            python_requires=">=3.11",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[DependencyRequirement(name="tiktoken", specifier=">=0.7")],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
        ),
        EnvironmentRecipe(
            recipe_id="capability-model-tokenizer",
            display_name="Model tokenizers (tokenizers)",
            layer=DependencyLayer.capability,
            description="Loads a training target model's own tokenizer for exact per-model budgets.",
            target="model",
            python_requires=">=3.11",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[DependencyRequirement(name="tokenizers", specifier=">=0.15")],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
        ),
        EnvironmentRecipe(
            recipe_id="capability-data",
            display_name="Columnar data (pyarrow)",
            layer=DependencyLayer.capability,
            description="Parquet import/export (else a clear install hint).",
            target="data",
            python_requires=">=3.11",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[DependencyRequirement(name="pyarrow", specifier=">=15")],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
        ),
        EnvironmentRecipe(
            recipe_id="backend-corpus-studio",
            display_name="Backend: CorpusStudio first-party trainer (TRL/PEFT QLoRA)",
            layer=DependencyLayer.backend_worker,
            description="The reference training backend - TRL + PEFT + bitsandbytes 4-bit QLoRA. This "
            "is the current `[train]` extra, promoted to an isolated backend environment.",
            target="corpus_studio",
            python_requires=">=3.11",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier=">=2", reason="worker contracts"),
                DependencyRequirement(name="typer", specifier=">=0.12", reason="worker CLI"),
                DependencyRequirement(name="orjson", specifier=">=3.10", reason="engine runtime"),
                DependencyRequirement(name="torch", specifier=">=2.1"),
                DependencyRequirement(name="transformers", specifier=">=4.44"),
                DependencyRequirement(name="peft", specifier=">=0.11"),
                DependencyRequirement(name="trl", specifier=">=0.9"),
                DependencyRequirement(name="accelerate", specifier=">=0.30"),
                DependencyRequirement(name="datasets", specifier=">=2.19"),
                DependencyRequirement(
                    name="bitsandbytes",
                    specifier=">=0.43; platform_system != 'Darwin'",
                    reason="4-bit QLoRA (CUDA-only; skipped on macOS)",
                ),
            ],
            cuda_index_urls={k: v for k, v in PYTORCH_INDEX_URLS.items()},
            requires_cuda=False,  # the CPU-toy path runs without CUDA; a real GPU run needs a CUDA build
            supported_os=all_os,
            capability_probes=["cuda_available", "bf16_matmul", "bnb_4bit_load", "checkpoint_reload"],
            # Verified on native Windows/WDDM RTX 5070 plus separately labeled WSL probes; this is not
            # native-Linux, offload, or full-sequence 7B proof.
            verification=RecipeVerification.hardware_verified,
            notes=[
                "A default `pip` pulls the CPU torch build; a real GPU run needs the CUDA wheel index.",
                "bitsandbytes is CUDA-only - skipped on macOS (CPU/MPS have no 4-bit path).",
            ],
        ),
        EnvironmentRecipe(
            recipe_id=READINESS_V2_RECIPE_ID,
            display_name="Backend: CorpusStudio readiness v2 (exact QLoRA tuple)",
            layer=DependencyLayer.backend_worker,
            description="Exact-pinned native-Linux CorpusStudio worker environment whose lock is "
            "sealed only after one complete BF16/NF4/double-quant QLoRA math-SDPA tuple passes.",
            target="corpus_studio",
            python_requires=">=3.12",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier="==2.13.4", reason="worker contracts"),
                DependencyRequirement(name="typer", specifier="==0.26.8", reason="worker CLI"),
                DependencyRequirement(name="orjson", specifier="==3.11.9", reason="engine runtime"),
                DependencyRequirement(name="torch", specifier="==2.11.0+cu128"),
                DependencyRequirement(name="transformers", specifier="==5.13.1"),
                DependencyRequirement(name="peft", specifier="==0.19.1"),
                DependencyRequirement(name="trl", specifier="==1.8.0"),
                DependencyRequirement(name="accelerate", specifier="==1.14.0"),
                DependencyRequirement(name="datasets", specifier="==5.0.0"),
                DependencyRequirement(name="bitsandbytes", specifier="==0.49.2"),
                DependencyRequirement(name="safetensors", specifier="==0.8.0"),
                DependencyRequirement(name="tokenizers", specifier="==0.22.2"),
            ],
            cuda_index_urls={"cu128": PYTORCH_INDEX_URLS["cu128"]},
            requires_cuda=True,
            min_compute_capability="12.0",
            supported_os=[OperatingSystem.linux],
            capability_probes=[
                "gpu_responsive",
                "cuda_available",
                "bf16_matmul",
                "bnb_4bit_load",
                "math_sdpa_backward",
                "trainer_contract",
                "cuda_qlora_math_execution",
            ],
            required_execution_probe=QloraExecutionProbeSpec(
                execution_combination=ExecutionCapabilityCombination.model_validate(
                    {
                        "runtime_mode": "training",
                        "device": "cuda",
                        "precision": "bf16",
                        "quantization": "nf4",
                        "adapter_method": "qlora",
                        "attention_impl": "math",
                        "attention_kernel": "torch_sdpa_math",
                        "optimizer": "adamw_torch",
                        "loss_impl": "cross_entropy",
                        "checkpoint_impl": "adapter_only",
                        "export_format": "adapter_peft",
                        "execution_contract_version": "1.0.0",
                        "probe": "cuda_qlora_math_execution",
                    }
                ),
                required_distributions=sorted(
                    [
                        "accelerate",
                        "bitsandbytes",
                        "datasets",
                        "peft",
                        "safetensors",
                        "tokenizers",
                        "torch",
                        "transformers",
                        "trl",
                    ]
                ),
            ),
            requires_worker_wheel=True,
            bootstrap_pip_version="26.1.2",
            verification=RecipeVerification.declared,
            notes=[
                "Plan-only until a separately authorized environment creation.",
                "Every direct dependency is exact-pinned; transitive installs are sealed from pip "
                "reports plus installed RECORD integrity evidence.",
                "HARDWARE_VERIFIED requires the one complete cuda_qlora_math_execution tuple; "
                "independent probe passes cannot be unioned.",
            ],
        ),
        EnvironmentRecipe(
            recipe_id=READINESS_FLASH_V1_RECIPE_ID,
            display_name="Backend: CorpusStudio readiness flash v1 (exact QLoRA flash tuple)",
            layer=DependencyLayer.backend_worker,
            description="Exact-pinned native-Linux CorpusStudio worker environment whose lock is "
            "sealed only after one complete BF16/NF4/double-quant QLoRA forced-flash-SDPA tuple "
            "passes. Independent from the math readiness-v2 baseline.",
            target="corpus_studio",
            python_requires=">=3.12",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier="==2.13.4", reason="worker contracts"),
                DependencyRequirement(name="typer", specifier="==0.26.8", reason="worker CLI"),
                DependencyRequirement(name="orjson", specifier="==3.11.9", reason="engine runtime"),
                DependencyRequirement(name="torch", specifier="==2.11.0+cu128"),
                DependencyRequirement(name="transformers", specifier="==5.13.1"),
                DependencyRequirement(name="peft", specifier="==0.19.1"),
                DependencyRequirement(name="trl", specifier="==1.8.0"),
                DependencyRequirement(name="accelerate", specifier="==1.14.0"),
                DependencyRequirement(name="datasets", specifier="==5.0.0"),
                DependencyRequirement(name="bitsandbytes", specifier="==0.49.2"),
                DependencyRequirement(name="safetensors", specifier="==0.8.0"),
                DependencyRequirement(name="tokenizers", specifier="==0.22.2"),
            ],
            cuda_index_urls={"cu128": PYTORCH_INDEX_URLS["cu128"]},
            requires_cuda=True,
            min_compute_capability="12.0",
            supported_os=[OperatingSystem.linux],
            capability_probes=[
                "gpu_responsive",
                "cuda_available",
                "bf16_matmul",
                "bnb_4bit_load",
                "flash_attn_backward",
                "trainer_contract",
                "cuda_qlora_sdpa_flash_execution",
            ],
            required_execution_probe=QloraExecutionProbeSpec(
                probe="cuda_qlora_sdpa_flash_execution",
                execution_combination=ExecutionCapabilityCombination.model_validate(
                    {
                        "runtime_mode": "training",
                        "device": "cuda",
                        "precision": "bf16",
                        "quantization": "nf4",
                        "adapter_method": "qlora",
                        "attention_impl": "sdpa",
                        "attention_kernel": "torch_sdpa_flash",
                        "optimizer": "adamw_torch",
                        "loss_impl": "cross_entropy",
                        "checkpoint_impl": "adapter_only",
                        "export_format": "adapter_peft",
                        "execution_contract_version": "1.0.0",
                        "probe": "cuda_qlora_sdpa_flash_execution",
                    }
                ),
                flash_sdp_enabled=True,
                math_sdp_enabled=False,
                required_distributions=sorted(
                    [
                        "accelerate",
                        "bitsandbytes",
                        "datasets",
                        "peft",
                        "safetensors",
                        "tokenizers",
                        "torch",
                        "transformers",
                        "trl",
                    ]
                ),
            ),
            requires_worker_wheel=True,
            bootstrap_pip_version="26.1.2",
            verification=RecipeVerification.declared,
            notes=[
                "Plan-only until a separately authorized environment creation.",
                "Requires forced SDPBackend.FLASH_ATTENTION with math and mem-efficient disabled; "
                "automatic SDPA dispatch is not accepted.",
                "Complete tuple forward/backward uses CUDA bf16 autocast (trainer-aligned); "
                "float32 residual after PEFT k-bit prep is not accepted as flash proof.",
                "HARDWARE_VERIFIED requires the one complete cuda_qlora_sdpa_flash_execution tuple; "
                "independent probe passes cannot be unioned.",
                "Does not replace backend-corpus-studio-readiness-v2 (math baseline/rollback).",
                "Linux-only recipe: native Windows/WDDM fused flash SDPA is refused elsewhere; "
                "do not claim flash from a Windows math environment.",
                "This is torch_sdpa_flash identity, not Transformers flash_attention_2 or an "
                "external flash-attn package.",
            ],
        ),
        EnvironmentRecipe(
            recipe_id=READINESS_LIGER_V1_RECIPE_ID,
            display_name="Backend: CorpusStudio readiness liger v1 (exact QLoRA + Liger fused-CE tuple)",
            layer=DependencyLayer.backend_worker,
            description="Exact-pinned native-Linux CorpusStudio worker environment - the math "
            "readiness-v2 stack plus liger-kernel - whose lock is sealed only after one complete "
            "BF16/NF4/double-quant QLoRA math-SDPA tuple with Liger fused-linear-cross-entropy passes. "
            "The fused-linear-CE never materializes the full [seq, vocab] logits: the seq-4096 "
            "logits-wall lever.",
            target="corpus_studio",
            python_requires=">=3.12",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier="==2.13.4", reason="worker contracts"),
                DependencyRequirement(name="typer", specifier="==0.26.8", reason="worker CLI"),
                DependencyRequirement(name="orjson", specifier="==3.11.9", reason="engine runtime"),
                DependencyRequirement(name="torch", specifier="==2.11.0+cu128"),
                DependencyRequirement(name="transformers", specifier="==5.13.1"),
                DependencyRequirement(name="peft", specifier="==0.19.1"),
                DependencyRequirement(name="trl", specifier="==1.8.0"),
                DependencyRequirement(name="accelerate", specifier="==1.14.0"),
                DependencyRequirement(name="datasets", specifier="==5.0.0"),
                DependencyRequirement(name="bitsandbytes", specifier="==0.49.2"),
                DependencyRequirement(name="safetensors", specifier="==0.8.0"),
                DependencyRequirement(name="tokenizers", specifier="==0.22.2"),
                DependencyRequirement(
                    name="liger-kernel",
                    specifier=">=0.3.0",
                    reason="fused-linear-cross-entropy; transformers 5.13.1 use_liger_kernel requires >=0.3.0",
                ),
            ],
            cuda_index_urls={"cu128": PYTORCH_INDEX_URLS["cu128"]},
            requires_cuda=True,
            min_compute_capability="12.0",
            supported_os=[OperatingSystem.linux],
            capability_probes=[
                "gpu_responsive",
                "cuda_available",
                "bf16_matmul",
                "bnb_4bit_load",
                "math_sdpa_backward",
                "trainer_contract",
                "cuda_qlora_liger_execution",
            ],
            required_execution_probe=QloraExecutionProbeSpec(
                probe="cuda_qlora_liger_execution",
                execution_combination=ExecutionCapabilityCombination.model_validate(
                    {
                        "runtime_mode": "training",
                        "device": "cuda",
                        "precision": "bf16",
                        "quantization": "nf4",
                        "adapter_method": "qlora",
                        "attention_impl": "math",
                        "attention_kernel": "torch_sdpa_math",
                        "optimizer": "adamw_torch",
                        "loss_impl": "liger_fused_ce",
                        "checkpoint_impl": "adapter_only",
                        "export_format": "adapter_peft",
                        "execution_contract_version": "1.0.0",
                        "probe": "cuda_qlora_liger_execution",
                    }
                ),
                flash_sdp_enabled=False,
                math_sdp_enabled=True,
                required_distributions=sorted(
                    [
                        "accelerate",
                        "bitsandbytes",
                        "datasets",
                        "liger-kernel",
                        "peft",
                        "safetensors",
                        "tokenizers",
                        "torch",
                        "transformers",
                        "trl",
                    ]
                ),
            ),
            requires_worker_wheel=True,
            bootstrap_pip_version="26.1.2",
            verification=RecipeVerification.declared,
            notes=[
                "Plan-only until a separately authorized environment creation.",
                "The math readiness-v2 exact stack + liger-kernel; the only added semantic is the fused "
                "loss axis (cross_entropy -> liger_fused_ce). Attention stays forced math SDPA.",
                "HARDWARE_VERIFIED requires the one complete cuda_qlora_liger_execution tuple; "
                "independent probe passes cannot be unioned.",
                "Liger applies via the same transformers apply_liger_kernel path the trainer uses; "
                "installed != proven - the probe must load and train the fused kernels on this host.",
                "Does not replace backend-corpus-studio-readiness-v2 (the cross_entropy math baseline).",
            ],
        ),
        EnvironmentRecipe(
            recipe_id=READINESS_FLASH_LIGER_V1_RECIPE_ID,
            display_name="Backend: CorpusStudio readiness flash+liger v1 (forced flash SDPA + Liger fused-CE)",
            layer=DependencyLayer.backend_worker,
            description="Exact-pinned native-Linux CorpusStudio worker environment - the exact pinned "
            "stack plus liger-kernel - sealed only after one complete BF16/NF4/double-quant QLoRA tuple "
            "that FORCES flash SDPA AND applies Liger fused-linear-cross-entropy passes. Flash removes "
            "the forward attention-score matrix and Liger removes the loss-stage logits transient: the "
            "only combination that clears both seq-scaling walls, for the seq-4096 frontier.",
            target="corpus_studio",
            python_requires=">=3.12",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier="==2.13.4", reason="worker contracts"),
                DependencyRequirement(name="typer", specifier="==0.26.8", reason="worker CLI"),
                DependencyRequirement(name="orjson", specifier="==3.11.9", reason="engine runtime"),
                DependencyRequirement(name="torch", specifier="==2.11.0+cu128"),
                DependencyRequirement(name="transformers", specifier="==5.13.1"),
                DependencyRequirement(name="peft", specifier="==0.19.1"),
                DependencyRequirement(name="trl", specifier="==1.8.0"),
                DependencyRequirement(name="accelerate", specifier="==1.14.0"),
                DependencyRequirement(name="datasets", specifier="==5.0.0"),
                DependencyRequirement(name="bitsandbytes", specifier="==0.49.2"),
                DependencyRequirement(name="safetensors", specifier="==0.8.0"),
                DependencyRequirement(name="tokenizers", specifier="==0.22.2"),
                DependencyRequirement(
                    name="liger-kernel",
                    specifier=">=0.3.0",
                    reason="fused-linear-cross-entropy; transformers 5.13.1 use_liger_kernel requires >=0.3.0",
                ),
            ],
            cuda_index_urls={"cu128": PYTORCH_INDEX_URLS["cu128"]},
            requires_cuda=True,
            min_compute_capability="12.0",
            supported_os=[OperatingSystem.linux],
            capability_probes=[
                "gpu_responsive",
                "cuda_available",
                "bf16_matmul",
                "bnb_4bit_load",
                "flash_attn_backward",
                "trainer_contract",
                "cuda_qlora_flash_liger_execution",
            ],
            required_execution_probe=QloraExecutionProbeSpec(
                probe="cuda_qlora_flash_liger_execution",
                execution_combination=ExecutionCapabilityCombination.model_validate(
                    {
                        "runtime_mode": "training",
                        "device": "cuda",
                        "precision": "bf16",
                        "quantization": "nf4",
                        "adapter_method": "qlora",
                        "attention_impl": "sdpa",
                        "attention_kernel": "torch_sdpa_flash",
                        "optimizer": "adamw_torch",
                        "loss_impl": "liger_fused_ce",
                        "checkpoint_impl": "adapter_only",
                        "export_format": "adapter_peft",
                        "execution_contract_version": "1.0.0",
                        "probe": "cuda_qlora_flash_liger_execution",
                    }
                ),
                flash_sdp_enabled=True,
                math_sdp_enabled=False,
                required_distributions=sorted(
                    [
                        "accelerate",
                        "bitsandbytes",
                        "datasets",
                        "liger-kernel",
                        "peft",
                        "safetensors",
                        "tokenizers",
                        "torch",
                        "transformers",
                        "trl",
                    ]
                ),
            ),
            requires_worker_wheel=True,
            bootstrap_pip_version="26.1.2",
            verification=RecipeVerification.declared,
            notes=[
                "Plan-only until a separately authorized environment creation.",
                "Forces SDPBackend.FLASH_ATTENTION (math + mem-efficient disabled) AND applies Liger "
                "fused-linear-CE - the composed seq-4096 frontier tuple.",
                "HARDWARE_VERIFIED requires the one complete cuda_qlora_flash_liger_execution tuple; "
                "independent probe passes cannot be unioned.",
                "torch_sdpa_flash identity, not Transformers flash_attention_2 or an external flash-attn "
                "package. Does not replace the math or flash cross_entropy baselines.",
            ],
        ),
        EnvironmentRecipe(
            recipe_id=READINESS_FLASH_LIGER_PAGED_V1_RECIPE_ID,
            display_name="Backend: CorpusStudio readiness flash+liger+paged v1 (memory-efficient frontier)",
            layer=DependencyLayer.backend_worker,
            description="The fully memory-efficient seq-4096 frontier tuple: forced flash SDPA + Liger "
            "fused-linear-cross-entropy + bitsandbytes paged 8-bit AdamW. Flash removes the forward "
            "attention scores, Liger removes the loss logits, and the paged optimizer keeps the fp32 "
            "LoRA optimizer state out of resident VRAM - the last lever for 7B r16 seq-4096 on 12 GB.",
            target="corpus_studio",
            python_requires=">=3.12",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier="==2.13.4", reason="worker contracts"),
                DependencyRequirement(name="typer", specifier="==0.26.8", reason="worker CLI"),
                DependencyRequirement(name="orjson", specifier="==3.11.9", reason="engine runtime"),
                DependencyRequirement(name="torch", specifier="==2.11.0+cu128"),
                DependencyRequirement(name="transformers", specifier="==5.13.1"),
                DependencyRequirement(name="peft", specifier="==0.19.1"),
                DependencyRequirement(name="trl", specifier="==1.8.0"),
                DependencyRequirement(name="accelerate", specifier="==1.14.0"),
                DependencyRequirement(name="datasets", specifier="==5.0.0"),
                DependencyRequirement(name="bitsandbytes", specifier="==0.49.2"),
                DependencyRequirement(name="safetensors", specifier="==0.8.0"),
                DependencyRequirement(name="tokenizers", specifier="==0.22.2"),
                DependencyRequirement(
                    name="liger-kernel",
                    specifier=">=0.3.0",
                    reason="fused-linear-cross-entropy; transformers 5.13.1 use_liger_kernel requires >=0.3.0",
                ),
            ],
            cuda_index_urls={"cu128": PYTORCH_INDEX_URLS["cu128"]},
            requires_cuda=True,
            min_compute_capability="12.0",
            supported_os=[OperatingSystem.linux],
            capability_probes=[
                "gpu_responsive",
                "cuda_available",
                "bf16_matmul",
                "bnb_4bit_load",
                "flash_attn_backward",
                "trainer_contract",
                "cuda_qlora_flash_liger_paged_execution",
            ],
            required_execution_probe=QloraExecutionProbeSpec(
                probe="cuda_qlora_flash_liger_paged_execution",
                execution_combination=ExecutionCapabilityCombination.model_validate(
                    {
                        "runtime_mode": "training",
                        "device": "cuda",
                        "precision": "bf16",
                        "quantization": "nf4",
                        "adapter_method": "qlora",
                        "attention_impl": "sdpa",
                        "attention_kernel": "torch_sdpa_flash",
                        "optimizer": "paged_adamw_8bit",
                        "loss_impl": "liger_fused_ce",
                        "checkpoint_impl": "adapter_only",
                        "export_format": "adapter_peft",
                        "execution_contract_version": "1.0.0",
                        "probe": "cuda_qlora_flash_liger_paged_execution",
                    }
                ),
                optimizer="paged_adamw_8bit",
                flash_sdp_enabled=True,
                math_sdp_enabled=False,
                required_distributions=sorted(
                    [
                        "accelerate",
                        "bitsandbytes",
                        "datasets",
                        "liger-kernel",
                        "peft",
                        "safetensors",
                        "tokenizers",
                        "torch",
                        "transformers",
                        "trl",
                    ]
                ),
            ),
            requires_worker_wheel=True,
            bootstrap_pip_version="26.1.2",
            verification=RecipeVerification.declared,
            notes=[
                "Plan-only until a separately authorized environment creation.",
                "Forced flash SDPA + Liger fused-linear-CE + paged 8-bit AdamW - the composed "
                "memory-efficient seq-4096 frontier. bnb 8-bit optimizer sm_120 support is proven only "
                "by the one complete cuda_qlora_flash_liger_paged_execution tuple.",
                "HARDWARE_VERIFIED requires that one complete tuple; independent probe passes cannot be "
                "unioned. Does not replace the adamw_torch baselines.",
            ],
        ),
        EnvironmentRecipe(
            recipe_id="backend-unsloth",
            display_name="Backend: Unsloth (accelerated QLoRA)",
            layer=DependencyLayer.backend_worker,
            description="Unsloth's fused QLoRA kernels. CUDA-only; pins its own torch/xformers, so it "
            "MUST be an isolated environment (cannot share the corpus_studio env).",
            target="unsloth",
            python_requires=">=3.11",
            default_index_url=PYPI_INDEX_URL,
            dependency_requirements=[
                DependencyRequirement(name="unsloth", reason="brings its own pinned torch/trl/xformers")
            ],
            cuda_index_urls={
                k: v for k, v in PYTORCH_INDEX_URLS.items() if k.startswith("cu")
            },
            requires_cuda=True,
            min_compute_capability="7.5",
            supported_os=[OperatingSystem.linux, OperatingSystem.wsl, OperatingSystem.windows],
            known_conflicts=[
                DependencyConflict(
                    packages=["unsloth", "torch"],
                    condition="pins specific torch/xformers builds; do not co-install with backend-corpus-studio",
                    severity="block",
                )
            ],
            capability_probes=["cuda_available"],
            # We've verified the ABSENCE guard + Blackwell refusal, not the happy training path (our
            # Blackwell host routes AWAY from Unsloth — needs an Ampere/Ada GPU to functionally verify).
            verification=RecipeVerification.declared,
            notes=["Blackwell/sm_120 forces the math attention path Unsloth does not provide - the "
                   "planner refuses Unsloth there and routes to backend-corpus-studio."],
        ),
    ]


def get_recipe(recipe_id: str) -> EnvironmentRecipe | None:
    """The built-in recipe with this id, or None."""
    return next((r for r in builtin_recipes() if r.recipe_id == recipe_id), None)


def recipes_for_layer(layer: DependencyLayer) -> list[EnvironmentRecipe]:
    """Built-in recipes in one dependency layer."""
    return [r for r in builtin_recipes() if r.layer == layer]


def _parse_min_python(python_requires: str) -> tuple[int, int] | None:
    """Parse a ``>=3.10``-style floor into ``(3, 10)``; None when not a simple floor."""
    spec = python_requires.strip()
    if not spec.startswith(">="):
        return None
    parts = spec[2:].strip().split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


def _python_tuple(python_version: str) -> tuple[int, int] | None:
    parts = python_version.strip().split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


def _requirement_string(req: DependencyRequirement) -> str:
    """Render a requirement as a single pip argv token, e.g. ``transformers>=4.44``. The specifier may
    carry an environment marker (``; platform_system != 'Darwin'``) — pip evaluates it, so it stays in
    the token."""
    return f"{req.name}{req.specifier or ''}"


def _env_python_path(os_value: OperatingSystem) -> str:
    """The interpreter path inside a freshly-created venv, with a placeholder root the creation step
    substitutes. Windows puts it in Scripts\\; POSIX in bin/."""
    if os_value == OperatingSystem.windows:
        return "<ENV_ROOT>\\Scripts\\python.exe"
    return "<ENV_ROOT>/bin/python"


def _estimate_download_mb(requirements: list[DependencyRequirement], accelerator_tag: str) -> int:
    total = 0
    for req in requirements:
        if req.name in _TORCH_DISTRIBUTIONS:
            total += _TORCH_CPU_MB if accelerator_tag == "cpu" else _TORCH_CUDA_MB
        else:
            total += _DOWNLOAD_MB.get(req.name, _DEFAULT_PACKAGE_MB)
    return total


def recipe_digest(recipe: EnvironmentRecipe) -> str:
    """Stable sha256 over a recipe declaration for recipe-drift detection."""
    body = recipe.model_dump(mode="json")
    # Preserve the digest of pre-readiness-v2 recipes so existing managed environments remain valid
    # rollback targets after the additive contract migration.
    if (
        recipe.required_execution_probe is None
        and not recipe.requires_worker_wheel
        and recipe.bootstrap_pip_version is None
    ):
        body.pop("required_execution_probe", None)
        body.pop("requires_worker_wheel", None)
        body.pop("bootstrap_pip_version", None)
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolution_digest(resolution: DependencyResolution) -> str:
    """Stable sha256 over the reviewed resolution, excluding its own seal."""
    body = resolution.model_dump(mode="json", exclude={"resolution_hash"})
    # Additive floor field: only worker-wheel plans carry a per-lineage floor. Pop it when None so a
    # non-worker plan's resolution/confirmation hash is byte-identical to before the field existed.
    if resolution.required_git_ancestor is None:
        body.pop("required_git_ancestor", None)
    # Additive optional worker source commit: pop it when None so any plan that does not review one
    # (every plan before this field existed, and non-worker plans) keeps a byte-identical seal.
    if resolution.worker_source_commit is None:
        body.pop("worker_source_commit", None)
    payload = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _materialize_steps(
    steps: list[InstallStep], *, runtime: PythonRuntime | None, environment_root: str | None
) -> list[InstallStep]:
    """Substitute only the manager-owned placeholders; every other argv token stays literal."""
    if runtime is None and environment_root is None:
        return steps
    replacements = {
        "<BASE_PYTHON>": runtime.executable if runtime is not None else "<BASE_PYTHON>",
        "<CONTROL_PLANE_PYTHON>": runtime.executable
        if runtime is not None
        else "<CONTROL_PLANE_PYTHON>",
        "<ENV_ROOT>": environment_root or "<ENV_ROOT>",
    }
    materialized: list[InstallStep] = []
    for step in steps:
        argv = []
        for token in step.argv:
            for placeholder, value in replacements.items():
                token = token.replace(placeholder, value)
            argv.append(token)
        materialized.append(
            step.model_copy(
                update={
                    "argv": argv,
                    "working_directory": step.working_directory or environment_root,
                    "evidence_path": step.evidence_path.replace(
                        "<ENV_ROOT>", environment_root or "<ENV_ROOT>"
                    )
                    if step.evidence_path
                    else None,
                    "expected_outputs": [
                        output.replace("<ENV_ROOT>", environment_root or "<ENV_ROOT>")
                        for output in step.expected_outputs
                    ],
                }
            )
        )
    return materialized


def resolve_dependencies(
    recipe: EnvironmentRecipe,
    *,
    os_value: OperatingSystem,
    accelerator_tag: str = "cpu",
    python_version: str = "",
    runtime: PythonRuntime | None = None,
    environment_id: str | None = None,
    environment_root: str | None = None,
    manager_version: str = "",
    required_git_ancestor: str | None = None,
    worker_source_commit: str | None = None,
) -> DependencyResolution:
    """Render the argv-structured install PREVIEW for provisioning ``recipe`` on this host — the exact
    steps, the CUDA-aware wheel index, and rough disk/network cost — WITHOUT installing anything.

    Layer-aware: a ``backend_worker`` recipe creates its own isolated venv; a ``capability`` recipe
    installs into the existing control-plane interpreter (it augments the core process). ``resolvable``
    is False (with reasons) when the host can't satisfy the recipe — unmet python floor, unsupported
    OS, or a CUDA-required recipe on a host with no CUDA accelerator.

    ``required_git_ancestor`` is the exact reviewed per-lineage source floor for a worker-wheel plan,
    supplied here (never taken from the recipe or a global constant). A worker-wheel recipe REQUIRES a
    canonical (40-char lowercase-hex) value - omission or a malformed value makes the plan unresolvable;
    a non-worker recipe must not carry one.
    """
    blocking: list[str] = []
    warnings: list[str] = []

    # --- reviewed per-lineage source floor (sealed into the resolution, never from the recipe) ---
    sealed_floor: str | None = None
    if recipe.requires_worker_wheel:
        if required_git_ancestor is None:
            blocking.append(
                "this worker-wheel recipe requires an exact reviewed --required-git-ancestor "
                "(40-char lowercase-hex source floor)"
            )
        elif not _CANONICAL_GIT_SHA1.match(required_git_ancestor):
            blocking.append(
                "--required-git-ancestor must be an exact 40-character lowercase-hex commit"
            )
        else:
            sealed_floor = required_git_ancestor
    elif required_git_ancestor is not None:
        blocking.append("a non-worker recipe does not accept a --required-git-ancestor floor")

    # --- optional reviewed worker source commit (sealed; equality-checked at admission) ---
    sealed_source_commit: str | None = None
    if worker_source_commit is not None:
        if not recipe.requires_worker_wheel:
            blocking.append("a non-worker recipe does not accept a --worker-source-commit")
        elif not _CANONICAL_GIT_SHA1.match(worker_source_commit):
            blocking.append(
                "--worker-source-commit must be an exact 40-character lowercase-hex commit"
            )
        else:
            sealed_source_commit = worker_source_commit

    # --- feasibility ---
    if recipe.supported_os and os_value not in recipe.supported_os:
        supported = ", ".join(o.value for o in recipe.supported_os)
        blocking.append(f"OS '{os_value.value}' is not supported by this recipe (supported: {supported})")

    floor = _parse_min_python(recipe.python_requires)
    have = _python_tuple(python_version)
    if floor is not None and have is not None and have < floor:
        blocking.append(
            f"Python {have[0]}.{have[1]} is below the recipe floor {recipe.python_requires}"
        )

    if recipe.requires_cuda and accelerator_tag == "cpu":
        blocking.append("recipe requires a CUDA accelerator, but none was detected on this host")

    if recipe.requires_native_build:
        warnings.append("needs a native build - a C/C++ compiler toolchain must be present")

    # --- accelerator / index selection ---
    has_torch = any(r.name in _TORCH_DISTRIBUTIONS for r in recipe.dependency_requirements)
    resolved_indexes: list[str] = []
    torch_index: str | None = None
    if has_torch:
        if accelerator_tag in recipe.cuda_index_urls:
            torch_index = recipe.cuda_index_urls[accelerator_tag]
        elif "cpu" in recipe.cuda_index_urls:
            torch_index = recipe.cuda_index_urls["cpu"]
        if torch_index:
            resolved_indexes.append(torch_index)
        else:
            # Fail closed: a torch-bearing recipe with no wheel index for this host's accelerator (and
            # no 'cpu' fallback) must NOT report itself resolvable. _build_install_steps gates the torch
            # install step on torch_index, so a missing index would silently omit torch while still
            # installing transformers/peft/trl from PyPI (with deps), pulling an unpinned transitive
            # torch instead of the recipe's exact pin - breaking the sealing guarantee.
            blocking.append(
                f"no PyTorch wheel index for accelerator '{accelerator_tag}' in this recipe "
                f"(available: {sorted(recipe.cuda_index_urls)})"
            )
        if accelerator_tag == "cpu":
            warnings.append("no CUDA selected - installing the CPU PyTorch build (no GPU training)")
    if recipe.dependency_requirements:
        package_index = recipe.default_index_url or PYPI_INDEX_URL
        if package_index not in resolved_indexes:
            resolved_indexes.append(package_index)

    # --- macOS bitsandbytes caveat (its env marker skips it, but say so) ---
    if os_value == OperatingSystem.macos and any(
        "Darwin" in (r.specifier or "") for r in recipe.dependency_requirements
    ):
        warnings.append("bitsandbytes (4-bit QLoRA) is skipped on macOS - no CPU/MPS 4-bit path")

    # --- render the argv steps (never a shell string) ---
    steps = _materialize_steps(
        _build_install_steps(
            recipe,
            os_value=os_value,
            torch_index=torch_index,
            package_index=recipe.default_index_url or PYPI_INDEX_URL,
        ),
        runtime=runtime,
        environment_root=environment_root,
    )

    download_mb = _estimate_download_mb(recipe.dependency_requirements, accelerator_tag)
    disk_mb = int(download_mb * _DISK_FOOTPRINT_MULTIPLIER)

    resolution = DependencyResolution(
        recipe_ref=Ref(id=recipe.recipe_id, hash=HashRef(value=recipe_digest(recipe))),
        environment_ref=Ref(id=environment_id) if environment_id else None,
        runtime=runtime,
        environment_root=environment_root,
        manager_version=manager_version,
        python_version=python_version or (runtime.version if runtime else ""),
        os=os_value,
        accelerator_tag=accelerator_tag,
        resolved_index_urls=resolved_indexes,
        install_steps=steps,
        required_execution_probe=recipe.required_execution_probe,
        # Seal the exact reviewed per-lineage floor supplied at plan time (never from the recipe), so it
        # is bound into the resolution/confirmation hash and reaches env-create through the confirmed
        # plan. None (and popped from the digest) for non-worker plans.
        required_git_ancestor=sealed_floor,
        # Seal the optional reviewed worker source commit (worker plans only); None (and popped from the
        # digest) otherwise, so admission can equality-check the wheel's embedded source_commit.
        worker_source_commit=sealed_source_commit,
        estimated_download_bytes=download_mb * _MB,
        estimated_disk_bytes=disk_mb * _MB,
        resolvable=not blocking,
        blocking_reasons=blocking,
        warnings=warnings,
    )
    return resolution.model_copy(update={"resolution_hash": resolution_digest(resolution)})


def _build_install_steps(
    recipe: EnvironmentRecipe,
    *,
    os_value: OperatingSystem,
    torch_index: str | None,
    package_index: str,
) -> list[InstallStep]:
    """Build the ordered, argv-structured install steps for a recipe. Imported lazily-free (returns a
    list of InstallStep). Backend workers get an isolated venv; capability profiles install into the
    control-plane interpreter."""
    reqs = recipe.dependency_requirements
    torch_reqs = [r for r in reqs if r.name in _TORCH_DISTRIBUTIONS]
    other_reqs = [r for r in reqs if r.name not in _TORCH_DISTRIBUTIONS]
    torch_bootstrap_reqs = tuple(
        requirement
        for torch_requirement in torch_reqs
        for requirement in _HASHED_PYTORCH_BOOTSTRAP_REQUIREMENTS.get(
            _requirement_string(torch_requirement), ()
        )
    )

    if recipe.layer == DependencyLayer.backend_worker:
        env_py = _env_python_path(os_value)
        evidence_root = "<ENV_ROOT>/.corpusstudio-install-evidence"
        pip_requirement = (
            f"pip=={recipe.bootstrap_pip_version}"
            if recipe.bootstrap_pip_version
            else "pip"
        )
        steps = [
            InstallStep(
                phase="create_venv",
                description="Create an isolated virtual environment for this backend",
                argv=["<BASE_PYTHON>", "-m", "venv", "<ENV_ROOT>"],
                timeout_seconds=300,
                expected_outputs=[_env_python_path(os_value)],
            ),
            InstallStep(
                phase="upgrade_pip",
                description="Upgrade pip in the new environment",
                argv=[
                    env_py,
                    "-m",
                    "pip",
                    "--isolated",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--index-url",
                    package_index,
                    "--report",
                    f"{evidence_root}/upgrade-pip.json",
                    "--upgrade",
                    pip_requirement,
                ],
                environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
                timeout_seconds=900,
                network_required=True,
                evidence_path=f"{evidence_root}/upgrade-pip.json",
                configured_index_urls=[package_index],
            ),
        ]
        if torch_bootstrap_reqs:
            steps.append(
                InstallStep(
                    phase="install",
                    description="Install hash-backed PyTorch prerequisites from PyPI",
                    argv=[
                        env_py,
                        "-m",
                        "pip",
                        "--isolated",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--no-deps",
                        "--index-url",
                        package_index,
                        "--report",
                        f"{evidence_root}/install-pytorch-prerequisites.json",
                    ]
                    + list(torch_bootstrap_reqs),
                    environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
                    timeout_seconds=900,
                    network_required=True,
                    evidence_path=f"{evidence_root}/install-pytorch-prerequisites.json",
                    configured_index_urls=[package_index],
                )
            )
        if torch_reqs and torch_index:
            steps.append(
                InstallStep(
                    phase="install",
                    description="Install PyTorch from its accelerator-specific wheel index",
                    argv=[env_py, "-m", "pip", "--isolated", "install", "--index-url", torch_index]
                    + ["--disable-pip-version-check", "--no-input"]
                    + ["--report", f"{evidence_root}/install-torch.json"]
                    + [_requirement_string(r) for r in torch_reqs],
                    environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
                    timeout_seconds=3600,
                    network_required=True,
                    native_build_expected=recipe.requires_native_build,
                    evidence_path=f"{evidence_root}/install-torch.json",
                    configured_index_urls=[torch_index],
                )
            )
        if other_reqs:
            steps.append(
                InstallStep(
                    phase="install",
                    description="Install the remaining dependencies from PyPI",
                    argv=[
                        env_py,
                        "-m",
                        "pip",
                        "--isolated",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--index-url",
                        package_index,
                        "--report",
                        f"{evidence_root}/install-dependencies.json",
                    ]
                    + [_requirement_string(r) for r in other_reqs],
                    environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
                    timeout_seconds=3600,
                    network_required=True,
                    native_build_expected=recipe.requires_native_build,
                    evidence_path=f"{evidence_root}/install-dependencies.json",
                    configured_index_urls=[package_index],
                )
            )
        return steps

    # capability / control_plane: augment the existing control-plane interpreter (no new venv).
    all_reqs = torch_reqs + other_reqs
    return [
        InstallStep(
            phase="install",
            description="Install into the control-plane environment (augments the core process)",
            argv=[
                "<CONTROL_PLANE_PYTHON>",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--index-url",
                package_index,
                "--report",
                "<ENV_ROOT>/.corpusstudio-install-evidence/install-capability.json",
            ]
            + [_requirement_string(r) for r in all_reqs],
            environment={"PIP_NO_INPUT": "1", "PYTHONUTF8": "1"},
            timeout_seconds=3600,
            network_required=True,
            native_build_expected=recipe.requires_native_build,
            evidence_path="<ENV_ROOT>/.corpusstudio-install-evidence/install-capability.json",
            configured_index_urls=[package_index],
        )
    ]


def select_accelerator_tag(
    cuda_runtime_version: str | None, compute_capability_major: int | None, has_gpu: bool
) -> str:
    """Pick the PyTorch wheel tag for a host. Prefers the installed CUDA runtime version; falls back to
    the GPU's compute-capability major; ``cpu`` when there is no GPU. Returns a key of
    :data:`PYTORCH_INDEX_URLS`."""
    if not has_gpu:
        return "cpu"
    if cuda_runtime_version:
        major_minor = ".".join(cuda_runtime_version.split(".")[:2])
        by_runtime = {"12.8": "cu128", "12.6": "cu126", "12.4": "cu126", "12.1": "cu121", "11.8": "cu118"}
        if major_minor in by_runtime:
            return by_runtime[major_minor]
    if compute_capability_major is not None:
        if compute_capability_major >= 12:  # Blackwell
            return "cu128"
        if compute_capability_major >= 8:  # Ampere / Ada / Hopper
            return "cu121"
        return "cu118"
    return "cu121"  # a reasonable default for an unknown CUDA GPU
