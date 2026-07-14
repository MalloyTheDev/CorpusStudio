"""Functional capability probes — readiness means a kernel actually RAN, not "the package imports".

Each probe executes a tiny real operation (a bf16 matmul, a 4-bit load, a flash-attention backward, a
checkpoint round-trip) and returns a :class:`~corpus_studio.platform.enums.FailureTaxonomy` outcome. A
probe that PASSES contributes to the ``effective_capabilities`` (what actually works on THIS host),
which is what the planner should resolve a RunPlan against — not a backend's static claims.

Dependency-light: this module imports NO torch at load time. Every torch/bitsandbytes import is lazy,
inside a probe body, so a core-only install still runs the framework and reports each hardware probe as
``ENVIRONMENT_FAILURE`` (→ ``readiness = not_ready``) instead of crashing.

The one probe that must not actually execute on **native-Windows** Blackwell sm_120 is
``flash_attn_backward``: the fused flash SDPA backward deadlocks on the first backward under the
Windows WDDM driver (documented in ``training/environment.py``), so there it short-circuits to
``KERNEL_STALL`` rather than hanging the probe process. Outside native Windows, the known WDDM refusal
does not apply, so the probe executes and must itself PASS before it proves flash/sdpa. The native-
Linux RTX 5070 has passing evidence for the bounded forced-flash readiness tuple only; the first real
0.5B smoke stopped before adapter insertion and no real optimizer step or sequence 4096 is proved.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from corpus_studio.training.quantization import find_linear4bit_modules

from .common import PackageLock, Ref
from .contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvironmentProfile,
    ExecutionCapabilityCombination,
    ProbeResult,
)
from .enums import (
    AdapterMethod,
    AttentionImpl,
    AttentionKernel,
    CheckpointImpl,
    CommunicationBackend,
    FailureTaxonomy,
    MemoryTier,
    LossImpl,
    OffloadStrategy,
    Optimizer,
    ParallelismKind,
    PlacementMode,
    PrecisionMode,
    QuantizationMode,
)
from .gpu_health import classify_gpu_health, probe_gpu_responsive, wedged_gpu_remediation
from .host_platform import flash_sdpa_deadlocks

_TX = FailureTaxonomy

# training-stack distributions that gate readiness (a subset of the profile's package list).
_TRAIN_PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")


@dataclass
class ProbeOutcome:
    """A probe's result. ``proves`` maps a capability axis (precision/quantization/attention/adapter)
    to the concrete tokens this probe demonstrated when it PASSED — the input to
    ``effective_capabilities``."""

    taxonomy: FailureTaxonomy
    detail: str | None = None
    measured: dict = field(default_factory=dict)
    proves: dict[str, list[str]] = field(default_factory=dict)
    execution_combinations: list[ExecutionCapabilityCombination] = field(default_factory=list)


ProbeFn = Callable[[EnvironmentProfile], ProbeOutcome]


def _max_cc_major(profile: EnvironmentProfile) -> int:
    return max((g.compute_capability_major or 0 for g in profile.gpus), default=0)


# --- built-in probes ------------------------------------------------------------------------------


def _probe_cuda_available(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    try:
        if torch.cuda.is_available():
            return ProbeOutcome(_TX.PASS, f"{torch.cuda.device_count()} CUDA device(s)")
        return ProbeOutcome(_TX.FAIL, "torch present but no CUDA device (CPU build or no GPU)")
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_bf16_matmul(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
        b = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
        finite = bool(torch.isfinite(a @ b).all().item())
        if finite:
            return ProbeOutcome(
                _TX.PASS, f"bf16 matmul on {device}", proves={"precision": ["bf16"]}
            )
        return ProbeOutcome(_TX.NUMERICAL_FAILURE, "bf16 matmul produced non-finite values")
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_bnb_4bit_load(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import bitsandbytes  # noqa: F401,PLC0415
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"bitsandbytes/torch not importable: {exc}")
    try:
        if not torch.cuda.is_available():
            return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, "bitsandbytes 4-bit requires CUDA")
        from bitsandbytes.nn import Linear4bit  # noqa: PLC0415

        layer = Linear4bit(
            16,
            16,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            quant_storage=torch.uint8,
        ).to("cuda")
        inputs = torch.randn(
            2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out = layer(inputs)
        out.float().sum().backward()
        finite = bool(torch.isfinite(out).all().item()) and inputs.grad is not None
        quant_type = getattr(getattr(layer.weight, "quant_state", None), "quant_type", None)
        if finite and quant_type == "nf4":
            return ProbeOutcome(
                _TX.PASS,
                "NF4 Linear4bit BF16 forward+backward ok",
                proves={"quantization": ["nf4"]},
            )
        return ProbeOutcome(
            _TX.NUMERICAL_FAILURE,
            "NF4 Linear4bit did not preserve NF4 identity or produced an invalid gradient",
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_flash_attn_backward(profile: EnvironmentProfile) -> ProbeOutcome:
    # Known-hazard short-circuit: the fused flash SDPA backward deadlocks on Blackwell sm_120 ONLY
    # under the native-Windows WDDM driver. Report it WITHOUT executing THERE so the probe never hangs.
    # Outside native Windows the known WDDM refusal does not apply, so the probe executes. Only its
    # PASS result proves flash/sdpa on that exact host. WSL evidence does not prove bare-Linux behavior.
    if flash_sdpa_deadlocks(profile.host.os, _max_cc_major(profile)):
        return ProbeOutcome(
            _TX.KERNEL_STALL,
            "native Windows + sm_120 (Blackwell): the fused flash SDPA backward deadlocks under the "
            "Windows WDDM driver — not executed to avoid hanging the probe; use math/eager SDPA, or "
            "use a non-WDDM host only after its flash capability probe passes.",
        )
    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as functional  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    if not torch.cuda.is_available():
        return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, "no CUDA GPU for a flash-attention probe")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

        q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = functional.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
        return ProbeOutcome(
            _TX.PASS,
            "flash SDPA forward+backward ok",
            proves={
                "attention": ["sdpa"],
                "attention_kernel": ["torch_sdpa_flash"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_math_sdpa_backward(profile: EnvironmentProfile) -> ProbeOutcome:
    """Force the PyTorch math SDPA kernel through a tiny forward/backward."""

    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as functional  # noqa: PLC0415
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch SDPA not importable: {exc}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        q = torch.randn(1, 2, 8, 16, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(1, 2, 8, 16, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(1, 2, 8, 16, device=device, dtype=dtype, requires_grad=True)
        with sdpa_kernel([SDPBackend.MATH]):
            out = functional.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
        return ProbeOutcome(
            _TX.PASS,
            f"math SDPA forward+backward ok on {device}",
            proves={
                "attention": ["math", "sdpa"],
                "attention_kernel": ["torch_sdpa_math"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_dense_optimizer_step(profile: EnvironmentProfile) -> ProbeOutcome:
    """Run the exact reference loss/optimizer path; field presence alone is not capability proof."""

    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as functional  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch optimizer path unavailable: {exc}")
    try:
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3, bias=False, device="cpu", dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.weight.detach().clone()
        logits = model(torch.ones(2, 4, dtype=torch.float32))
        loss = functional.cross_entropy(logits, torch.tensor([0, 2]))
        loss.backward()
        optimizer.step()
        finite = bool(torch.isfinite(loss).item()) and bool(torch.isfinite(model.weight).all().item())
        changed = not bool(torch.equal(before, model.weight.detach()))
        if not finite or not changed:
            return ProbeOutcome(
                _TX.NUMERICAL_FAILURE,
                "reference cross-entropy/AdamW step was non-finite or made no update",
            )
        return ProbeOutcome(
            _TX.PASS,
            "FP32 cross-entropy forward/backward and AdamW update ok",
            proves={
                "loss": ["cross_entropy"],
                "optimizer": ["adamw_torch"],
                "precision": ["fp32"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_trainer_contract(profile: EnvironmentProfile) -> ProbeOutcome:
    """Capture the exact TRL field surface without silently filtering semantic arguments."""

    try:
        import dataclasses  # noqa: PLC0415
        import inspect  # noqa: PLC0415

        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"TRL trainer surface unavailable: {exc}")
    try:
        fields = sorted(item.name for item in dataclasses.fields(SFTConfig))
        init_fields = sorted(inspect.signature(SFTTrainer.__init__).parameters)
        if not ({"max_length", "max_seq_length"} & set(fields)):
            return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, "SFTConfig has no sequence field")
        if not ({"processing_class", "tokenizer"} & set(init_fields)):
            return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, "SFTTrainer has no tokenizer input")
        baseline = {
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "data_seed",
            "dataset_text_field",
            "disable_tqdm",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "learning_rate",
            "logging_steps",
            "lr_scheduler_type",
            "num_train_epochs",
            "max_grad_norm",
            "optim",
            "output_dir",
            "packing",
            "per_device_train_batch_size",
            "report_to",
            "save_steps",
            "save_strategy",
            "save_total_limit",
            "seed",
        }
        missing = sorted(baseline - set(fields))
        if missing:
            return ProbeOutcome(
                _TX.UNSUPPORTED_CONFIGURATION,
                "SFTConfig is missing required fields: " + ", ".join(missing),
            )
        return ProbeOutcome(
            _TX.PASS,
            "TRL trainer field contract captured",
            measured={"sft_config_fields": fields, "sft_trainer_init_fields": init_fields},
            proves={
                "trainer_field": fields,
                "trainer_init_field": init_fields,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_checkpoint_reload(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    try:
        tensor = torch.randn(4, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            torch.save({"w": tensor}, path)
            # weights_only=True: defensive default even though this file is the probe's own tensor.
            loaded = torch.load(path, map_location="cpu", weights_only=True)["w"]
        if bool(torch.equal(tensor, loaded)):
            return ProbeOutcome(
                _TX.PASS,
                "checkpoint save/reload round-trip ok",
                proves={"checkpoint": ["adapter_only"]},
            )
        return ProbeOutcome(_TX.CHECKPOINT_FAILURE, "reloaded tensor differs from saved")
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.CHECKPOINT_FAILURE, str(exc))


def _combination(
    *,
    runtime_mode: str,
    device: str,
    precision: str,
    quantization: str,
    adapter_method: str,
    attention_impl: str,
    attention_kernel: str,
    probe: str,
) -> ExecutionCapabilityCombination:
    return ExecutionCapabilityCombination.model_validate(
        {
            "runtime_mode": runtime_mode,
            "device": device,
            "precision": precision,
            "quantization": quantization,
            "adapter_method": adapter_method,
            "attention_impl": attention_impl,
            "attention_kernel": attention_kernel,
            "optimizer": "adamw_torch",
            "loss_impl": "cross_entropy",
            "checkpoint_impl": "adapter_only",
            "export_format": "adapter_peft",
            "execution_contract_version": "1.0.0",
            "probe": probe,
        }
    )


# These bounded execution probes are exercised only by managed worker environments.  The core CI
# environment intentionally omits torch/Transformers/PEFT/bitsandbytes and has no CUDA device, so
# their bodies are integration-test territory; the report/evidence plumbing remains unit-covered.
def _tiny_llama_config() -> Any:  # pragma: no cover - optional training-stack integration
    from transformers import LlamaConfig  # noqa: PLC0415

    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _probe_cpu_lora_execution(
    profile: EnvironmentProfile,
) -> ProbeOutcome:  # pragma: no cover - optional training-stack integration
    """Run one complete FP32 LoRA tuple, including an adapter save/reload."""

    try:
        import tempfile  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415
        from transformers import LlamaForCausalLM  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"CPU LoRA probe unavailable: {exc}")
    try:
        torch.manual_seed(0)
        config = _tiny_llama_config()
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).to(device="cpu", dtype=torch.float32)
        model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable or any(
            parameter.device.type != "cpu" or parameter.dtype != torch.float32
            for parameter in trainable
        ):
            return ProbeOutcome(
                _TX.UNSUPPORTED_CONFIGURATION,
                "CPU LoRA adapter placement or FP32 master-weight policy was not enforced",
            )
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        before = trainable[0].detach().clone()
        input_ids = torch.randint(3, 64, (2, 8), dtype=torch.long)
        output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not gradients or any(gradient.dtype != torch.float32 for gradient in gradients):
            return ProbeOutcome(_TX.NUMERICAL_FAILURE, "CPU LoRA gradients were not FP32")
        optimizer.step()
        if not bool(torch.isfinite(output.loss).item()) or torch.equal(before, trainable[0]):
            return ProbeOutcome(
                _TX.NUMERICAL_FAILURE,
                "CPU LoRA cross-entropy/AdamW step was non-finite or made no update",
            )
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp, safe_serialization=True)
            from pathlib import Path  # noqa: PLC0415

            if not (Path(tmp) / "adapter_model.safetensors").is_file():
                return ProbeOutcome(_TX.CHECKPOINT_FAILURE, "adapter safetensors was not written")
            fresh_config = _tiny_llama_config()
            fresh_config._attn_implementation = "eager"
            fresh = LlamaForCausalLM(fresh_config).to(device="cpu", dtype=torch.float32)
            reloaded = PeftModel.from_pretrained(fresh, tmp)
            with torch.no_grad():
                reload_loss = reloaded(input_ids=input_ids, labels=input_ids).loss
        if not bool(torch.isfinite(reload_loss).item()):
            return ProbeOutcome(_TX.CHECKPOINT_FAILURE, "reloaded CPU LoRA adapter was non-finite")
        combination = _combination(
            runtime_mode="cpu_toy",
            device="cpu",
            precision="fp32",
            quantization="none",
            adapter_method="lora",
            attention_impl="eager",
            attention_kernel="eager",
            probe="cpu_lora_execution",
        )
        return ProbeOutcome(
            _TX.PASS,
            "FP32 eager LoRA forward/backward, AdamW update, and safetensors reload passed on CPU",
            measured={"loss": float(output.loss.detach().item())},
            proves={
                "precision": ["fp32"],
                "attention": ["eager"],
                "attention_kernel": ["eager"],
                "adapter": ["lora"],
                "loss": ["cross_entropy"],
                "optimizer": ["adamw_torch"],
                "checkpoint": ["adapter_only"],
            },
            execution_combinations=[combination],
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, f"CPU LoRA execution tuple failed: {exc}")


def _probe_cuda_qlora_math_execution(
    profile: EnvironmentProfile,
) -> ProbeOutcome:  # pragma: no cover - requires a real CUDA worker environment
    """Run one bounded BF16/NF4/QLoRA tuple with only math SDPA enabled."""

    return _probe_cuda_qlora_sdpa_execution_tuple(
        profile,
        probe_name="cuda_qlora_math_execution",
        attention_impl="math",
        attention_kernel="torch_sdpa_math",
        sdp_backend_name="MATH",
        enable_flash=False,
        enable_math=True,
        pass_detail=(
            "BF16/NF4/QLoRA math-SDPA backward, AdamW update, and adapter reload passed"
        ),
        proves_attention=["math", "sdpa"],
        record_phase_timing=False,
    )


def _probe_cuda_qlora_sdpa_flash_execution(
    profile: EnvironmentProfile,
) -> ProbeOutcome:  # pragma: no cover - requires a real CUDA worker environment
    """Run one bounded BF16/NF4/QLoRA tuple forced onto torch_sdpa_flash only."""

    return _probe_cuda_qlora_sdpa_execution_tuple(
        profile,
        probe_name="cuda_qlora_sdpa_flash_execution",
        attention_impl="sdpa",
        attention_kernel="torch_sdpa_flash",
        sdp_backend_name="FLASH_ATTENTION",
        enable_flash=True,
        enable_math=False,
        pass_detail=(
            "BF16/NF4/QLoRA forced-flash-SDPA backward, AdamW update, and adapter reload passed"
        ),
        proves_attention=["sdpa"],
        record_phase_timing=True,
    )


def _restore_qlora_probe_process_state(
    torch_module: Any,
    *,
    previous_toggles: tuple[bool, bool, bool],
    cpu_rng_state: Any | None,
    cuda_rng_states: Any | None,
    sampler_stop: Any | None,
    sampler_thread: Any | None,
) -> None:
    """Restore process-global probe state and fail if a sampler descendant remains alive."""

    failures: list[str] = []
    if sampler_stop is not None:
        sampler_stop.set()
    if sampler_thread is not None:
        sampler_thread.join(timeout=3)
        if sampler_thread.is_alive():
            failures.append("resource sampler did not terminate")
    restorations = (
        ("flash SDPA", torch_module.backends.cuda.enable_flash_sdp, previous_toggles[0]),
        (
            "memory-efficient SDPA",
            torch_module.backends.cuda.enable_mem_efficient_sdp,
            previous_toggles[1],
        ),
        ("math SDPA", torch_module.backends.cuda.enable_math_sdp, previous_toggles[2]),
    )
    for label, restore, value in restorations:
        try:
            restore(value)
        except Exception:  # noqa: BLE001 - attempt every independent restoration.
            failures.append(f"{label} setting was not restored")
    if cpu_rng_state is not None:
        try:
            torch_module.random.set_rng_state(cpu_rng_state)
        except Exception:  # noqa: BLE001
            failures.append("CPU RNG state was not restored")
    if cuda_rng_states is not None:
        try:
            torch_module.cuda.set_rng_state_all(cuda_rng_states)
        except Exception:  # noqa: BLE001
            failures.append("CUDA RNG state was not restored")
    if failures:
        raise RuntimeError("QLoRA probe cleanup failed: " + "; ".join(failures))


def _probe_cuda_qlora_sdpa_execution_tuple(
    profile: EnvironmentProfile,
    *,
    probe_name: str,
    attention_impl: str,
    attention_kernel: str,
    sdp_backend_name: str,
    enable_flash: bool,
    enable_math: bool,
    pass_detail: str,
    proves_attention: list[str],
    record_phase_timing: bool,
) -> ProbeOutcome:  # pragma: no cover - requires a real CUDA worker environment
    """Shared complete QLoRA tuple used by independent math and flash readiness probes.

    Uses the same tiny model, seed, batch, sequence length, LoRA rank, and optimizer so math and
    flash measurements remain comparable. Never falls back to another SDPA backend.

    Forward/backward run under CUDA bf16 autocast so attention Q/K/V match the first-party trainer
    (TRL SFTConfig bf16) and forced flash SDPA (which rejects float32). PEFT k-bit prep still keeps
    non-4bit master weights and LoRA grads in FP32; those policies are checked outside autocast.
    """

    del profile  # host refusal is decided by the managed environment recipe/OS, not this body
    try:
        import importlib.metadata as metadata  # noqa: PLC0415
        import os  # noqa: PLC0415
        import platform  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        import threading  # noqa: PLC0415
        import time  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import accelerate  # noqa: F401,PLC0415
        import datasets  # noqa: F401,PLC0415
        import safetensors  # noqa: F401,PLC0415
        import tokenizers  # noqa: F401,PLC0415
        import torch  # noqa: PLC0415
        import trl  # noqa: F401,PLC0415
        from bitsandbytes.nn import Linear4bit as BnbLinear4bit  # noqa: PLC0415
        from corpus_studio.training.trainer import (  # noqa: PLC0415
            _buffer_devices,
            _offload_hook_deviations,
            _parameter_devices,
        )
        from peft import (  # noqa: PLC0415
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            LlamaForCausalLM,
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"CUDA QLoRA probe unavailable: {exc}")
    if not torch.cuda.is_available():
        return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, "no CUDA GPU for the QLoRA tuple probe")
    try:
        sdp_backend = getattr(SDPBackend, sdp_backend_name)
    except AttributeError:
        return ProbeOutcome(
            _TX.UNSUPPORTED_CONFIGURATION,
            f"PyTorch SDPBackend.{sdp_backend_name} is unavailable on this stack",
        )
    previous = (
        bool(torch.backends.cuda.flash_sdp_enabled()),
        bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        bool(torch.backends.cuda.math_sdp_enabled()),
    )
    previous_toggles = {
        "flash_sdp_enabled": previous[0],
        "memory_efficient_sdp_enabled": previous[1],
        "math_sdp_enabled": previous[2],
    }
    sampler_stop = None
    sampler_thread = None
    samples: dict[str, Any] = {}
    measured_started: float | None = None
    phase_timings: dict[str, float] = {}
    thermal: dict[str, float | None] = {
        "gpu_temperature_celsius": None,
        "gpu_power_watts": None,
    }
    baseline_gpu_allocated = 0
    baseline_gpu_reserved = 0
    baseline_host_rss = 0
    baseline_nvidia: int | None = None
    cpu_rng_state = None
    cuda_rng_states = None
    try:
        # nvidia-smi uses physical ordinals; CUDA device 0 may be remapped by CUDA_VISIBLE_DEVICES.
        nvidia_gpu_id: str | None = str(torch.cuda._get_nvml_device_index(0))
    except Exception:  # noqa: BLE001 - unavailable mapping makes nvidia-smi evidence unavailable.
        nvidia_gpu_id = None

    def _rss_bytes() -> int:
        try:
            import psutil  # noqa: PLC0415

            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:  # noqa: BLE001
            try:
                pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
                return pages * int(os.sysconf("SC_PAGE_SIZE"))
            except Exception:  # noqa: BLE001
                return 0

    def _nvidia_process_bytes() -> int | None:
        if nvidia_gpu_id is None:
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={nvidia_gpu_id}",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        total_mib = 0
        found = False
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or fields[0] != str(os.getpid()):
                continue
            try:
                total_mib += int(fields[1])
                found = True
            except ValueError:
                continue
        return total_mib * 1024 * 1024 if found else None

    def _nvidia_thermal() -> dict[str, float | None]:
        if nvidia_gpu_id is None:
            return {"gpu_temperature_celsius": None, "gpu_power_watts": None}
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={nvidia_gpu_id}",
                    "--query-gpu=temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return {"gpu_temperature_celsius": None, "gpu_power_watts": None}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {"gpu_temperature_celsius": None, "gpu_power_watts": None}
        fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
        temperature: float | None = None
        power: float | None = None
        if fields:
            try:
                temperature = float(fields[0])
            except ValueError:
                temperature = None
        if len(fields) > 1:
            try:
                power = float(fields[1])
            except ValueError:
                power = None
        return {"gpu_temperature_celsius": temperature, "gpu_power_watts": power}

    def _memory_evidence() -> dict[str, Any] | None:
        if measured_started is None:
            return None
        try:
            torch.cuda.synchronize(0)
        except Exception:  # noqa: BLE001
            pass
        if sampler_stop is not None:
            sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=3)
        samples["peak_host_rss"] = max(
            int(samples.get("peak_host_rss") or 0), _rss_bytes()
        )
        final_nvidia = _nvidia_process_bytes()
        if final_nvidia is not None:
            prior_nvidia = samples.get("peak_nvidia")
            samples["peak_nvidia"] = (
                final_nvidia
                if prior_nvidia is None
                else max(int(prior_nvidia), final_nvidia)
            )
        memory: dict[str, Any] = {
            "gpu_allocator_scope": "pytorch_cuda_allocator_process",
            "gpu_device_scope": "nvidia_smi_current_process"
            if baseline_nvidia is not None or samples.get("peak_nvidia") is not None
            else "unavailable",
            "host_memory_scope": "current_process_rss",
            "baseline_gpu_allocated_bytes": baseline_gpu_allocated,
            "baseline_gpu_reserved_bytes": baseline_gpu_reserved,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "baseline_nvidia_smi_process_bytes": baseline_nvidia,
            "peak_nvidia_smi_process_bytes": samples.get("peak_nvidia"),
            "baseline_host_rss_bytes": baseline_host_rss,
            "peak_host_rss_bytes": int(samples["peak_host_rss"]),
            "duration_seconds": time.perf_counter() - measured_started,
        }
        if record_phase_timing:
            memory.update(phase_timings)
            memory.update(thermal)
        return memory

    def _with_memory(outcome: ProbeOutcome) -> ProbeOutcome:
        memory = _memory_evidence()
        if memory is not None:
            outcome.measured["memory"] = memory
        return outcome

    def _placement_deviation(candidate: Any) -> str | None:
        expected_map = {"": "cuda:0"}
        parameters = _parameter_devices(candidate)
        if not parameters:
            return "the QLoRA model exposes no parameter inventory"
        bad_parameters = [name for name, device in parameters if device != "cuda:0"]
        bad_buffers = [
            name for name, device in _buffer_devices(candidate) if device != "cuda:0"
        ]
        hidden_offload = _offload_hook_deviations(candidate, expected_map)
        if bad_parameters or bad_buffers or hidden_offload:
            return (
                f"parameters={bad_parameters[:5]}, buffers={bad_buffers[:5]}, "
                f"hidden_offload={hidden_offload[:5]}"
            )
        return None

    try:
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        torch.manual_seed(0)
        torch.backends.cuda.enable_flash_sdp(enable_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(enable_math)
        attention_toggles = {
            "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "memory_efficient_sdp_enabled": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        }
        expected_toggles = {
            "flash_sdp_enabled": enable_flash,
            "memory_efficient_sdp_enabled": False,
            "math_sdp_enabled": enable_math,
        }
        if attention_toggles != expected_toggles:
            return ProbeOutcome(
                _TX.UNSUPPORTED_CONFIGURATION,
                f"{attention_kernel} SDPA toggles were not enforced: {attention_toggles}",
            )
        torch.cuda.synchronize(0)
        baseline_gpu_allocated = int(torch.cuda.memory_allocated(0))
        baseline_gpu_reserved = int(torch.cuda.memory_reserved(0))
        baseline_host_rss = _rss_bytes()
        baseline_nvidia = _nvidia_process_bytes()
        samples = {
            "peak_host_rss": baseline_host_rss,
            "peak_nvidia": baseline_nvidia,
        }
        sampler_stop = threading.Event()

        def _sample_resources() -> None:
            while sampler_stop is not None and not sampler_stop.wait(0.1):
                samples["peak_host_rss"] = max(
                    int(samples["peak_host_rss"]), _rss_bytes()
                )
                observed = _nvidia_process_bytes()
                if observed is not None:
                    prior = samples.get("peak_nvidia")
                    samples["peak_nvidia"] = observed if prior is None else max(int(prior), observed)

        # Reset immediately before the measured QLoRA section. Allocator peaks and nvidia-smi
        # process residency remain separate evidence and are never converted into fit claims.
        torch.cuda.reset_peak_memory_stats(0)
        measured_started = time.perf_counter()
        sampler_thread = threading.Thread(target=_sample_resources, daemon=True)
        sampler_thread.start()
        with tempfile.TemporaryDirectory() as root:
            base_dir = Path(root) / "base"
            adapter_dir = Path(root) / "adapter"
            base_config = _tiny_llama_config()
            base_config._attn_implementation = "sdpa"
            LlamaForCausalLM(base_config).save_pretrained(base_dir, safe_serialization=True)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            def _load_base() -> object:
                return AutoModelForCausalLM.from_pretrained(
                    base_dir,
                    use_safetensors=True,
                    trust_remote_code=False,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map={"": "cuda:0"},
                )

            model = prepare_model_for_kbit_training(_load_base(), use_gradient_checkpointing=True)
            model = get_peft_model(
                model,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules="all-linear",
                ),
            )
            placement_deviation = _placement_deviation(model)
            if placement_deviation is not None:
                return _with_memory(
                    ProbeOutcome(
                        _TX.UNSUPPORTED_CONFIGURATION,
                        "QLoRA singleton cuda:0 placement was not realized: "
                        + placement_deviation,
                    )
                )
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not trainable or any(
                parameter.device != torch.device("cuda:0") or parameter.dtype != torch.float32
                for parameter in trainable
            ):
                return _with_memory(
                    ProbeOutcome(
                        _TX.UNSUPPORTED_CONFIGURATION,
                        "QLoRA adapter placement or FP32 master-weight policy was not enforced",
                    )
                )
            linear4bit_modules = find_linear4bit_modules(model, BnbLinear4bit)
            quant_types = {
                getattr(getattr(module.weight, "quant_state", None), "quant_type", None)
                for module in linear4bit_modules
            }
            compute_dtypes = {
                getattr(module, "compute_dtype", None) for module in linear4bit_modules
            }
            nested_states = {
                bool(getattr(getattr(module.weight, "quant_state", None), "nested", False))
                for module in linear4bit_modules
            }
            quantized_devices = {str(module.weight.device) for module in linear4bit_modules}
            if (
                not linear4bit_modules
                or quant_types != {"nf4"}
                or compute_dtypes != {torch.bfloat16}
                or nested_states != {True}
                or quantized_devices != {"cuda:0"}
            ):
                return _with_memory(
                    ProbeOutcome(
                        _TX.UNSUPPORTED_CONFIGURATION,
                        "observed "
                        f"linear4bit={len(linear4bit_modules)}, "
                        f"quantization={quant_types}, compute_dtypes={compute_dtypes}, "
                        f"double_quant={nested_states}, devices={quantized_devices}",
                    )
                )
            optimizer = torch.optim.AdamW(trainable, lr=1e-3)
            before = [parameter.detach().clone() for parameter in trainable]
            # Shared with math/flash readiness probes for comparable measurements.
            input_ids = torch.randint(3, 64, (1, 8), device="cuda", dtype=torch.long)
            operation_toggles = {
                "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "memory_efficient_sdp_enabled": bool(
                    torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
                "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            }
            if operation_toggles != expected_toggles:
                return _with_memory(
                    ProbeOutcome(
                        _TX.UNSUPPORTED_CONFIGURATION,
                        f"SDPA toggles changed before forced {sdp_backend_name}: {operation_toggles}",
                    )
                )
            try:
                torch.cuda.synchronize(0)
                forward_started = time.perf_counter()
                # Autocast matches real QLoRA training compute dtype. Without it, PEFT k-bit prep
                # leaves residual/attention tensors in float32; math SDPA accepts float32 but forced
                # FLASH_ATTENTION aborts ("No available kernel"). Never enable math/mem-efficient
                # as a silent fallback when the forced backend fails.
                with (
                    sdpa_kernel([sdp_backend]),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    output = model(input_ids=input_ids, labels=input_ids)
                    torch.cuda.synchronize(0)
                    phase_timings["forward_duration_seconds"] = (
                        time.perf_counter() - forward_started
                    )
                    if not bool(torch.isfinite(output.loss).item()):
                        return _with_memory(
                            ProbeOutcome(
                                _TX.NUMERICAL_FAILURE,
                                f"BF16/NF4/QLoRA {attention_kernel} forward loss was non-finite",
                            )
                        )
                    backward_started = time.perf_counter()
                    output.loss.backward()
                    torch.cuda.synchronize(0)
                    phase_timings["backward_duration_seconds"] = (
                        time.perf_counter() - backward_started
                    )
            except Exception as exc:  # noqa: BLE001
                # No fallback to math / mem-efficient / eager: forced kernel failure is a FAIL.
                return _with_memory(
                    ProbeOutcome(
                        _TX.FAIL,
                        f"forced SDPBackend.{sdp_backend_name} execution failed: {exc}",
                    )
                )
            gradients = [parameter.grad for parameter in trainable]
            if any(
                gradient is None
                or gradient.dtype != torch.float32
                or gradient.device != torch.device("cuda:0")
                or not bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            ):
                return _with_memory(
                    ProbeOutcome(
                        _TX.NUMERICAL_FAILURE,
                        "QLoRA gradients were not finite FP32 tensors on cuda:0",
                    )
                )
            torch.cuda.synchronize(0)
            optimizer_started = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(0)
            phase_timings["optimizer_step_duration_seconds"] = (
                time.perf_counter() - optimizer_started
            )
            if not bool(torch.isfinite(output.loss).item()) or all(
                torch.equal(old, parameter) for old, parameter in zip(before, trainable, strict=True)
            ):
                return _with_memory(
                    ProbeOutcome(
                        _TX.NUMERICAL_FAILURE,
                        f"BF16/NF4/QLoRA {attention_kernel} AdamW step was invalid",
                    )
                )
            adapter_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
                if "lora_" in name
            }
            if not adapter_state:
                return _with_memory(
                    ProbeOutcome(_TX.CHECKPOINT_FAILURE, "adapter state inventory was empty")
                )
            model.save_pretrained(adapter_dir, safe_serialization=True)
            adapter_path = adapter_dir / "adapter_model.safetensors"
            if not adapter_path.is_file():
                return _with_memory(
                    ProbeOutcome(_TX.CHECKPOINT_FAILURE, "adapter safetensors was not written")
                )
            reloaded = PeftModel.from_pretrained(_load_base(), adapter_dir)
            try:
                with (
                    torch.no_grad(),
                    sdpa_kernel([sdp_backend]),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    reload_loss = reloaded(input_ids=input_ids, labels=input_ids).loss
            except Exception as exc:  # noqa: BLE001
                return _with_memory(
                    ProbeOutcome(
                        _TX.CHECKPOINT_FAILURE,
                        f"forced SDPBackend.{sdp_backend_name} reload forward failed: {exc}",
                    )
                )
            if not bool(torch.isfinite(reload_loss).item()):
                return _with_memory(
                    ProbeOutcome(_TX.CHECKPOINT_FAILURE, "reloaded QLoRA adapter was non-finite")
                )
            reloaded_state = {
                name: tensor.detach().cpu()
                for name, tensor in reloaded.state_dict().items()
                if "lora_" in name
            }
            if set(reloaded_state) != set(adapter_state) or any(
                not torch.equal(adapter_state[name], reloaded_state[name])
                for name in adapter_state
            ):
                return _with_memory(
                    ProbeOutcome(
                        _TX.CHECKPOINT_FAILURE,
                        "reloaded Safetensors adapter bytes differ from the saved adapter state",
                    )
                )
            reload_placement_deviation = _placement_deviation(reloaded)
            if reload_placement_deviation is not None:
                return _with_memory(
                    ProbeOutcome(
                        _TX.CHECKPOINT_FAILURE,
                        "reloaded QLoRA singleton cuda:0 placement was not realized: "
                        + reload_placement_deviation,
                    )
                )
            if record_phase_timing:
                thermal.update(_nvidia_thermal())
            adapter_bytes = adapter_path.stat().st_size
            package_versions = {
                name: metadata.version(name)
                for name in (
                    "accelerate",
                    "bitsandbytes",
                    "datasets",
                    "peft",
                    "safetensors",
                    "tokenizers",
                    "torch",
                    "transformers",
                    "trl",
                )
            }
            runtime_evidence = {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "torch_cuda_runtime": str(torch.version.cuda),
                "gpu_name": str(torch.cuda.get_device_name(0)),
                "compute_capability": ".".join(
                    str(item) for item in torch.cuda.get_device_capability(0)
                ),
                "packages": package_versions,
            }
        combination = _combination(
            runtime_mode="training",
            device="cuda",
            precision="bf16",
            quantization="nf4",
            adapter_method="qlora",
            attention_impl=attention_impl,
            attention_kernel=attention_kernel,
            probe=probe_name,
        )
        return _with_memory(
            ProbeOutcome(
                _TX.PASS,
                pass_detail,
                measured={
                    "loss": float(output.loss.detach().item()),
                    "reload_loss": float(reload_loss.detach().item()),
                    "adapter_weight_bytes": adapter_bytes,
                    "adapter_round_trip_verified": True,
                    "runtime": runtime_evidence,
                    "configuration": {
                        "compute_dtype": "bf16",
                        "forward_autocast": "bf16",
                        "quantization": "nf4",
                        "double_quantization": True,
                        "attention_api": "sdpa",
                        "attention_kernel": attention_kernel,
                        "forced_sdp_backend": sdp_backend_name,
                        "attention_toggles_before": previous_toggles,
                        "attention_toggles": attention_toggles,
                        "attention_toggles_during": operation_toggles,
                        "device_map": {"": "cuda:0"},
                        "target_modules": "all-linear",
                        "gradient_checkpointing": True,
                        "optimizer": "adamw_torch",
                        "batch_size": 1,
                        "sequence_length": 8,
                        "lora_r": 2,
                        "lora_alpha": 4,
                        "seed": 0,
                    },
                },
                proves={
                    "precision": ["bf16"],
                    "quantization": ["nf4"],
                    "attention": proves_attention,
                    "attention_kernel": [attention_kernel],
                    "adapter": ["qlora"],
                    "loss": ["cross_entropy"],
                    "optimizer": ["adamw_torch"],
                    "checkpoint": ["adapter_only"],
                },
                execution_combinations=[combination],
            )
        )
    except Exception as exc:  # noqa: BLE001
        return _with_memory(
            ProbeOutcome(_TX.FAIL, f"CUDA QLoRA execution tuple failed: {exc}")
        )
    finally:
        _restore_qlora_probe_process_state(
            torch,
            previous_toggles=previous,
            cpu_rng_state=cpu_rng_state,
            cuda_rng_states=cuda_rng_states,
            sampler_stop=sampler_stop,
            sampler_thread=sampler_thread,
        )


def _probe_gpu_responsive(profile: EnvironmentProfile) -> ProbeOutcome:
    # Detect a WEDGED GPU up front — the WSL2 GPU-PV state a crashed run leaves behind, where every
    # subsequent process fails with 'device not ready' regardless of config. Diagnosing it here (with
    # the OS-specific reset instruction) turns a cascade of cryptic failures into one clear "reset your
    # GPU" message. Runs first so a wedge is caught before the heavier probes hit the same wall.
    error = probe_gpu_responsive()
    health = classify_gpu_health(error)
    if health == "healthy":
        return ProbeOutcome(_TX.PASS, "GPU responds to a tiny CUDA op (not wedged)")
    if health == "absent":
        return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, error or "no CUDA GPU for a health probe")
    if health == "wedged":
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, wedged_gpu_remediation(profile.host.os))
    return ProbeOutcome(_TX.FAIL, f"GPU health probe returned an unclassified error: {error}")


BUILTIN_PROBES: dict[str, ProbeFn] = {
    "gpu_responsive": _probe_gpu_responsive,
    "cuda_available": _probe_cuda_available,
    "bf16_matmul": _probe_bf16_matmul,
    "bnb_4bit_load": _probe_bnb_4bit_load,
    "math_sdpa_backward": _probe_math_sdpa_backward,
    "flash_attn_backward": _probe_flash_attn_backward,
    "dense_optimizer_step": _probe_dense_optimizer_step,
    "trainer_contract": _probe_trainer_contract,
    "checkpoint_reload": _probe_checkpoint_reload,
    "cpu_lora_execution": _probe_cpu_lora_execution,
    "cuda_qlora_math_execution": _probe_cuda_qlora_math_execution,
    "cuda_qlora_sdpa_flash_execution": _probe_cuda_qlora_sdpa_flash_execution,
}


# --- runner ---------------------------------------------------------------------------------------


def _resolve_readiness(
    combinations: Sequence[ExecutionCapabilityCombination],
) -> Literal["ready", "cpu_toy_only", "not_ready"]:
    """Readiness is earned only by a complete execution tuple, not package presence."""

    if any(item.runtime_mode == "training" for item in combinations):
        return "ready"
    if any(item.runtime_mode == "cpu_toy" for item in combinations):
        return "cpu_toy_only"
    return "not_ready"


def _effective(
    proven: dict[str, set[str]],
    combinations: Sequence[ExecutionCapabilityCombination],
) -> EffectiveCapabilities:
    return EffectiveCapabilities(
        precision_modes=[PrecisionMode(v) for v in sorted(proven.get("precision", set()))],
        quantization_modes=[
            QuantizationMode(v) for v in sorted(proven.get("quantization", set()))
        ],
        attention_impls=[AttentionImpl(v) for v in sorted(proven.get("attention", set()))],
        attention_kernels=[
            AttentionKernel(v) for v in sorted(proven.get("attention_kernel", set()))
        ],
        adapter_methods=[AdapterMethod(v) for v in sorted(proven.get("adapter", set()))],
        loss_impls=[LossImpl(v) for v in sorted(proven.get("loss", set()))],
        optimizers=[Optimizer(v) for v in sorted(proven.get("optimizer", set()))],
        checkpoint_impls=[
            CheckpointImpl(v) for v in sorted(proven.get("checkpoint", set()))
        ],
        execution_contract_versions=sorted(proven.get("execution_contract", set())),
        execution_combinations=sorted(combinations, key=lambda item: item.canonical_key()),
        trainer_fields=sorted(proven.get("trainer_field", set())),
        trainer_init_fields=sorted(proven.get("trainer_init_field", set())),
        offload_strategies=[
            OffloadStrategy(v) for v in sorted(proven.get("offload", set()))
        ],
        placement_tiers=[
            MemoryTier(v) for v in sorted(proven.get("placement_tier", set()))
        ],
        placement_modes=[
            PlacementMode(v) for v in sorted(proven.get("placement_mode", set()))
        ],
        parallelism_kinds=[
            ParallelismKind(v) for v in sorted(proven.get("parallelism", set()))
        ],
        communication_backends=[
            CommunicationBackend(v)
            for v in sorted(proven.get("communication_backend", set()))
        ],
    )


def run_capability_probes(
    profile: EnvironmentProfile,
    *,
    backend_id: str = "corpus_studio",
    backend_version: str | None = None,
    probes: Sequence[str] | None = None,
    registry: dict[str, ProbeFn] | None = None,
) -> CapabilityReport:
    """Run the requested probes against ``profile`` and build a :class:`CapabilityReport`.

    A probe is never allowed to crash the runner — any exception becomes an ``ENVIRONMENT_FAILURE``
    result. ``registry`` (defaulting to :data:`BUILTIN_PROBES`) is injectable so the framework can be
    unit-tested with fakes. ``effective_capabilities`` is the union of what the PASSED probes proved on
    this host — the intersection with a backend's declared surface belongs to the planner.
    """
    if backend_version is None:
        from corpus_studio.platform.backends import get_backend  # noqa: PLC0415

        backend = get_backend(backend_id)
        backend_version = backend.backend_version if backend is not None else None

    reg = registry if registry is not None else BUILTIN_PROBES
    names: Iterable[str] = probes if probes is not None else list(reg)

    results: list[ProbeResult] = []
    by_probe: dict[str, FailureTaxonomy] = {}
    execution_combinations: list[ExecutionCapabilityCombination] = []
    proven: dict[str, set[str]] = {
        "precision": set(),
        "quantization": set(),
        "attention": set(),
        "attention_kernel": set(),
        "adapter": set(),
        "loss": set(),
        "optimizer": set(),
        "checkpoint": set(),
        "execution_contract": set(),
        "trainer_field": set(),
        "trainer_init_field": set(),
        "offload": set(),
        "placement_tier": set(),
        "placement_mode": set(),
        "parallelism": set(),
        "communication_backend": set(),
    }
    for name in names:
        fn = reg.get(name)
        if fn is None:
            outcome = ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, f"unknown probe '{name}'")
        else:
            try:
                outcome = fn(profile)
            except Exception as exc:  # noqa: BLE001 - a probe must never crash the runner.
                outcome = ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"probe raised: {exc}")
        passing_evidence = outcome.taxonomy == _TX.PASS
        results.append(
            ProbeResult(
                probe=name,
                outcome=outcome.taxonomy,
                detail=outcome.detail,
                measured=outcome.measured,
                proves=(
                    {axis: sorted(set(tokens)) for axis, tokens in outcome.proves.items()}
                    if passing_evidence
                    else {}
                ),
                execution_combinations=sorted(
                    outcome.execution_combinations if passing_evidence else [],
                    key=lambda item: item.canonical_key(),
                ),
            )
        )
        by_probe[name] = outcome.taxonomy
        if outcome.taxonomy == _TX.PASS:
            for axis, tokens in outcome.proves.items():
                proven.setdefault(axis, set()).update(tokens)
            execution_combinations.extend(outcome.execution_combinations)

    # Contract support is conjunctive: the exact trainer field surface and at least one complete
    # execution tuple must both pass. Independent axis probes can never mint this capability.
    if by_probe.get("trainer_contract") == _TX.PASS and execution_combinations:
        proven["execution_contract"].add("1.0.0")

    installed = [p for p in profile.packages if p.version is not None]
    missing = [p.name for p in profile.packages if p.version is None and p.name in _TRAIN_PACKAGES]
    return CapabilityReport(
        backend_id=backend_id,
        backend_version=backend_version,
        environment_ref=Ref(id=profile.environment_signature),
        generated_at=datetime.now(timezone.utc).isoformat(),
        readiness=_resolve_readiness(execution_combinations),
        bitsandbytes_ok=by_probe.get("bnb_4bit_load") == _TX.PASS,
        installed_packages=[PackageLock(name=p.name, version=p.version) for p in installed],
        missing_packages=missing,
        probe_results=results,
        effective_capabilities=_effective(proven, execution_combinations),
    )
