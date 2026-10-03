"""Lower a sealed model/tokenizer identity and loader policy in the newer first-party workers.

The DPO, reward, full-parameter SFT and on-policy RL seals pin the same loader fields as the adapter SFT
seal: a model binding and a SEPARATE tokenizer binding (each an immutable Hugging Face commit or a
digest-pinned local directory), ``use_safetensors`` and ``trust_remote_code``, the attention API plus
one exact kernel and its SDPA toggles, an explicit device map, and the storage/compute/master dtypes. The
adapter SFT trainer lowers and observes every one of those fields; this module lets the newer workers do
exactly the same through the SAME helpers instead of hand-written ``from_pretrained`` calls:

1. :func:`sealed_loader_view` maps a sealed execution onto the import-light ``TrainRunConfig`` those
   helpers are typed on (identity, placement, precision, attention and data-policy fields only);
2. :func:`load_sealed_tokenizer` loads the TOKENIZER binding (its own location and revision) and checks a
   sealed chat-template digest;
3. :func:`prepare_sealed_attention` applies the three SDPA toggles, observes them, and runs the isolated
   sealed-kernel probe before any weights are allocated;
4. :func:`load_sealed_model` builds the loader arguments with ``build_model_load_kwargs`` (revision,
   safetensors-only, sealed quantization and compute dtype or storage dtype, explicit device map, sealed
   attention API) and then re-hashes local inputs and observes the attention API and placement;
5. :func:`verify_sealed_adapter_precision` (QLoRA lanes, after PEFT attachment) lowers the sealed master
   dtype and observes nf4 storage, the dequantization dtype, trainable dtypes and post-adapter
   placement; :func:`verify_full_parameter_storage` (full-parameter lane) observes the storage dtype.

The training call itself runs inside ``trainer.enforced_attention_training_kernel`` so only the sealed
SDPA kernel can dispatch. Anything these helpers cannot lower exactly is refused before loading by
``execution_config.verify_loader_policy_supported`` (planner, runner and worker). The pre-load re-hash of
local model/tokenizer bindings is ``execution_config.verify_execution_non_dataset_inputs``, called by the
runners; the dataset read is ``training.sealed_inputs``.

Torch-free at import: heavy modules and loader classes arrive as parameters, exactly like the SFT helpers.
Every refusal is a ``TrainerError`` subclass, so the runners map it to the same taxonomy as the adapter
SFT lane (placement deviation, environment failure, or unsupported configuration).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from corpus_studio.platform.contracts import (
    ResolvedFullFinetuneExecutionConfiguration,
    ResolvedRolloutExecutionConfiguration,
)
from corpus_studio.platform.execution_config import LoaderLaneExecution
from corpus_studio.training.trainer import (
    ExecutionPlacementDeviation,
    GradientObservationTracker,
    TrainerError,
    TrainRunConfig,
    _torch_dtype,
    apply_attention_execution_policy,
    build_model_load_kwargs,
    probe_effective_attention_kernel,
    reassert_trainable_precision,
    verify_loaded_model_execution,
    verify_local_inputs_after_load,
    verify_model_state_execution,
)

StageFn = Callable[[str, str], None]

# Kernels ``probe_effective_attention_kernel`` exercises; eager and external kernels do not dispatch
# through PyTorch SDPA, so there is nothing for the isolated SDPA probe to observe.
_PROBED_KERNELS = frozenset({"torch_sdpa_math", "torch_sdpa_flash", "torch_sdpa_mem_efficient"})


def no_stage(_name: str, _message: str) -> None:
    """The stage sink a worker uses when its caller supplied none."""


def sealed_truncation_permitted(execution: LoaderLaneExecution) -> bool:
    """Whether this seal permits cutting content that exceeds the sealed sequence window.

    The DATA policy is the enforced key, exactly as on the adapter SFT lane (``trainer.py`` reads
    ``data.truncation_policy``): ``data.truncation_policy`` here, or ``experience.truncation_policy``
    on the on-policy RL lane. ``sequence.truncation_allowed`` must agree, which makes the rule
    fail-closed from either direction.

    Reading ``sequence.truncation_allowed`` ALONE inverts the contract's own safe default.
    ``SequenceSpec.truncation_allowed`` defaults to True and ``truncation_policy`` defaults to
    ``"refuse"``, and the contract validators reject only the opposite pair
    (``truncation_allowed=False`` with policy ``"allow"``). So ``(True, "refuse")`` - which the
    contract documents as the stricter safe default - would permit silent truncation on a lane that
    consulted the sequence flag by itself.
    """

    policy = (
        execution.experience.truncation_policy
        if isinstance(execution, ResolvedRolloutExecutionConfiguration)
        else execution.data.truncation_policy
    )
    return policy == "allow" and execution.sequence.truncation_allowed


def sealed_loader_view(execution: LoaderLaneExecution) -> TrainRunConfig:
    """Map a sealed DPO, reward, full-parameter SFT or on-policy RL execution onto ``TrainRunConfig``.

    Only identity, placement, precision, attention, schedule and data-policy fields are populated. LoRA,
    optimizer and trainer-interface fields keep inert defaults, so the view must never be passed to
    ``build_training_kwargs`` or ``build_lora_kwargs``; each worker lowers those from its own seal.
    ``execution_configuration_hash`` is always set, which switches every helper to its sealed,
    fail-closed behavior.
    """

    inputs = execution.inputs
    precision = execution.precision
    attention = execution.attention
    data_fields: dict[str, Any]
    if isinstance(execution, ResolvedRolloutExecutionConfiguration):
        data_fields = {
            "chat_template_sha256": execution.experience.chat_template_sha256,
            "truncation_allowed": sealed_truncation_permitted(execution),
        }
    else:
        data_fields = {
            "chat_template_sha256": execution.data.chat_template_sha256,
            "truncation_allowed": sealed_truncation_permitted(execution),
        }
    if isinstance(execution, ResolvedFullFinetuneExecutionConfiguration):
        data_fields.update(
            dataset_format=execution.data.dataset_format,
            formatter_id=execution.data.formatter_id,
            formatter_sha256=execution.data.formatter_sha256,
            packing=execution.data.packing,
            dataset_text_field=execution.data.dataset_text_field,
        )
    return TrainRunConfig(
        base_model=inputs.model.location,
        model_revision=inputs.model.resolved_revision,
        model_source=inputs.model.source,
        model_content_sha256=inputs.model.content_sha256,
        tokenizer_location=inputs.tokenizer.location,
        tokenizer_revision=inputs.tokenizer.resolved_revision,
        tokenizer_source=inputs.tokenizer.source,
        tokenizer_content_sha256=inputs.tokenizer.content_sha256,
        dataset_path=inputs.dataset.location,
        dataset_sha256=inputs.dataset.content_sha256,
        execution_configuration_hash=execution.configuration_hash,
        output_dir=execution.output_dir,
        sequence_len=execution.sequence.max_sequence_len,
        seed=execution.seed,
        data_seed=execution.data_seed,
        cpu_toy=execution.runtime_mode == "cpu_toy",
        max_steps=execution.schedule.max_steps,
        num_train_epochs=execution.schedule.num_train_epochs,
        attn_implementation=attention.model_attention_api.value,
        attention_kernel=attention.effective_backend_required.value,
        flash_sdp_enabled=attention.flash_sdp_enabled,
        mem_efficient_sdp_enabled=attention.mem_efficient_sdp_enabled,
        math_sdp_enabled=attention.math_sdp_enabled,
        quantization_mode=precision.quantized_storage_format.value,
        weight_storage_dtype=(
            precision.weight_storage_dtype.value
            if precision.weight_storage_dtype is not None
            else None
        ),
        dequantization_dtype=precision.dequantization_dtype.value,
        forward_compute_dtype=precision.forward_compute_dtype.value,
        gradient_dtype=precision.gradient_dtype.value,
        optimizer_state_dtype=precision.optimizer_state_dtype.value,
        optimizer_auxiliary_dtype=precision.optimizer_auxiliary_dtype.value,
        master_weight_dtype=(
            precision.master_weight_dtype.value
            if precision.master_weight_dtype is not None
            else None
        ),
        # Collapsing the sealed list into a dict is lossless only because every worker first passes
        # ``verify_loader_policy_supported``, which admits exactly one root entry and no repeats.
        device_map={entry.module: entry.device for entry in execution.device_map},
        trust_remote_code=execution.trust_remote_code,
        use_safetensors=execution.use_safetensors,
        bnb_4bit_use_double_quant=execution.bnb_4bit_use_double_quant,
        adapter_task_type=execution.adapter_task_type,
        export_format=execution.export_format.value,
        gradient_checkpointing=execution.gradient_checkpointing,
        **data_fields,
    )


def _pin(source: str | None, revision: str | None, digest: str | None) -> str:
    if source == "huggingface" and revision is not None:
        return f"@{revision}"
    if digest is not None:
        return f" sha256:{digest}"
    return ""


def describe_sealed_identity(view: TrainRunConfig) -> str:
    """One ASCII line naming the pinned model and tokenizer identities and the safety loader flags."""

    tokenizer_location = view.tokenizer_location or view.base_model
    model_pin = _pin(view.model_source, view.model_revision, view.model_content_sha256)
    tokenizer_pin = _pin(
        view.tokenizer_source, view.tokenizer_revision, view.tokenizer_content_sha256
    )
    return (
        f"model {view.base_model}{model_pin} ({view.model_source}), "
        f"tokenizer {tokenizer_location}{tokenizer_pin} ({view.tokenizer_source}), "
        f"use_safetensors={view.use_safetensors}, trust_remote_code={view.trust_remote_code}"
    )


def sealed_tokenizer_load_args(view: TrainRunConfig) -> tuple[str, dict[str, Any]]:
    """The exact ``AutoTokenizer.from_pretrained`` location and keyword arguments for the TOKENIZER
    binding: its own location (never the model's when they differ), its own immutable revision when it is
    a Hub commit, and the sealed ``trust_remote_code`` (never the library default)."""

    kwargs: dict[str, Any] = {"trust_remote_code": view.trust_remote_code}
    if view.tokenizer_revision is not None:
        kwargs["revision"] = view.tokenizer_revision
    return view.tokenizer_location or view.base_model, kwargs


def verify_sealed_chat_template(tokenizer: Any, expected_sha256: str | None) -> None:
    """Refuse a loaded tokenizer whose chat template differs from a sealed chat-template digest.

    ``None`` means the seal pins no separate template digest; the template is then pinned only through
    the tokenizer binding itself (its revision or directory digest)."""

    if expected_sha256 is None:
        return
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise TrainerError("the pinned tokenizer has no usable chat template")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise TrainerError("the pinned tokenizer cannot apply its chat template")
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != expected_sha256:
        raise TrainerError("the tokenizer chat template changed after planning")


def load_sealed_tokenizer(auto_tokenizer: Any, view: TrainRunConfig, *, stage: StageFn) -> Any:
    """Load the sealed tokenizer binding and verify its sealed chat-template digest."""

    location, kwargs = sealed_tokenizer_load_args(view)
    stage("tokenizer_load", f"loading the sealed tokenizer; {describe_sealed_identity(view)}")
    tokenizer = auto_tokenizer.from_pretrained(location, **kwargs)
    verify_sealed_chat_template(tokenizer, view.chat_template_sha256)
    stage("tokenizer_load", "loaded and verified the sealed tokenizer")
    return tokenizer


def prepare_sealed_attention(torch_module: Any, view: TrainRunConfig, *, stage: StageFn) -> str:
    """Apply and observe the sealed SDPA toggles, then probe the sealed kernel in isolation, before any
    model weights are allocated. Returns the required kernel name."""

    stage("attention_policy_applied", "applying the sealed SDP toggles before model allocation")
    kernel = apply_attention_execution_policy(torch_module, view)
    probe_effective_attention_kernel(torch_module, view)
    probe = (
        "passed its isolated forward/backward probe"
        if kernel in _PROBED_KERNELS
        else "does not dispatch through PyTorch SDPA (no SDPA probe applies)"
    )
    stage(
        "attention_policy_applied",
        f"applied and observed the exact SDP toggles; required kernel {kernel} {probe}",
    )
    return kernel


def observe_sealed_placement(
    model: Any, view: TrainRunConfig, *, stage: StageFn, label: str = "model"
) -> None:
    """Observe a loaded model's attention API and the device of every parameter, buffer and Accelerate
    hook against the sealed device map; a deviation is streamed as ``placement_deviation`` and refused."""

    try:
        verify_loaded_model_execution(model, view)
    except ExecutionPlacementDeviation as exc:
        stage("placement_deviation", str(exc))
        raise
    stage(
        "placement_verified",
        f"observed the {label} on the exact sealed device map {view.device_map} with model attention "
        f"API {view.attn_implementation}",
    )


def verify_sealed_model_after_load(
    model: Any, view: TrainRunConfig, *, stage: StageFn, label: str = "model"
) -> None:
    """Re-hash local model/tokenizer inputs after the third-party load, then observe the loaded model's
    attention API and its placement against the sealed device map."""

    verify_local_inputs_after_load(view)
    observe_sealed_placement(model, view, stage=stage, label=label)


def load_sealed_model(
    loader: Any,
    torch_module: Any,
    view: TrainRunConfig,
    *,
    stage: StageFn,
    bitsandbytes_config_cls: Any | None = None,
    label: str = "model",
    **task_kwargs: Any,
) -> Any:
    """Load the sealed model binding with every sealed loader field lowered, then verify it.

    ``task_kwargs`` carries head-shape arguments that are not loader policy (``num_labels=1`` for a
    scalar score head); it may never override a sealed loader field.
    """

    quantize = view.quantization_mode not in (None, "none")
    kwargs = build_model_load_kwargs(
        view,
        torch_module,
        quantize=quantize,
        bitsandbytes_config_cls=bitsandbytes_config_cls,
    )
    overridden = sorted(set(kwargs) & set(task_kwargs))
    if overridden:
        raise TrainerError(
            f"task loader arguments cannot override sealed loader fields: {', '.join(overridden)}"
        )
    stage("model_load", f"loading the sealed {label} weights")
    model = loader.from_pretrained(view.base_model, **kwargs, **task_kwargs)
    stage("model_load", f"materialized the sealed {label} weights")
    verify_sealed_model_after_load(model, view, stage=stage, label=label)
    return model


def verify_sealed_adapter_precision(
    model: Any,
    torch_module: Any,
    view: TrainRunConfig,
    gradient_tracker: GradientObservationTracker,
    *,
    stage: StageFn,
    linear4bit_type: type[Any] | None = None,
) -> None:
    """QLoRA lanes, after PEFT attachment and gradient-hook registration: put the identity-bound
    trainable parameters in the sealed master dtype, then observe post-adapter placement, bitsandbytes
    storage type, dequantization dtype and trainable dtypes against the seal."""

    reassert_trainable_precision(model, torch_module, view, gradient_tracker)
    verify_model_state_execution(
        model, torch_module, view, quantize=True, linear4bit_type=linear4bit_type
    )
    stage(
        "placement_verified",
        f"observed post-adapter parameter placement on {(view.device_map or {}).get('', '')}",
    )
    stage(
        "precision_verified",
        f"observed {view.quantization_mode} base storage, {view.dequantization_dtype} dequantization, "
        f"and {view.master_weight_dtype} trainable master weights",
    )


def verify_full_parameter_storage(
    model: Any, torch_module: Any, view: TrainRunConfig, *, stage: StageFn
) -> None:
    """Full-parameter lane: every floating parameter must be stored in the sealed weight-storage dtype
    (a model class that silently keeps some modules in another dtype is refused, not trained)."""

    if view.weight_storage_dtype is None:
        raise TrainerError("sealed full-parameter execution omitted its weight-storage dtype")
    expected = _torch_dtype(torch_module, view.weight_storage_dtype)
    observed = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if observed != {expected}:
        rendered = ", ".join(sorted(str(dtype) for dtype in observed)) or "none"
        raise TrainerError(
            f"full-parameter weight-storage dtype deviation: observed {rendered}, expected {expected}"
        )
    stage(
        "precision_verified",
        f"observed every floating parameter stored as the sealed {view.weight_storage_dtype}",
    )
