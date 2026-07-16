"""First-party SFT/QLoRA trainer (the opt-in ``[train]`` extra).

Runs training **in-process** so CorpusStudio can go config → *train* → adapter without the user
bringing an external trainer. Every heavy import (torch / transformers / peft / trl / datasets /
bitsandbytes) is **lazy** — done inside :func:`run_training`, never at module load — so importing this
module in the dependency-light engine costs nothing and never fails when the extra is absent.

Two paths share one code route:

* **GPU 4-bit QLoRA** — the real run: NF4 bitsandbytes quantization + LoRA on the config's base model.
* **CPU toy** (``cpu_toy=True``) — a tiny model, no quantization, a few steps, so the *plumbing* is
  provable without a GPU (this is what makes the trainer verifiable in CI-less environments).

The pure helpers (config load, row→text formatting, the LoRA/SFT arg mapping, run-plan resolution)
carry no heavy imports and are unit-tested with the deps mocked; :func:`run_training` itself is
verified via the CPU toy path.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
import re
import sys
from pathlib import Path
from typing import Any, Callable, Literal, cast

from pydantic import BaseModel, Field, model_validator

from corpus_studio.importers.jsonl_importer import read_jsonl, read_jsonl_bytes
from corpus_studio.platform.contracts import (
    AdapterExportStateEvidence,
    GradientCoverageEvidence,
    OptimizerStepLossEvidence,
    TrainableStateChangeEvidence,
    TrainingExecutionEvidence,
)
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.training.environment import INSTALL_HINT, probe_training_runtime
from corpus_studio.training.quantization import find_linear4bit_modules

# Tiny Llama-architecture model for the CPU toy path. The CPU toy builds the model FROM CONFIG
# (random weights — no weights download, so it works offline/behind a firewall), and only the small
# config + tokenizer are fetched (once). nn.Linear layers → LoRA "all-linear" works, matching the real
# Qwen family. It is a smoke test of the pipeline, not a model meant to learn the task.
TINY_TOY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

ProgressCallback = Callable[[int, int, float | None], None]
# (optimizer_step, nonpadding_tokens, supervised_tokens, observed_microbatches) — the real token counts
# the trainer consumed for one completed optimizer step (summed across its accumulation microbatches),
# plus HOW MANY microbatches were actually observed. ``observed_microbatches == 0`` means the observer
# never saw a batch (the count is UNAVAILABLE, not a measured zero); a downstream sink must map that to
# null, never 0.0. Kept separate from ProgressCallback so the loss path is untouched; token accounting is
# pure observation and never influences training.
TokenCallback = Callable[[int, int, int, int], None]
# (stage_name, message) — platform-agnostic strings so the trainer stays decoupled from the platform
# enums. Fires at setup milestones so a supervisor sees progress during the long silent model-load.
StageCallback = Callable[[str, str], None]
_MAX_PREFLIGHT_PROGRESS_EVENTS = 20


def _flatten_ints(value: Any) -> Iterator[int]:
    """Yield every scalar in ``value`` as an int, flattening nested rows. Frameworks expose ``tolist``
    (torch/numpy), which collapses a tensor to nested python lists; a plain nested list/tuple is walked
    directly. Read-only - it never mutates the batch it inspects."""

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_ints(item)
    else:
        yield int(value)


def _count_ne(value: Any, sentinel: int) -> int:
    """Number of scalar elements of ``value`` not equal to ``sentinel``.

    Prefers a vectorized, non-materializing path for objects with tensor/ndarray semantics
    (``value != sentinel`` -> boolean tensor -> ``.sum()`` -> scalar), which does NOT move the input off
    its device, change its dtype, or retain it. Falls back to flattening for python lists/tuples. Purely
    observational and read-only."""

    if not isinstance(value, (list, tuple)) and hasattr(value, "sum"):
        try:
            comparison = value != sentinel
            total = comparison.sum()
            item = getattr(total, "item", None)
            return int(item() if callable(item) else total)
        except Exception:  # noqa: BLE001 - fall back to the framework-agnostic path
            pass
    return sum(1 for element in _flatten_ints(value) if element != sentinel)


def _count_all(value: Any) -> int:
    """Total scalar element count of a tensor/ndarray/nested list (non-materializing when possible)."""

    numel = getattr(value, "numel", None)  # torch tensor
    if callable(numel):
        try:
            return int(numel())
        except Exception:  # noqa: BLE001
            pass
    size = getattr(value, "size", None)  # numpy ndarray exposes an int ``size`` attribute
    if isinstance(size, int):
        return size
    return sum(1 for _ in _flatten_ints(value))


def count_batch_tokens(batch: Mapping[str, Any]) -> tuple[int, int]:
    """Count ``(nonpadding, supervised)`` tokens in one collated microbatch. Non-padding = attention
    mask ones (or, absent a mask, every ``input_ids`` position); supervised = label positions not equal
    to the ``-100`` cross-entropy ignore index. Framework-agnostic (torch tensor / numpy / nested list)
    and purely observational, so it is unit-testable without the training stack and can never change a
    training byte. Reads a tensor's counts with vectorized ops - it never moves, mutates, or retains it."""

    mask = batch.get("attention_mask")
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if mask is not None:
        nonpadding = _count_ne(mask, 0)
    elif input_ids is not None:
        nonpadding = _count_all(input_ids)
    else:
        nonpadding = 0
    supervised = _count_ne(labels, -100) if labels is not None else 0
    return nonpadding, supervised


class _TokenAccumulator:
    """Sums per-microbatch token counts into a per-optimizer-step total, and counts HOW MANY microbatches
    it actually observed, flushed and reset at each step boundary. With the sealed single-worker
    dataloader (no background prefetch) every microbatch for step N is consumed by ``training_step``
    before step N's boundary fires, so the flush attributes exactly step N's tokens. Fully observational:
    a counting fault is swallowed (and NOT counted as an observed microbatch) so training is never
    disturbed and an observer failure surfaces downstream as ``observed == 0`` (unavailable), never as a
    measured zero."""

    def __init__(self) -> None:
        self._nonpadding = 0
        self._supervised = 0
        self._observed = 0

    def observe(self, batch: Mapping[str, Any]) -> None:
        try:
            nonpadding, supervised = count_batch_tokens(batch)
        except Exception:  # noqa: BLE001 - token accounting must never affect training
            return
        self._nonpadding += nonpadding
        self._supervised += supervised
        self._observed += 1

    def flush(self) -> tuple[int, int, int]:
        totals = (self._nonpadding, self._supervised, self._observed)
        self._nonpadding = 0
        self._supervised = 0
        self._observed = 0
        return totals


class TrainerError(Exception):
    """Raised for an unrunnable request (runtime missing, bad config) — a clean CLI exit, not a crash."""


class ExecutionPlacementDeviation(TrainerError):
    """The loaded model does not match the device placement sealed by the RunPlan."""


class TrainerEnvironmentError(TrainerError):
    """The sealed runtime, package set, or device is unavailable or drifted."""


class TrainingEvidenceError(TrainerError):
    """A classified trainer-side success invariant failed at an exact lifecycle stage."""

    def __init__(
        self,
        message: str,
        *,
        taxonomy: FailureTaxonomy,
        stage: StageMarker,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.taxonomy = taxonomy
        self.stage = stage
        self.remediation = remediation


class TrainRunConfig(BaseModel):
    """Resolved inputs for one training run (from a CorpusStudio training config + overrides)."""

    base_model: str
    dataset_path: str
    execution_configuration_hash: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_location: str | None = None
    model_source: str | None = None
    tokenizer_source: str | None = None
    model_content_sha256: str | None = None
    tokenizer_content_sha256: str | None = None
    dataset_sha256: str | None = None
    output_dir: str = "output"
    dataset_format: str = "instruction"
    sequence_len: int = Field(default=4096, gt=0)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    lora_bias: str = "none"
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.999, ge=0, lt=1)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, ge=0)
    lr_scheduler: str = "linear"
    warmup_ratio: float = Field(default=0.0, ge=0, le=1)
    seed: int = Field(default=42, ge=0)
    data_seed: int = Field(default=42, ge=0)
    cpu_toy: bool = False
    max_steps: int | None = None
    num_train_epochs: float | None = None
    # Attention backend passed to from_pretrained (e.g. "eager" / "sdpa" / "flash_attention_2").
    # None = auto: honor transformers' default, except native-Windows/WDDM Blackwell where measured
    # fused flash deadlock evidence requires the safe fallback. Other hosts still need their probes.
    attn_implementation: str | None = None
    attention_kernel: str | None = None
    flash_sdp_enabled: bool | None = None
    mem_efficient_sdp_enabled: bool | None = None
    math_sdp_enabled: bool | None = None
    quantization_mode: str | None = None
    weight_storage_dtype: str | None = None
    dequantization_dtype: str = "bf16"
    forward_compute_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    optimizer_auxiliary_dtype: str = "fp32"
    master_weight_dtype: str | None = None
    device_map: dict[str, str] | None = None
    trust_remote_code: bool = False
    use_safetensors: bool = True
    bnb_4bit_use_double_quant: bool = True
    adapter_task_type: str = "CAUSAL_LM"
    export_format: str = "adapter_peft"
    # --- memory / spill-avoidance levers (opt-in) ---
    # The optimizer TRL/transformers uses. "paged_adamw_8bit" (bitsandbytes) pages optimizer state to
    # host RAM under pressure via CUDA managed memory — a spike-safe SAFE-SPILL for the optimizer
    # (verified on a real 5070 under WSL2). For QLoRA the optimizer is small (LoRA params only), so this
    # is more crash-safety than a big saving; the base 4-bit model dominates VRAM.
    optim: str = "adamw_torch"
    # Fuse the cross-entropy loss (Liger) so the full-vocab logits are never materialized — removes the
    # ~2.5 GB fp32 logits spike at long sequence_len + large vocab (the real long-seq memory lever).
    # Requires the `liger-kernel` package and a matching proven SFTConfig field. Sealed runs refuse
    # this option if the installed trainer cannot honor it. Blackwell support is not verified here.
    use_liger: bool = False
    gradient_checkpointing: bool = True
    # Intermediate checkpoints are disabled because the first-party trainer has no compatible
    # resume lineage yet. ``steps`` remains parseable only for legacy document compatibility and is
    # refused before execution.
    save_steps: int | None = Field(default=None, gt=0)
    save_total_limit: int | None = Field(default=None, ge=1)
    save_strategy: Literal["no", "steps"] = "no"
    logging_strategy: Literal["steps"] = "steps"
    logging_steps: Literal[1] = 1
    logging_nan_inf_filter: Literal[False] = False
    report_to: list[str] = Field(default_factory=list)
    dataset_text_field: str = "text"
    disable_tqdm: bool = True
    packing: bool = False
    truncation_allowed: bool = True
    formatter_id: str | None = None
    formatter_sha256: str | None = None
    chat_template_sha256: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    required_sft_config_fields: list[str] = Field(default_factory=list)
    sequence_length_field: str = "auto"
    tokenizer_parameter: str = "auto"

    @model_validator(mode="after")
    def _validate_checkpoint_settings(self) -> TrainRunConfig:
        if self.save_strategy == "no":
            if self.save_steps is not None or self.save_total_limit is not None:
                raise ValueError("disabled checkpointing cannot carry save cadence or retention")
        elif self.save_steps is None:
            raise ValueError("step checkpointing requires save_steps")
        return self


class TrainResult(BaseModel):
    output_dir: str
    adapter_path: str
    base_model: str
    cpu_toy: bool
    steps: int = 0
    final_loss: float | None = None
    checkpoints: list[str] = Field(default_factory=list)
    execution_evidence: TrainingExecutionEvidence | None = None


def _require_checkpoint_free_execution(config: TrainRunConfig) -> None:
    """Refuse every intermediate-checkpoint spelling on the in-process SFTTrainer path, including
    unvalidated model copies. The SFTTrainer body stays checkpoint-free; exact-lineage checkpoint
    writing/resume runs through :mod:`corpus_studio.training.checkpoint_io` (the reviewed engine proven
    by the real-torch fresh-process equivalence test), which a long run binds on first authorization."""

    if (
        config.save_strategy != "no"
        or config.save_steps is not None
        or config.save_total_limit is not None
    ):
        raise TrainerError(
            "intermediate checkpoints on the SFTTrainer path are unsupported; exact-lineage "
            "checkpointing runs through corpus_studio.training.checkpoint_io"
        )


@dataclass(frozen=True)
class CheckpointExecutionPolicy:
    """The resolved checkpoint policy for one run: ``enabled=False`` is the checkpoint-free policy a
    short run keeps (byte-identical to a no-checkpoint run); ``enabled=True`` writes exact-lineage
    checkpoints every ``cadence_optimizer_steps`` and keeps at most ``keep_last`` of them."""

    enabled: bool
    cadence_optimizer_steps: int | None = None
    keep_last: int | None = None


def resolve_checkpoint_execution_policy(config: TrainRunConfig) -> CheckpointExecutionPolicy:
    """Classify a run's checkpoint policy from its sealed save settings. ``save_strategy="no"`` is the
    checkpoint-free policy (short runs, unchanged behavior); ``save_strategy="steps"`` is the sealed
    checkpoint-enabled policy that requires a positive optimizer-step cadence. Fails closed on any
    inconsistent spelling so a run can never half-enable checkpoints."""

    if config.save_strategy == "no":
        if config.save_steps is not None or config.save_total_limit is not None:
            raise TrainerError("disabled checkpointing cannot carry save cadence or retention")
        return CheckpointExecutionPolicy(enabled=False)
    if config.save_steps is None or config.save_steps < 1:
        raise TrainerError("step checkpointing requires a positive save_steps cadence")
    return CheckpointExecutionPolicy(
        enabled=True,
        cadence_optimizer_steps=config.save_steps,
        keep_last=config.save_total_limit,
    )


def train_config_from_resolved(execution: Any) -> TrainRunConfig:
    """Map the typed platform contract to the import-light trainer boundary without defaults."""

    if (
        execution.trainer_interface.logging_strategy != "steps"
        or execution.trainer_interface.logging_steps != 1
        or execution.trainer_interface.logging_nan_inf_filter is not False
    ):
        raise TrainerError(
            "resolved execution predates exact per-step loss logging; regenerate the RunPlan"
        )
    package_versions = {
        item.name: item.version
        for item in execution.trainer_interface.package_versions
        if item.version is not None
    }
    device_map = {item.module: item.device for item in execution.device_map}
    return TrainRunConfig(
        base_model=execution.inputs.model.location,
        model_revision=execution.inputs.model.resolved_revision,
        tokenizer_revision=execution.inputs.tokenizer.resolved_revision,
        tokenizer_location=execution.inputs.tokenizer.location,
        model_source=execution.inputs.model.source,
        tokenizer_source=execution.inputs.tokenizer.source,
        model_content_sha256=execution.inputs.model.content_sha256,
        tokenizer_content_sha256=execution.inputs.tokenizer.content_sha256,
        dataset_path=execution.inputs.dataset.location,
        dataset_sha256=execution.inputs.dataset.content_sha256,
        execution_configuration_hash=execution.configuration_hash,
        output_dir=execution.output_dir,
        dataset_format=execution.data.dataset_format,
        sequence_len=execution.sequence.max_sequence_len,
        lora_r=execution.adapter.lora_r,
        lora_alpha=execution.adapter.lora_alpha,
        lora_dropout=execution.adapter.lora_dropout,
        lora_bias=execution.adapter.bias,
        lora_target_modules=execution.adapter.target_modules,
        micro_batch_size=execution.batching.micro_batch_size,
        gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps,
        learning_rate=execution.optimizer.learning_rate,
        weight_decay=execution.optimizer.weight_decay,
        adam_beta1=execution.optimizer.adam_beta1,
        adam_beta2=execution.optimizer.adam_beta2,
        adam_epsilon=execution.optimizer.adam_epsilon,
        max_grad_norm=execution.optimizer.max_grad_norm,
        lr_scheduler=execution.optimizer.lr_scheduler,
        warmup_ratio=execution.optimizer.warmup_ratio,
        seed=execution.seed,
        data_seed=execution.data_seed,
        cpu_toy=execution.runtime_mode == "cpu_toy",
        max_steps=execution.schedule.max_steps,
        num_train_epochs=execution.schedule.num_train_epochs,
        attn_implementation=execution.attention.model_attention_api.value,
        attention_kernel=execution.attention.effective_backend_required.value,
        flash_sdp_enabled=execution.attention.flash_sdp_enabled,
        mem_efficient_sdp_enabled=execution.attention.mem_efficient_sdp_enabled,
        math_sdp_enabled=execution.attention.math_sdp_enabled,
        quantization_mode=execution.precision.quantized_storage_format.value,
        weight_storage_dtype=(
            execution.precision.weight_storage_dtype.value
            if execution.precision.weight_storage_dtype is not None
            else None
        ),
        dequantization_dtype=execution.precision.dequantization_dtype.value,
        forward_compute_dtype=execution.precision.forward_compute_dtype.value,
        gradient_dtype=execution.precision.gradient_dtype.value,
        optimizer_state_dtype=execution.precision.optimizer_state_dtype.value,
        optimizer_auxiliary_dtype=execution.precision.optimizer_auxiliary_dtype.value,
        master_weight_dtype=(
            execution.precision.master_weight_dtype.value
            if execution.precision.master_weight_dtype is not None
            else None
        ),
        device_map=device_map,
        trust_remote_code=execution.trust_remote_code,
        use_safetensors=execution.use_safetensors,
        bnb_4bit_use_double_quant=execution.bnb_4bit_use_double_quant,
        adapter_task_type=execution.adapter_task_type,
        export_format=execution.export_format.value,
        optim=execution.optimizer.impl.value,
        use_liger=execution.loss_impl.value == "liger_fused_ce",
        gradient_checkpointing=execution.gradient_checkpointing,
        save_steps=execution.checkpoint_policy.cadence_optimizer_steps,
        save_total_limit=execution.checkpoint_policy.keep_last,
        save_strategy=execution.save_strategy,
        logging_strategy=execution.trainer_interface.logging_strategy,
        logging_steps=execution.trainer_interface.logging_steps,
        logging_nan_inf_filter=execution.trainer_interface.logging_nan_inf_filter,
        report_to=execution.trainer_interface.report_to,
        disable_tqdm=execution.trainer_interface.disable_tqdm,
        packing=execution.data.packing,
        dataset_text_field=execution.data.dataset_text_field,
        truncation_allowed=execution.data.truncation_policy == "allow",
        formatter_id=execution.data.formatter_id,
        formatter_sha256=execution.data.formatter_sha256,
        chat_template_sha256=execution.data.chat_template_sha256,
        package_versions=package_versions,
        required_sft_config_fields=execution.trainer_interface.required_sft_config_fields,
        sequence_length_field=execution.trainer_interface.sequence_length_field,
        tokenizer_parameter=execution.trainer_interface.tokenizer_parameter,
    )


def _parse_config_text(text: str) -> dict[str, Any]:
    """Parse a training config that may be JSON **or** YAML. The ``corpus_studio`` target renders JSON,
    but a user can point ``train-run`` at a YAML config (e.g. one named ``*.yaml``, or an axolotl-style
    file) — so a valid config never dies on format. Tries JSON first (fast, no dep); on failure falls
    back to YAML (PyYAML ships with the ``[train]`` deps that ``train-run`` needs anyway)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # noqa: PLC0415 - lazy; PyYAML is a transitive [train] dep (transformers/datasets).
        except ImportError as exc:
            raise TrainerError(
                "The training config is not valid JSON, and PyYAML isn't available to parse it as YAML. "
                "Use a JSON config (the corpus_studio target emits one) or install pyyaml."
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TrainerError(f"The training config is neither valid JSON nor valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise TrainerError("The training config must be a JSON object / YAML mapping of fields.")
    return data


def load_run_config_from_file(
    config_path: Path | str,
    *,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    base_model: str | None = None,
    cpu_toy: bool = False,
    max_steps: int | None = None,
    attn_implementation: str | None = None,
    optim: str | None = None,
    use_liger: bool | None = None,
) -> TrainRunConfig:
    """Build a :class:`TrainRunConfig` from a CorpusStudio training config (the ``training-config``
    output: base_model / dataset_path / format / sequence_len / lora_* / seed …), applying overrides.
    Accepts JSON or YAML so a config never dies on format.

    The CPU toy path forces a tiny model + a short sequence + a few steps so it runs in seconds,
    unless the caller overrides them explicitly."""
    data: dict[str, Any] = _parse_config_text(Path(config_path).read_text(encoding="utf-8-sig"))

    resolved_base = base_model or data.get("base_model") or ""
    resolved_dataset = dataset_path or data.get("dataset_path") or ""
    seq = int(data.get("sequence_len", 4096))
    steps = max_steps

    if cpu_toy:
        resolved_base = base_model or TINY_TOY_MODEL
        seq = min(seq, 128)
        steps = max_steps if max_steps is not None else 3

    if not resolved_base:
        raise TrainerError("No base model: pass --base-model or a config with 'base_model'.")
    if not resolved_dataset:
        raise TrainerError("No dataset: pass --dataset-path or a config with 'dataset_path'.")

    return TrainRunConfig(
        base_model=resolved_base,
        dataset_path=resolved_dataset,
        output_dir=output_dir or data.get("output_dir", "output"),
        dataset_format=str(data.get("format", "instruction")),
        sequence_len=seq,
        lora_r=int(data.get("lora_r", 16)),
        lora_alpha=int(data.get("lora_alpha", 32)),
        micro_batch_size=int(data.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 8)),
        learning_rate=float(data.get("learning_rate", 2e-4)),
        seed=int(data.get("seed", 42)),
        data_seed=int(data.get("data_seed", data.get("seed", 42))),
        cpu_toy=cpu_toy,
        max_steps=steps,
        attn_implementation=attn_implementation or data.get("attn_implementation"),
        optim=optim or str(data.get("optim", "adamw_torch")),
        use_liger=use_liger if use_liger is not None else bool(data.get("use_liger", False)),
        save_steps=data.get("save_steps"),
        save_total_limit=data.get("save_total_limit"),
        save_strategy=cast(Literal["no", "steps"], data.get("save_strategy", "no")),
    )


def verify_sealed_runtime(
    config: TrainRunConfig,
    *,
    dataset_progress_callback: Callable[[int, int], None] | None = None,
) -> bytes | None:
    """Fail closed on package or dataset drift and return the exact stable dataset bytes.

    A sealed caller parses these returned bytes instead of reopening the path. This preserves the
    before/open/after file-identity checks while eliminating the prior second full read and hash.
    Unsealed compatibility execution has no pinned dataset identity and returns ``None``.
    """

    if config.execution_configuration_hash is None:
        return None
    for package, expected in sorted(config.package_versions.items()):
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TrainerEnvironmentError(
                f"sealed package is missing at execution: {package}=={expected}"
            ) from exc
        if observed != expected:
            raise TrainerEnvironmentError(
                f"sealed package drift: {package} expected {expected}, observed {observed}"
            )
    if config.dataset_sha256 is None:
        raise TrainerError("sealed execution omitted the dataset content digest")
    from corpus_studio.platform.execution_config import (  # noqa: PLC0415
        ExecutionConfigurationError,
        stable_file_bytes,
    )

    try:
        dataset_bytes, observed_dataset = stable_file_bytes(
            config.dataset_path,
            progress_callback=dataset_progress_callback,
        )
    except ExecutionConfigurationError as exc:
        raise TrainerError(str(exc)) from exc
    if observed_dataset != config.dataset_sha256:
        raise TrainerError("dataset bytes changed after the execution configuration was sealed")
    return dataset_bytes


def verify_local_inputs_after_load(config: TrainRunConfig) -> None:
    """Detect local model/tokenizer mutation across the third-party loading window."""

    if config.execution_configuration_hash is None:
        return
    from corpus_studio.platform.execution_config import (  # noqa: PLC0415
        ExecutionConfigurationError,
        stable_directory_sha256,
    )

    targets: dict[str, str] = {}
    for label, source, location, expected in (
        ("model", config.model_source, config.base_model, config.model_content_sha256),
        (
            "tokenizer",
            config.tokenizer_source,
            config.tokenizer_location or config.base_model,
            config.tokenizer_content_sha256,
        ),
    ):
        if source == "huggingface":
            continue
        if source != "local_directory" or expected is None:
            raise TrainerError(f"sealed {label} input has an unsupported local binding")
        previous = targets.setdefault(location, expected)
        if previous != expected:
            raise TrainerError("model and tokenizer bindings disagree for the same local directory")
    for location, expected in targets.items():
        try:
            observed = stable_directory_sha256(location)
        except ExecutionConfigurationError as exc:
            raise TrainerError(str(exc)) from exc
        if observed != expected:
            raise TrainerError(f"local model/tokenizer bytes changed while loading: {location}")


def _torch_dtype(torch_module: Any, name: str) -> Any:
    mapping = {
        "fp32": "float32",
        "tf32": "float32",
        "fp16": "float16",
        "mixed_fp16": "float16",
        "bf16": "bfloat16",
        "mixed_bf16": "bfloat16",
    }
    attribute = mapping.get(name)
    if attribute is None or not hasattr(torch_module, attribute):
        raise TrainerEnvironmentError(
            f"the sealed tensor dtype {name!r} is not supported by this torch build"
        )
    return getattr(torch_module, attribute)


def apply_attention_execution_policy(torch_module: Any, config: TrainRunConfig) -> str:
    """Apply all three SDP toggles exactly and verify the observed global state."""

    if config.execution_configuration_hash is None:
        return config.attn_implementation or "auto"
    requested = (
        config.flash_sdp_enabled,
        config.mem_efficient_sdp_enabled,
        config.math_sdp_enabled,
    )
    if any(value is None for value in requested) or config.attention_kernel is None:
        raise TrainerError("sealed execution omitted the exact attention-kernel policy")
    backend = getattr(torch_module, "backends", None)
    cuda = getattr(backend, "cuda", None)
    required_methods = (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
    )
    if cuda is None or any(not hasattr(cuda, name) for name in required_methods):
        raise TrainerEnvironmentError(
            "this torch build cannot enforce and observe all SDP kernel toggles"
        )
    try:
        cuda.enable_flash_sdp(bool(config.flash_sdp_enabled))
        cuda.enable_mem_efficient_sdp(bool(config.mem_efficient_sdp_enabled))
        cuda.enable_math_sdp(bool(config.math_sdp_enabled))
    except Exception as exc:  # noqa: BLE001 - normalize torch/driver enforcement failures.
        raise TrainerEnvironmentError(
            f"the sealed SDP kernel policy could not be applied or observed: {exc}"
        ) from exc
    verify_attention_execution_policy(torch_module, config)
    return config.attention_kernel


def verify_attention_execution_policy(torch_module: Any, config: TrainRunConfig) -> str:
    """Observe the three global SDP toggles without correcting a mid-execution deviation."""

    requested = (
        config.flash_sdp_enabled,
        config.mem_efficient_sdp_enabled,
        config.math_sdp_enabled,
    )
    backend = getattr(torch_module, "backends", None)
    cuda = getattr(backend, "cuda", None)
    if cuda is None:
        raise TrainerEnvironmentError("this torch build cannot observe SDP kernel toggles")
    try:
        observed = (
            bool(cuda.flash_sdp_enabled()),
            bool(cuda.mem_efficient_sdp_enabled()),
            bool(cuda.math_sdp_enabled()),
        )
    except Exception as exc:  # noqa: BLE001 - normalize torch/driver observation failures.
        raise TrainerEnvironmentError(
            f"the sealed SDP kernel policy could not be observed: {exc}"
        ) from exc
    if observed != requested:
        raise TrainerEnvironmentError(
            f"attention policy deviation: requested SDP toggles {requested}, observed {observed}"
        )
    return config.attention_kernel or "auto"


@contextlib.contextmanager
def enforced_attention_training_kernel(
    torch_module: Any,
    config: TrainRunConfig,
    *,
    kernel_context_factory: Callable[[list[Any]], Any] | None = None,
    backend_values: Mapping[str, Any] | None = None,
):
    """Keep the sealed torch-SDPA backend exclusive for the complete train call."""

    if config.execution_configuration_hash is None or config.attention_kernel in {
        "eager",
        "flash_attention_2",
        "flash_attention_3",
        "xformers",
    }:
        yield
        return
    kernel_name = config.attention_kernel
    if kernel_name is None:  # pragma: no cover - sealed config validation rejects this.
        raise TrainerError("sealed attention kernel is missing")
    if kernel_context_factory is None or backend_values is None:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - normalize an incompatible torch build.
            raise TrainerEnvironmentError(
                "this torch build cannot provide an exclusive SDPA training context"
            ) from exc
        kernel_context_factory = sdpa_kernel
        backend_values = {
            "torch_sdpa_math": SDPBackend.MATH,
            "torch_sdpa_flash": SDPBackend.FLASH_ATTENTION,
            "torch_sdpa_mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        }
    try:
        selected_backend = backend_values[kernel_name]
        context = kernel_context_factory([selected_backend])
        stack = contextlib.ExitStack()
        stack.enter_context(context)
    except Exception as exc:  # noqa: BLE001 - normalize context construction/entry failures.
        raise TrainerEnvironmentError(
            f"the sealed SDPA training context could not be entered: {exc}"
        ) from exc
    try:
        apply_attention_execution_policy(torch_module, config)
        yield
        verify_attention_execution_policy(torch_module, config)
    finally:
        # If the training body already raised, that workload failure is the primary truth. A cleanup
        # or seal-restoration error must not replace its taxonomy/stage, so subordinate any exception
        # raised while a workload failure is already propagating (torch's SDPA host-side toggles do
        # not normally fail, but a masked GRADIENT/OPTIMIZER failure would be reported as an
        # environment error). On a clean run, a restoration failure still surfaces.
        primary_failure = sys.exc_info()[1]
        try:
            stack.close()
        except Exception:  # noqa: BLE001 - normalize SDPA context-exit failures.
            if primary_failure is None:
                raise
        try:
            # sdpa_kernel restores the prior process-global state. Reassert the seal so later export
            # and any subsequent operation cannot inherit an unsealed fallback policy.
            apply_attention_execution_policy(torch_module, config)
        except Exception:  # noqa: BLE001 - a restoration failure must not mask the workload failure.
            if primary_failure is None:
                raise


def probe_effective_attention_kernel(torch_module: Any, config: TrainRunConfig) -> None:
    """Run one tiny forward/backward with only the sealed SDPA kernel permitted."""

    if config.execution_configuration_hash is None or config.attention_kernel in {
        "eager",
        "flash_attention_2",
        "flash_attention_3",
        "xformers",
    }:
        return
    if not torch_module.cuda.is_available():
        raise TrainerEnvironmentError(
            "the sealed GPU attention policy cannot be probed without CUDA"
        )
    kernel_name = config.attention_kernel
    if kernel_name is None:  # narrowed for type checkers; sealed validation rejects this above.
        raise TrainerError("sealed attention kernel is missing")
    try:
        import torch.nn.functional as functional  # noqa: PLC0415
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

        backend = {
            "torch_sdpa_math": SDPBackend.MATH,
            "torch_sdpa_flash": SDPBackend.FLASH_ATTENTION,
            "torch_sdpa_mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        }[kernel_name]
        dtype = _torch_dtype(torch_module, config.forward_compute_dtype)
        q = torch_module.randn(1, 2, 8, 16, device="cuda", dtype=dtype, requires_grad=True)
        k = torch_module.randn(1, 2, 8, 16, device="cuda", dtype=dtype, requires_grad=True)
        v = torch_module.randn(1, 2, 8, 16, device="cuda", dtype=dtype, requires_grad=True)
        with sdpa_kernel([backend]):
            out = functional.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
    except Exception as exc:  # noqa: BLE001 - normalize framework/kernel failures.
        raise TrainerEnvironmentError(
            f"the sealed attention kernel {config.attention_kernel!r} failed its runtime probe: {exc}"
        ) from exc


def build_model_load_kwargs(
    config: TrainRunConfig,
    torch_module: Any,
    *,
    quantize: bool,
    bitsandbytes_config_cls: Any | None = None,
) -> dict[str, Any]:
    """Build the exact ``from_pretrained`` arguments; sealed placement is never ``auto``."""

    kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if config.model_revision is not None:
        kwargs["revision"] = config.model_revision
    if config.execution_configuration_hash is not None:
        kwargs["use_safetensors"] = config.use_safetensors
    if quantize:
        if bitsandbytes_config_cls is None:
            raise TrainerError("quantized execution requires BitsAndBytesConfig")
        quantization_mode = config.quantization_mode or "nf4"
        if quantization_mode != "nf4":
            raise TrainerError(
                f"the first-party resolved executor does not implement {quantization_mode!r} weights"
            )
        kwargs["quantization_config"] = bitsandbytes_config_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization_mode,
            bnb_4bit_compute_dtype=_torch_dtype(torch_module, config.dequantization_dtype),
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
    else:
        kwargs["torch_dtype"] = _torch_dtype(
            torch_module,
            config.weight_storage_dtype or "fp32",
        )
    if config.execution_configuration_hash is not None:
        if config.device_map is None or "auto" in config.device_map.values():
            raise TrainerError("sealed execution requires an explicit non-auto device map")
        kwargs["device_map"] = config.device_map
        if config.attn_implementation is None:
            raise TrainerError("sealed execution omitted the model attention API")
        kwargs["attn_implementation"] = config.attn_implementation
    return kwargs


def _normalized_device(value: Any) -> str:
    if isinstance(value, bool):
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: boolean values are not device identities"
        )
    if isinstance(value, int):
        if value < 0:
            raise ExecutionPlacementDeviation(
                "PLACEMENT_DEVIATION: a CUDA device index cannot be negative"
            )
        return f"cuda:{value}"
    if isinstance(value, str):
        if value == "cuda":
            return "cuda:0"
        if re.fullmatch(r"cuda:[0-9]+", value):
            return value
        if value in {"cpu", "cpu:0"}:
            return "cpu"
        if value in {"meta", "disk"}:
            return value
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: device string is malformed or unsupported"
        )

    # Do not trust an arbitrary object's ``type``/``index`` attributes or a spoofed class name as
    # device evidence. Import torch only on this runtime path (never at module import) and require
    # exact identity with its extension type.
    value_type = type(value)
    try:
        import torch as torch_module  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - missing torch cannot prove an opaque device object.
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: unknown device object type "
            f"{value_type.__module__}.{value_type.__qualname__}; "
            "torch.device identity is unavailable"
        ) from exc
    if value_type is not torch_module.device:
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: unknown device object type "
            f"{value_type.__module__}.{value_type.__qualname__}"
        )
    device_type = getattr(value, "type", None)
    index = getattr(value, "index", None)
    if device_type == "cuda" and (index is None or type(index) is int):
        if index is not None and index < 0:
            raise ExecutionPlacementDeviation(
                "PLACEMENT_DEVIATION: a CUDA device index cannot be negative"
            )
        return f"cuda:{0 if index is None else index}"
    if device_type == "cpu" and index in {None, 0}:
        return "cpu"
    if device_type == "meta" and index is None:
        return "meta"
    raise ExecutionPlacementDeviation(
        "PLACEMENT_DEVIATION: torch.device value is malformed or unsupported"
    )


def _is_singleton_root_map(device_map: dict[str, str] | None) -> bool:
    """True when the sealed plan places the whole model on one root device via {\"\": device}."""

    return (
        isinstance(device_map, dict)
        and set(device_map.keys()) == {""}
        and bool(device_map[""])
        and device_map[""] != "auto"
    )


def _raw_hf_device_map(model: Any) -> dict[str, str] | None:
    raw = getattr(model, "hf_device_map", None)
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: hf_device_map is malformed"
        )
    if any(not isinstance(module, str) for module in raw):
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: hf_device_map contains a non-string module name"
        )
    return {module: _normalized_device(device) for module, device in raw.items()}


def _named_tensor_items(model: Any, method_name: str) -> list[tuple[str, Any]]:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: loaded model exposes no {method_name} inventory"
        )
    try:
        raw_items = method(remove_duplicate=False)
    except TypeError:
        # Older torch and dependency-light test doubles do not expose remove_duplicate. This fallback
        # still checks every item they expose; current torch uses the alias-preserving call above.
        raw_items = method()
    items: list[tuple[str, Any]] = []
    for item in raw_items:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not hasattr(item[1], "device")
        ):
            raise ExecutionPlacementDeviation(
                f"PLACEMENT_DEVIATION: {method_name} returned malformed state"
            )
        items.append((item[0], item[1]))
    return items


def _parameter_devices(model: Any) -> list[tuple[str, str]]:
    return [
        (name, _normalized_device(parameter.device))
        for name, parameter in _named_tensor_items(model, "named_parameters")
    ]


def _buffer_devices(model: Any) -> list[tuple[str, str]]:
    """Every registered buffer, including integer quantization and execution state."""

    return [
        (name, _normalized_device(buffer.device))
        for name, buffer in _named_tensor_items(model, "named_buffers")
    ]


def _named_modules(model: Any) -> list[tuple[str, Any]]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        return [("", model)]
    try:
        raw_modules = method(remove_duplicate=False)
    except TypeError:
        raw_modules = method()
    modules: list[tuple[str, Any]] = []
    for item in raw_modules:
        if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
            raise ExecutionPlacementDeviation(
                "PLACEMENT_DEVIATION: named_modules returned malformed state"
            )
        modules.append((item[0], item[1]))
    return modules or [("", model)]


def _expected_device_for_state(name: str, device_map: dict[str, str]) -> str | None:
    """Resolve a tensor/module name through the longest matching sealed device-map prefix."""

    matches = [
        (len(module_name), device)
        for module_name, device in device_map.items()
        if module_name == "" or name == module_name or name.startswith(f"{module_name}.")
    ]
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def _offload_hook_deviations(model: Any, expected_map: dict[str, str]) -> list[str]:
    """Return safe module labels for any hidden Accelerate/offload execution state."""

    deviations: list[str] = []
    seen_hooks: set[tuple[int, str]] = set()

    def _inspect_hook(hook: Any, module_name: str) -> None:
        hook_identity = (id(hook), module_name)
        if hook is None or hook_identity in seen_hooks:
            return
        # One hook object can be attached to more than one module. Inspect it once per attachment so
        # a shared execution_device cannot evade a different sealed-map prefix; the pair still
        # prevents a malformed SequentialHook cycle from recursing forever.
        seen_hooks.add(hook_identity)
        hook_runtime_type = type(hook)
        hook_type = hook_runtime_type.__name__
        label = module_name or "<root>"
        if hook_runtime_type.__module__ != "accelerate.hooks":
            deviations.append(
                f"{label}: unsupported _hf_hook runtime "
                f"{hook_runtime_type.__module__}.{hook_runtime_type.__qualname__}"
            )
            return
        if hook_type == "SequentialHook":
            nested = getattr(hook, "hooks", None)
            if not isinstance(nested, (list, tuple)) or not nested:
                deviations.append(f"{label}: malformed SequentialHook")
                return
            for child in nested:
                _inspect_hook(child, module_name)
            return
        if hook_type != "AlignDevicesHook":
            deviations.append(f"{label}: unsupported _hf_hook type {hook_type}")
            return
        offload = getattr(hook, "offload", False)
        if isinstance(offload, dict):
            offload_enabled = any(value is not False for value in offload.values())
        else:
            offload_enabled = offload is not False and offload is not None
        if offload_enabled:
            deviations.append(f"{label}: Accelerate offload hook")
        if getattr(hook, "weights_map", None) is not None:
            deviations.append(f"{label}: Accelerate weights_map")
        if getattr(hook, "offload_buffers", False) is not False:
            deviations.append(f"{label}: Accelerate buffer offload")
        for attribute in (
            "original_devices",
            "param_original_devices",
            "buffer_original_devices",
        ):
            original = getattr(hook, attribute, None)
            if original:
                deviations.append(f"{label}: Accelerate {attribute}")
        execution_device = getattr(hook, "execution_device", None)
        if execution_device is not None:
            if isinstance(execution_device, dict):
                values = tuple(execution_device.values())
                if not values:
                    deviations.append(f"{label}: hook execution-device map is empty")
                    return
            else:
                values = (execution_device,)
            expected_device = _expected_device_for_state(module_name, expected_map)
            if expected_device is None or any(
                _normalized_device(device) != expected_device for device in values
            ):
                deviations.append(f"{label}: hook execution device differs")

    for module_name, module in _named_modules(model):
        _inspect_hook(getattr(module, "_hf_hook", None), module_name)
        label = module_name or "<root>"
        for attribute in (
            "weights_map",
            "_weights_map",
            "offload_index",
            "_offload_index",
            "offload_dir",
            "_offload_dir",
        ):
            if getattr(module, attribute, None) is not None:
                deviations.append(f"{label}: {attribute} is populated")
        for attribute in ("offload", "offload_buffers", "disk_offload"):
            value = getattr(module, attribute, None)
            if value is not None and value is not False and not callable(value):
                deviations.append(f"{label}: {attribute} is enabled")
    return sorted(set(deviations))


def _verify_singleton_root_placement(
    model: Any,
    *,
    expected_device: str,
    expected_map: dict[str, str],
) -> None:
    """Accept root, expanded, or missing hf_device_map only when placement is fully on expected_device.

    Preserves fail-closed behavior for CPU/disk/meta/other-GPU map entries and any parameter or
    registered buffer off the sealed singleton device. Hidden Accelerate CPU/disk offload state is
    rejected even when the current parameter snapshot happens to be resident. Does not rewrite the
    sealed requested map and never treats "CUDA is available" as proof.
    """

    normalized_map = _raw_hf_device_map(model)
    parameter_devices = _parameter_devices(model)
    buffer_devices = _buffer_devices(model)

    if not parameter_devices:
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: sealed singleton placement cannot be verified - "
            "the loaded model exposes no parameters"
        )

    if normalized_map is not None:
        bad_map = {
            module: device
            for module, device in normalized_map.items()
            if device != expected_device
        }
        if bad_map:
            preview = ", ".join(f"{module}={device}" for module, device in list(bad_map.items())[:5])
            raise ExecutionPlacementDeviation(
                f"PLACEMENT_DEVIATION: requested device map {expected_map}, "
                f"observed hf_device_map entries outside {expected_device}: {preview}"
            )

    offload_deviations = _offload_hook_deviations(model, expected_map)
    if offload_deviations:
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: hidden offload or hook state: "
            + ", ".join(offload_deviations[:5])
        )

    bad_parameters = [
        (name, device) for name, device in parameter_devices if device != expected_device
    ]
    if bad_parameters:
        preview = ", ".join(
            f"{name}={device}" for name, device in bad_parameters[:5]
        )
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: parameters outside {expected_device}: {preview}"
        )

    bad_buffers = [(name, device) for name, device in buffer_devices if device != expected_device]
    if bad_buffers:
        preview = ", ".join(f"{name}={device}" for name, device in bad_buffers[:5])
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: execution-relevant buffers outside {expected_device}: {preview}"
        )
    # Success: sealed singleton root is realized. hf_device_map may be None, a root map, or an
    # expanded all-expected_device map; parameter+buffer inventory is the fail-closed authority.


def _verify_non_singleton_placement(
    model: Any,
    *,
    expected_map: dict[str, str],
) -> None:
    """Verify exact multi-entry map structure and every real tensor against its mapped device."""

    observed = _raw_hf_device_map(model)
    if observed != expected_map:
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: requested device map {expected_map}, observed {observed}"
        )
    parameter_devices = _parameter_devices(model)
    if not parameter_devices:
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: sealed non-singleton placement cannot be verified - "
            "the loaded model exposes no parameters"
        )
    offload_deviations = _offload_hook_deviations(model, expected_map)
    if offload_deviations:
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: hidden offload or hook state: "
            + ", ".join(offload_deviations[:5])
        )

    def _bad_state(items: list[tuple[str, str]]) -> list[tuple[str, str, str | None]]:
        return [
            (name, actual, expected)
            for name, actual in items
            if (expected := _expected_device_for_state(name, expected_map)) is None
            or actual != expected
        ]

    bad_parameters = _bad_state(parameter_devices)
    if bad_parameters:
        preview = ", ".join(
            f"{name}={actual} (expected {expected or 'covered map entry'})"
            for name, actual, expected in bad_parameters[:5]
        )
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: parameters disagree with the sealed device map: {preview}"
        )
    bad_buffers = _bad_state(_buffer_devices(model))
    if bad_buffers:
        preview = ", ".join(
            f"{name}={actual} (expected {expected or 'covered map entry'})"
            for name, actual, expected in bad_buffers[:5]
        )
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: buffers disagree with the sealed device map: {preview}"
        )



def verify_loaded_model_execution(model: Any, config: TrainRunConfig) -> None:
    """Observe the model API and placement chosen by Transformers/Accelerate, then fail closed.

    For a sealed singleton root map ``{\"\": \"cuda:0\"}`` (the production GPU plan shape), placement
    is verified semantically: every parameter and every registered buffer must
    resolve to the sealed device, and any ``hf_device_map`` entry must also resolve there. Expanded
    all-cuda maps and missing ``hf_device_map`` attributes (common with some bitsandbytes loads) are
    accepted only when that inventory is fully on the sealed device. Multi-key non-singleton sealed
    maps still require exact structural equality after device normalization.
    """

    if config.execution_configuration_hash is None:
        return
    model_config = getattr(model, "config", None)
    observed_attention = getattr(model_config, "_attn_implementation", None)
    if observed_attention != config.attn_implementation:
        raise TrainerError(
            "attention policy deviation: the loaded model reports "
            f"{observed_attention!r}, expected {config.attn_implementation!r}"
        )
    expected = config.device_map
    if expected is None:
        raise ExecutionPlacementDeviation("PLACEMENT_DEVIATION: sealed device map is missing")
    if config.cpu_toy:
        normalized_expected = {
            str(module): _normalized_device(device) for module, device in expected.items()
        }
        if not _is_singleton_root_map(normalized_expected):
            raise ExecutionPlacementDeviation(
                "PLACEMENT_DEVIATION: CPU toy execution requires one sealed root device"
            )
        _verify_singleton_root_placement(
            model,
            expected_device=normalized_expected[""],
            expected_map=normalized_expected,
        )
        return

    normalized_expected = {
        str(module): _normalized_device(device) for module, device in expected.items()
    }
    if _is_singleton_root_map(normalized_expected):
        expected_device = normalized_expected[""]
        _verify_singleton_root_placement(
            model,
            expected_device=expected_device,
            expected_map=normalized_expected,
        )
        return

    # Non-singleton maps remain structurally exact, but the map is not accepted as a substitute for
    # observing every real tensor. A matching all-CUDA map can still hide CPU/meta state otherwise.
    _verify_non_singleton_placement(model, expected_map=normalized_expected)


@dataclass(frozen=True)
class TrainableStateSnapshot:
    """Canonical logical bytes and shape/dtype inventory for every trainable tensor."""

    state_sha256: str
    tensor_sha256: dict[str, str]
    tensor_shapes: dict[str, tuple[int, ...]]
    tensor_dtypes: dict[str, str]


@dataclass(frozen=True)
class AdapterExportStateSnapshot:
    """Canonical PEFT export-key identity using the exact logical bytes saved by Safetensors."""

    state_sha256: str
    tensor_sha256: dict[str, str]
    tensor_shapes: dict[str, tuple[int, ...]]
    tensor_dtypes: dict[str, str]


class GradientObservationTracker:
    """Tracks which eligible adapter tensors ever materialized a verified leaf gradient."""

    def __init__(self, names: Sequence[str]) -> None:
        self.eligible_names = tuple(sorted(names))
        self.eligible_parameter_ids: dict[str, int] = {}
        self.observed_names: set[str] = set()

    def observe(self, name: str) -> None:
        self.observed_names.add(name)

    def verify_model_inventory(
        self,
        model: Any,
        *,
        stage: StageMarker = StageMarker.adapter_attached,
    ) -> None:
        """Refuse constructor-time additions or replacements that have no registered hook."""

        current = {
            name: id(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if tuple(sorted(current)) != self.eligible_names or (
            self.eligible_parameter_ids and current != self.eligible_parameter_ids
        ):
            raise TrainingEvidenceError(
                "trainable adapter inventory changed after gradient hooks were registered",
                taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                stage=stage,
                remediation="preserve the failed run and inspect trainer model wrapping",
            )

    def evidence(self) -> GradientCoverageEvidence:
        observed = sorted(self.observed_names)
        if not observed:
            raise TrainingEvidenceError(
                "no materialized adapter gradient was observed during training",
                taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                stage=StageMarker.backward,
                remediation="preserve the failed run and inspect the adapter targets and loss graph",
            )
        return GradientCoverageEvidence(
            eligible_tensor_count=len(self.eligible_names),
            eligible_tensor_names=list(self.eligible_names),
            observed_tensor_count=len(observed),
            observed_tensor_names=observed,
        )


def _logical_tensor_bytes(
    tensor: Any,
    torch_module: Any,
    *,
    stage: StageMarker = StageMarker.optimizer_step,
) -> bytes:
    """Copy a tensor's logical contiguous bytes to CPU without hashing unused storage."""

    try:
        value = tensor.detach().cpu().contiguous().view(torch_module.uint8)
        return bytes(value.numpy().tobytes())
    except Exception as exc:  # noqa: BLE001 - normalize framework-specific tensor failures.
        raise TrainingEvidenceError(
            "could not read canonical trainable tensor bytes",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=stage,
            remediation="preserve the failed run and inspect the adapter tensor representation",
        ) from exc


def _require_finite_tensor(
    tensor: Any,
    torch_module: Any,
    *,
    name: str,
    stage: StageMarker,
) -> None:
    try:
        observed = torch_module.isfinite(tensor.detach()).all()
        finite = bool(observed.item() if hasattr(observed, "item") else observed)
    except Exception as exc:  # noqa: BLE001 - absence is an evidence failure, never a pass.
        raise TrainingEvidenceError(
            f"could not verify finite adapter tensor state for {name}",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=stage,
            remediation="preserve the failed run and inspect the adapter tensor representation",
        ) from exc
    if not finite:
        raise TrainingEvidenceError(
            f"adapter tensor {name} contains non-finite values",
            taxonomy=FailureTaxonomy.NUMERICAL_FAILURE,
            stage=stage,
            remediation="preserve the failed run and inspect loss scaling, gradients, and optimizer state",
        )


def capture_trainable_state(
    model: Any,
    torch_module: Any,
    *,
    stage: StageMarker = StageMarker.optimizer_step,
) -> TrainableStateSnapshot:
    """Hash the sorted complete trainable-state inventory and exact logical tensor bytes."""

    tensor_sha256: dict[str, str] = {}
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    tensor_dtypes: dict[str, str] = {}
    canonical_entries: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in tensor_sha256:
            raise TrainingEvidenceError(
                f"duplicate trainable tensor name in adapter inventory: {name}",
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=stage,
            )
        _require_finite_tensor(parameter, torch_module, name=name, stage=stage)
        shape = tuple(int(value) for value in parameter.shape)
        dtype = str(parameter.dtype)
        digest = hashlib.sha256(
            _logical_tensor_bytes(parameter, torch_module, stage=stage)
        ).hexdigest()
        tensor_sha256[name] = digest
        tensor_shapes[name] = shape
        tensor_dtypes[name] = dtype
        canonical_entries.append(
            {"name": name, "shape": list(shape), "dtype": dtype, "content_sha256": digest}
        )
    if not canonical_entries:
        raise TrainingEvidenceError(
            "the sealed adapter configuration produced no trainable state to hash",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=stage,
        )
    canonical_entries.sort(key=lambda item: str(item["name"]))
    encoded = json.dumps(
        canonical_entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return TrainableStateSnapshot(
        state_sha256=hashlib.sha256(encoded).hexdigest(),
        tensor_sha256=tensor_sha256,
        tensor_shapes=tensor_shapes,
        tensor_dtypes=tensor_dtypes,
    )


def _safetensors_dtype_name(dtype: object) -> str:
    normalized = str(dtype).removeprefix("torch.")
    try:
        return {
            "bfloat16": "BF16",
            "float16": "F16",
            "float32": "F32",
            "float64": "F64",
            "int8": "I8",
            "int16": "I16",
            "int32": "I32",
            "int64": "I64",
            "uint8": "U8",
        }[normalized]
    except KeyError as exc:
        raise TrainingEvidenceError(
            f"adapter export tensor dtype {dtype!s} has no sealed Safetensors identity",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        ) from exc


def capture_adapter_export_state(
    state: Mapping[str, Any],
    torch_module: Any,
    *,
    stage: StageMarker,
) -> AdapterExportStateSnapshot:
    """Hash the exact PEFT state-dict keys and logical bytes used by save_pretrained."""

    from corpus_studio.platform.parameter_accounting import (  # noqa: PLC0415
        canonical_tensor_state_sha256,
    )

    records: list[dict[str, object]] = []
    tensor_sha256: dict[str, str] = {}
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    tensor_dtypes: dict[str, str] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not name or name in tensor_sha256:
            raise TrainingEvidenceError(
                "PEFT export state contains an invalid or duplicate tensor name",
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=stage,
            )
        _require_finite_tensor(tensor, torch_module, name=name, stage=stage)
        shape = tuple(int(value) for value in tensor.shape)
        dtype = _safetensors_dtype_name(tensor.dtype)
        digest = hashlib.sha256(
            _logical_tensor_bytes(tensor, torch_module, stage=stage)
        ).hexdigest()
        tensor_sha256[name] = digest
        tensor_shapes[name] = shape
        tensor_dtypes[name] = dtype
        records.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "content_sha256": digest,
            }
        )
    if not records:
        raise TrainingEvidenceError(
            "PEFT produced no adapter tensors for export",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=stage,
        )
    return AdapterExportStateSnapshot(
        state_sha256=canonical_tensor_state_sha256(records),
        tensor_sha256=tensor_sha256,
        tensor_shapes=tensor_shapes,
        tensor_dtypes=tensor_dtypes,
    )


def compare_adapter_export_states(
    before: AdapterExportStateSnapshot,
    after: AdapterExportStateSnapshot,
    *,
    adapter_config_semantic_sha256: str,
) -> AdapterExportStateEvidence:
    if (
        before.tensor_shapes != after.tensor_shapes
        or before.tensor_dtypes != after.tensor_dtypes
        or set(before.tensor_sha256) != set(after.tensor_sha256)
    ):
        raise TrainingEvidenceError(
            "adapter export inventory changed shape, dtype, or membership during training",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=StageMarker.optimizer_step,
        )
    changed = sorted(
        name
        for name, digest in before.tensor_sha256.items()
        if after.tensor_sha256[name] != digest
    )
    if not changed or before.state_sha256 == after.state_sha256:
        raise TrainingEvidenceError(
            "the canonical PEFT export state did not change during training",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=StageMarker.optimizer_step,
        )
    return AdapterExportStateEvidence(
        before_sha256=before.state_sha256,
        after_sha256=after.state_sha256,
        tensor_count=len(after.tensor_sha256),
        tensor_names=sorted(after.tensor_sha256),
        changed_tensor_count=len(changed),
        changed_tensor_names=changed,
        adapter_config_semantic_sha256=adapter_config_semantic_sha256,
    )


def compare_trainable_states(
    before: TrainableStateSnapshot,
    after: TrainableStateSnapshot,
) -> TrainableStateChangeEvidence:
    """Require one stable trainable inventory and at least one exact byte change."""

    if (
        before.tensor_shapes != after.tensor_shapes
        or before.tensor_dtypes != after.tensor_dtypes
        or set(before.tensor_sha256) != set(after.tensor_sha256)
    ):
        raise TrainingEvidenceError(
            "trainable adapter inventory changed shape, dtype, or membership during training",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=StageMarker.optimizer_step,
        )
    changed = sorted(
        name
        for name, digest in before.tensor_sha256.items()
        if after.tensor_sha256[name] != digest
    )
    if not changed or before.state_sha256 == after.state_sha256:
        raise TrainingEvidenceError(
            "completed optimizer steps did not change any trainable adapter tensor",
            taxonomy=FailureTaxonomy.UPDATE_FAILURE,
            stage=StageMarker.optimizer_step,
            remediation="preserve the failed run and inspect gradients, optimizer wiring, and targets",
        )
    return TrainableStateChangeEvidence(
        before_sha256=before.state_sha256,
        after_sha256=after.state_sha256,
        trainable_tensor_count=len(before.tensor_sha256),
        trainable_tensor_names=sorted(before.tensor_sha256),
        changed_tensor_count=len(changed),
        changed_tensor_names=changed,
    )


def enforce_trainable_precision(
    model: Any,
    torch_module: Any,
    config: TrainRunConfig,
) -> GradientObservationTracker | None:
    """Put adapter weights in the sealed master dtype and guard materialized gradients."""

    if config.execution_configuration_hash is None:
        return None
    if config.master_weight_dtype is None:
        raise TrainingEvidenceError(
            "sealed execution omitted the trainable master-weight dtype",
            taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
            stage=StageMarker.adapter_attached,
        )
    master_dtype = _torch_dtype(torch_module, config.master_weight_dtype)
    gradient_dtype = _torch_dtype(torch_module, config.gradient_dtype)
    expected_map = config.device_map or {}
    expected_device = _normalized_device(expected_map.get("", ""))
    trainable: list[str] = []
    tracker = GradientObservationTracker(trainable)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter.data = parameter.data.to(dtype=master_dtype)
        if parameter.dtype != master_dtype:
            raise TrainingEvidenceError(
                f"could not enforce master-weight dtype for {name}",
                taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                stage=StageMarker.adapter_attached,
            )

        register_post_accumulate = getattr(
            parameter, "register_post_accumulate_grad_hook", None
        )
        if not callable(register_post_accumulate):
            raise TrainingEvidenceError(
                "the training runtime cannot verify materialized adapter gradients after "
                f"accumulation: {name}",
                taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                stage=StageMarker.backward,
            )

        def _verify_gradient(
            materialized_parameter: Any,
            *,
            parameter_name: str = name,
            expected_parameter: Any = parameter,
        ) -> None:
            if materialized_parameter is not expected_parameter:
                raise TrainingEvidenceError(
                    f"post-accumulation gradient hook identity changed for {parameter_name}",
                    taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                    stage=StageMarker.backward,
                )
            gradient = getattr(expected_parameter, "grad", None)
            if gradient is None:
                raise TrainingEvidenceError(
                    f"materialized gradient is missing for {parameter_name}",
                    taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                    stage=StageMarker.backward,
                )
            if gradient.dtype != gradient_dtype:
                raise TrainingEvidenceError(
                    f"gradient dtype deviation for {parameter_name}: "
                    f"expected {gradient_dtype}, observed {gradient.dtype}",
                    taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                    stage=StageMarker.backward,
                )
            if _normalized_device(gradient.device) != expected_device:
                raise TrainingEvidenceError(
                    f"PLACEMENT_DEVIATION: gradient {parameter_name} is on {gradient.device}, "
                    f"expected {expected_device}",
                    taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                    stage=StageMarker.backward,
                )
            isfinite = getattr(torch_module, "isfinite", None)
            if callable(isfinite) and not bool(isfinite(gradient).all().item()):
                raise TrainingEvidenceError(
                    f"materialized gradient is non-finite for {parameter_name}",
                    taxonomy=FailureTaxonomy.NUMERICAL_FAILURE,
                    stage=StageMarker.backward,
                )
            tracker.observe(parameter_name)

        register_post_accumulate(_verify_gradient)
        trainable.append(name)
    if not trainable:
        raise TrainingEvidenceError(
            "the sealed adapter configuration produced no trainable parameters",
            taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
            stage=StageMarker.adapter_attached,
        )
    # ``tracker`` was constructed before iteration so hook closures share one stable object.  Bind
    # the final eligible inventory only after all hooks were registered successfully.
    tracker.eligible_names = tuple(sorted(trainable))
    tracker.eligible_parameter_ids = {
        name: id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return tracker


def reassert_trainable_precision(
    model: Any,
    torch_module: Any,
    config: TrainRunConfig,
    tracker: GradientObservationTracker,
) -> tuple[str, ...]:
    """Restore the sealed master dtype after trainer construction without replacing hooks.

    TRL may recast QLoRA adapter parameters while constructing ``SFTTrainer``.  The sealed executor
    owns the effective precision policy, so it must restore the exact master dtype after all trainer
    initialization mutations and before the first backward pass.  Parameter identity is checked both
    before and after the recast so an upstream wrapper cannot transfer an old hook observation to new
    trainable state.
    """

    if config.execution_configuration_hash is None:
        return ()
    tracker.verify_model_inventory(model, stage=StageMarker.adapter_attached)
    if config.master_weight_dtype is None:
        raise TrainingEvidenceError(
            "sealed execution omitted the trainable master-weight dtype",
            taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
            stage=StageMarker.adapter_attached,
        )
    master_dtype = _torch_dtype(torch_module, config.master_weight_dtype)
    restored: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != master_dtype:
            try:
                parameter.data = parameter.data.to(dtype=master_dtype)
            except Exception as exc:  # noqa: BLE001 - normalize framework tensor failures.
                raise TrainingEvidenceError(
                    f"could not restore sealed master-weight dtype for {name} after trainer "
                    "initialization",
                    taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                    stage=StageMarker.adapter_attached,
                ) from exc
            restored.append(name)
        if parameter.dtype != master_dtype:
            raise TrainingEvidenceError(
                f"could not restore sealed master-weight dtype for {name} after trainer "
                "initialization",
                taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                stage=StageMarker.adapter_attached,
            )
    tracker.verify_model_inventory(model, stage=StageMarker.adapter_attached)
    return tuple(sorted(restored))


def verify_model_state_execution(
    model: Any,
    torch_module: Any,
    config: TrainRunConfig,
    *,
    quantize: bool,
    linear4bit_type: type[Any] | None = None,
) -> None:
    """Observe post-PEFT placement plus frozen/trainable storage and dequantization formats."""

    if config.execution_configuration_hash is None:
        return
    expected_map = config.device_map or {}
    expected_device = _normalized_device(expected_map.get("", ""))
    offload_deviations = _offload_hook_deviations(model, {"": expected_device})
    if offload_deviations:
        raise ExecutionPlacementDeviation(
            "PLACEMENT_DEVIATION: post-adapter hidden offload or hook state: "
            + ", ".join(offload_deviations[:5])
        )
    misplaced = [
        name
        for name, parameter in model.named_parameters()
        if _normalized_device(parameter.device) != expected_device
    ]
    if misplaced:
        preview = ", ".join(misplaced[:5])
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: post-adapter parameters are outside {expected_device}: {preview}"
        )
    misplaced_buffers = [
        name for name, device in _buffer_devices(model) if device != expected_device
    ]
    if misplaced_buffers:
        preview = ", ".join(misplaced_buffers[:5])
        raise ExecutionPlacementDeviation(
            f"PLACEMENT_DEVIATION: post-adapter buffers are outside {expected_device}: {preview}"
        )
    if config.master_weight_dtype is None:
        raise TrainerError("sealed execution omitted the trainable master-weight dtype")
    master_dtype = _torch_dtype(torch_module, config.master_weight_dtype)
    trainable_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.requires_grad
    }
    if trainable_dtypes != {master_dtype}:
        raise TrainerError(
            f"trainable master-weight dtype deviation: observed {trainable_dtypes}, "
            f"expected {master_dtype}"
        )
    if quantize:
        if linear4bit_type is None:
            from bitsandbytes.nn import Linear4bit as BnbLinear4bit  # noqa: PLC0415

            linear4bit_type = BnbLinear4bit
        linear4bit = find_linear4bit_modules(model, linear4bit_type)
        if not linear4bit:
            raise TrainerError("sealed NF4 execution loaded no Linear4bit modules")
        quant_types = {
            getattr(
                getattr(getattr(module, "weight", None), "quant_state", None),
                "quant_type",
                None,
            )
            or getattr(getattr(module, "weight", None), "quant_type", None)
            for module in linear4bit
        }
        expected_quantization = config.quantization_mode
        if quant_types != {expected_quantization}:
            raise TrainerError(
                f"quantized storage deviation: observed {quant_types}, "
                f"expected {expected_quantization}"
            )
        compute_dtypes = {getattr(module, "compute_dtype", None) for module in linear4bit}
        if config.dequantization_dtype is None:
            raise TrainerError("sealed quantized execution omitted its dequantization dtype")
        expected_compute = _torch_dtype(torch_module, config.dequantization_dtype)
        if compute_dtypes != {expected_compute}:
            raise TrainerError(
                f"dequantization dtype deviation: observed {compute_dtypes}, "
                f"expected {expected_compute}"
            )
    else:
        if config.weight_storage_dtype is None:
            raise TrainerError("sealed unquantized execution omitted its weight-storage dtype")
        expected_storage = _torch_dtype(torch_module, config.weight_storage_dtype)
        frozen_dtypes = {
            parameter.dtype
            for parameter in model.parameters()
            if not parameter.requires_grad and parameter.is_floating_point()
        }
        if frozen_dtypes != {expected_storage}:
            raise TrainerError(
                f"base weight-storage dtype deviation: observed {frozen_dtypes}, "
                f"expected {expected_storage}"
            )


def verify_optimizer_state_precision(
    optimizer: Any,
    torch_module: Any,
    config: TrainRunConfig,
) -> None:
    """Verify materialized optimizer tensors against the sealed primary/auxiliary formats."""

    if config.execution_configuration_hash is None:
        return
    auxiliary = _torch_dtype(torch_module, config.optimizer_auxiliary_dtype)
    if config.optimizer_state_dtype == "int8":
        allowed = {torch_module.int8, torch_module.uint8, auxiliary}
    else:
        allowed = {_torch_dtype(torch_module, config.optimizer_state_dtype), auxiliary}
    expected_device = _normalized_device((config.device_map or {}).get("", ""))

    def _safe_state_name(value: object) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "?", str(value))[:80] or "?"

    def _materialized_values(
        value: Any,
        *,
        path: str,
        seen: set[int],
    ) -> list[tuple[str, Any]]:
        if hasattr(value, "dtype") or hasattr(value, "device"):
            return [(path, value)]
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen:
                return []
            seen.add(identity)
            found: list[tuple[str, Any]] = []
            for key, nested in value.items():
                found.extend(
                    _materialized_values(
                        nested,
                        path=f"{path}.{_safe_state_name(key)}",
                        seen=seen,
                    )
                )
            return found
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            identity = id(value)
            if identity in seen:
                return []
            seen.add(identity)
            found = []
            for index, nested in enumerate(value):
                found.extend(
                    _materialized_values(
                        nested,
                        path=f"{path}[{index}]",
                        seen=seen,
                    )
                )
            return found
        return []

    for parameter_state in optimizer.state.values():
        for name, value in _materialized_values(
            parameter_state,
            path="optimizer_state",
            seen=set(),
        ):
            if hasattr(value, "dtype") and value.dtype not in allowed:
                raise TrainingEvidenceError(
                    f"optimizer-state dtype deviation for {name}: "
                    f"expected one of {allowed}, observed {value.dtype}",
                    taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                    stage=StageMarker.optimizer_step,
                )
            if hasattr(value, "device"):
                observed_device = _normalized_device(value.device)
                # AdamW keeps its per-parameter ``step`` as a 0-dim scalar counter that, on the
                # default non-fused/non-capturable path, torch deliberately places on CPU even when
                # the parameters live on CUDA (torch.optim.adam._init_group: torch.tensor(0.0) with
                # no device). That bookkeeping counter is not numeric optimizer state, so it may sit
                # on the expected device OR on CPU. The real moment tensors (exp_avg/exp_avg_sq) are
                # never scalar, so a CPU-offloaded moment tensor is still rejected below.
                # torch.Size([]) (a 0-dim tensor's shape) compares equal to the empty tuple; a
                # missing or non-empty shape is treated as a real state tensor that must stay on the
                # expected device.
                is_scalar_counter = getattr(value, "shape", None) == ()
                permitted = {expected_device, "cpu"} if is_scalar_counter else {expected_device}
                if observed_device not in permitted:
                    raise TrainingEvidenceError(
                        f"PLACEMENT_DEVIATION: optimizer state {name} is on {value.device}, "
                        f"expected {expected_device}",
                        taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                        stage=StageMarker.optimizer_step,
                    )


def verify_completed_step_count(config: TrainRunConfig, steps: int) -> None:
    """Refuse a completed sealed artifact when the trainer exceeded or missed ``max_steps``."""

    if (
        config.execution_configuration_hash is not None
        and config.max_steps is not None
        and steps != config.max_steps
    ):
        raise TrainingEvidenceError(
            "sealed max_steps execution deviation: "
            f"expected {config.max_steps}, observed {steps}",
            taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
            stage=StageMarker.optimizer_step,
        )


class TrainingExecutionTracker:
    """Collect and reconcile optimizer, loss, gradient, and update evidence for one sealed run."""

    def __init__(
        self,
        *,
        config: TrainRunConfig,
        torch_module: Any,
        gradients: GradientObservationTracker,
        progress_callback: ProgressCallback | None,
        stage_callback: StageCallback | None,
    ) -> None:
        self.config = config
        self.torch_module = torch_module
        self.gradients = gradients
        self.progress_callback = progress_callback
        self.stage_callback = stage_callback
        self.optimizer: Any | None = None
        self.optimizer_parameter_ids: tuple[int, ...] = ()
        self.completed_steps: list[int] = []
        self.losses: dict[int, float] = {}

    def _bound_optimizer_parameter_ids(
        self,
        optimizer: Any,
        *,
        stage: StageMarker,
    ) -> tuple[int, ...]:
        param_groups = getattr(optimizer, "param_groups", None)
        if (
            not isinstance(param_groups, Sequence)
            or isinstance(param_groups, (str, bytes, bytearray))
            or not param_groups
            or not all(isinstance(group, Mapping) for group in param_groups)
        ):
            raise TrainingEvidenceError(
                "optimizer does not expose materialized parameter groups",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=stage,
            )
        parameters: list[Any] = []
        for group in param_groups:
            group_parameters = group.get("params")
            if not isinstance(group_parameters, Sequence) or isinstance(
                group_parameters, (str, bytes, bytearray)
            ):
                raise TrainingEvidenceError(
                    "optimizer parameter groups are not materialized sequences",
                    taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                    stage=stage,
                )
            parameters.extend(group_parameters)
        observed_ids = [id(parameter) for parameter in parameters]
        eligible_ids = sorted(self.gradients.eligible_parameter_ids.values())
        if (
            not eligible_ids
            or len(observed_ids) != len(set(observed_ids))
            or sorted(observed_ids) != eligible_ids
        ):
            raise TrainingEvidenceError(
                "optimizer parameters do not exactly match the complete trainable adapter inventory",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=stage,
                remediation="preserve the failed run and inspect Trainer optimizer construction",
            )
        return tuple(eligible_ids)

    def on_train_begin(self, optimizer: Any) -> None:
        if self.optimizer is not None:
            raise TrainingEvidenceError(
                "optimizer creation was reported more than once",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_created,
            )
        if (
            optimizer is None
            or not callable(getattr(optimizer, "step", None))
            or not callable(getattr(optimizer, "zero_grad", None))
        ):
            raise TrainingEvidenceError(
                "on_train_begin did not expose a real optimizer with materialized parameter groups",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_created,
                remediation="preserve the failed run and inspect Trainer optimizer construction",
            )
        optimizer_parameter_ids = self._bound_optimizer_parameter_ids(
            optimizer,
            stage=StageMarker.optimizer_created,
        )
        self.optimizer = optimizer
        self.optimizer_parameter_ids = optimizer_parameter_ids
        if self.stage_callback is not None:
            self.stage_callback(
                "optimizer_created",
                "observed the real optimizer at on_train_begin",
            )

    def on_step_end(self, step: int, optimizer: Any) -> None:
        if self.optimizer is None or optimizer is not self.optimizer:
            raise TrainingEvidenceError(
                "optimizer identity was absent or changed at optimizer-step completion",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_step,
            )
        if self._bound_optimizer_parameter_ids(
            optimizer,
            stage=StageMarker.optimizer_step,
        ) != self.optimizer_parameter_ids:
            raise TrainingEvidenceError(
                "optimizer parameter identity changed after creation",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_step,
            )
        expected = len(self.completed_steps) + 1
        if step != expected:
            raise TrainingEvidenceError(
                f"optimizer-step sequence deviation: expected {expected}, observed {step}",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_step,
            )
        verify_optimizer_state_precision(optimizer, self.torch_module, self.config)
        self.completed_steps.append(step)

    def on_log(self, step: int, total: int, logs: Mapping[str, Any] | None) -> None:
        if not logs or "loss" not in logs:
            return
        if step not in self.completed_steps:
            raise TrainingEvidenceError(
                f"loss record for optimizer step {step} arrived before that step completed",
                taxonomy=FailureTaxonomy.LOSS_EVIDENCE_FAILURE,
                stage=StageMarker.loss,
            )
        if step in self.losses:
            raise TrainingEvidenceError(
                f"duplicate loss record for optimizer step {step}",
                taxonomy=FailureTaxonomy.LOSS_EVIDENCE_FAILURE,
                stage=StageMarker.loss,
            )
        raw_loss = logs["loss"]
        if isinstance(raw_loss, bool) or not isinstance(raw_loss, Real):
            raise TrainingEvidenceError(
                f"loss record for optimizer step {step} is not numeric",
                taxonomy=FailureTaxonomy.LOSS_EVIDENCE_FAILURE,
                stage=StageMarker.loss,
            )
        loss = float(raw_loss)
        if not math.isfinite(loss):
            raise TrainingEvidenceError(
                f"loss record for optimizer step {step} is non-finite",
                taxonomy=FailureTaxonomy.NUMERICAL_FAILURE,
                stage=StageMarker.loss,
                remediation="preserve the failed run and inspect the batch, precision, and learning rate",
            )
        self.losses[step] = loss
        if self.progress_callback is not None:
            self.progress_callback(step, total, loss)

    def finalize(
        self,
        *,
        steps: int,
        before: TrainableStateSnapshot,
        before_export: AdapterExportStateSnapshot,
        after_export: AdapterExportStateSnapshot,
        adapter_config_semantic_sha256: str,
        model: Any,
    ) -> TrainingExecutionEvidence:
        verify_completed_step_count(self.config, steps)
        if self.optimizer is None:
            raise TrainingEvidenceError(
                "training completed without real optimizer-creation evidence",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_created,
            )
        expected = list(range(1, steps + 1))
        if self.completed_steps != expected:
            raise TrainingEvidenceError(
                "completed optimizer-step evidence does not match trainer global_step",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_step,
            )
        if sorted(self.losses) != expected:
            raise TrainingEvidenceError(
                "training did not emit exactly one finite loss for every completed optimizer step",
                taxonomy=FailureTaxonomy.LOSS_EVIDENCE_FAILURE,
                stage=StageMarker.loss,
                remediation=(
                    "keep logging_strategy=steps, logging_steps=1, and "
                    "logging_nan_inf_filter=false"
                ),
            )
        # Bind the after-state bytes to the same parameter objects that owned the registered
        # post-accumulation hooks. A framework must not be able to replace a parameter during
        # ``train()`` and reuse an old gradient observation under the replacement's name.
        self.gradients.verify_model_inventory(model, stage=StageMarker.optimizer_step)
        after = capture_trainable_state(
            model,
            self.torch_module,
            stage=StageMarker.optimizer_step,
        )
        return TrainingExecutionEvidence(
            trainable_state=compare_trainable_states(before, after),
            adapter_export_state=compare_adapter_export_states(
                before_export,
                after_export,
                adapter_config_semantic_sha256=adapter_config_semantic_sha256,
            ),
            gradient_coverage=self.gradients.evidence(),
            optimizer_created=True,
            completed_optimizer_steps=steps,
            step_losses=[
                OptimizerStepLossEvidence(optimizer_step=step, loss=self.losses[step])
                for step in expected
            ],
        )


def format_example_text(row: dict, dataset_format: str, tokenizer: Any | None = None) -> str:
    """Format one dataset row into a single training-text string.

    ``chat`` uses the model's chat template when a tokenizer is supplied (the correct format for a
    chat model), else a simple ``role: content`` join. ``instruction`` uses an Alpaca-style template.
    ``trace`` renders a reasoning trace (prompt + ``<think>reasoning</think>`` + answer — see
    ``training.traces``). Returns "" for an empty/unusable row (the caller drops it)."""
    if dataset_format == "trace":
        from corpus_studio.training.traces import format_trace, trace_from_row  # noqa: PLC0415

        return format_trace(trace_from_row(row), tokenizer=tokenizer)
    if dataset_format == "chat":
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return str(tokenizer.apply_chat_template(messages, tokenize=False))
            except Exception as exc:  # noqa: BLE001 - normalize third-party template exceptions.
                raise TrainerError(f"the tokenizer chat template failed: {exc}") from exc
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in messages if isinstance(m, dict)
        )

    instruction = str(row.get("instruction", "")).strip()
    extra_input = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    if not instruction and not output:
        return ""
    prompt = instruction + (f"\n\n{extra_input}" if extra_input else "")
    return f"### Instruction:\n{prompt}\n\n### Response:\n{output}"


def build_lora_kwargs(config: TrainRunConfig) -> dict[str, Any]:
    """peft ``LoraConfig`` kwargs. ``target_modules='all-linear'`` targets every linear layer, so it
    works across architectures (tiny Llama toy → real Qwen) without a per-model module list."""
    return {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": config.lora_bias,
        "task_type": config.adapter_task_type,
        "target_modules": (
            config.lora_target_modules[0]
            if config.lora_target_modules == ["all-linear"]
            else config.lora_target_modules
        ),
    }


def expected_saved_adapter_config_sha256(model: Any, config: TrainRunConfig) -> str:
    """Bind the complete in-memory PEFT config as save_pretrained will serialize it."""

    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or len(configs) != 1:
        raise TrainingEvidenceError(
            "the sealed run requires exactly one materialized PEFT adapter config",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    peft_config = next(iter(configs.values()))
    to_dict = getattr(peft_config, "to_dict", None)
    if not callable(to_dict):
        raise TrainingEvidenceError(
            "the materialized PEFT adapter config is not serializable",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        )
    try:
        payload = dict(to_dict())
    except Exception as exc:  # noqa: BLE001 - normalize PEFT serialization drift.
        raise TrainingEvidenceError(
            f"the PEFT adapter config could not be captured: {exc}",
            taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
            stage=StageMarker.export,
        ) from exc
    payload["inference_mode"] = True
    if payload.get("base_model_name_or_path") is None:
        payload["base_model_name_or_path"] = config.base_model
    from corpus_studio.platform.artifacts import (  # noqa: PLC0415
        canonical_adapter_config_sha256,
    )

    return canonical_adapter_config_sha256(payload)


def build_training_kwargs(config: TrainRunConfig) -> dict[str, Any]:
    """TRL ``SFTConfig`` kwargs from the run config, including the exact sealed save policy."""
    _require_checkpoint_free_execution(config)
    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "data_seed": config.data_seed,
        "weight_decay": config.weight_decay,
        "adam_beta1": config.adam_beta1,
        "adam_beta2": config.adam_beta2,
        "adam_epsilon": config.adam_epsilon,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler,
        "warmup_ratio": config.warmup_ratio,
        "logging_strategy": config.logging_strategy,
        "logging_steps": config.logging_steps,
        "logging_nan_inf_filter": config.logging_nan_inf_filter,
        "save_strategy": config.save_strategy,
        "report_to": config.report_to,
        "dataset_text_field": config.dataset_text_field,
        "disable_tqdm": config.disable_tqdm,
        "packing": config.packing,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
    sequence_field = (
        config.sequence_length_field
        if config.sequence_length_field != "auto"
        else "max_seq_length"
    )
    kwargs[sequence_field] = config.sequence_len
    if config.max_steps is not None:
        kwargs["max_steps"] = config.max_steps
    else:
        kwargs["num_train_epochs"] = config.num_train_epochs or 1
    if config.cpu_toy:
        # Force CPU + no half precision so the toy runs on a machine with no GPU. The paged optimizer
        # (bitsandbytes) and Liger (Triton) are CUDA-only, so the toy path never uses them — it would
        # crash on a GPU-less machine, defeating the point of the smoke test.
        kwargs["use_cpu"] = True
        kwargs["bf16"] = False
        kwargs["fp16"] = False
        kwargs["optim"] = "adamw_torch"
    else:
        kwargs["optim"] = config.optim
        kwargs["bf16"] = config.forward_compute_dtype in {"bf16", "mixed_bf16"}
        kwargs["fp16"] = config.forward_compute_dtype in {"fp16", "mixed_fp16"}
        if config.use_liger:
            kwargs["use_liger_kernel"] = True
    return kwargs


class TruncationReport(BaseModel):
    """How badly a ``sequence_len`` truncates a dataset — the guardrail against silently training on
    cut-off examples (a real WBG bug: seq_len 1536 truncated 100% of ~2.2k-token examples, cutting the
    end of every assistant/output and teaching the model to emit incomplete JSON)."""

    n_examples: int
    n_truncated: int
    pct_truncated: float
    seq_len: int
    max_tokens: int
    median_tokens: int
    # the smallest sequence_len that would truncate NOTHING (== the longest example)
    seq_len_for_zero_truncation: int

    @property
    def truncates(self) -> bool:
        return self.n_truncated > 0


def analyze_truncation(token_lengths: list[int], seq_len: int) -> TruncationReport:
    """PURE + unit-tested. Given tokenized example lengths + the training ``seq_len``, report how many
    are truncated (their tails — including the model's output — cut off) and the seq_len that keeps
    them whole."""
    lengths = sorted(token_lengths)
    n = len(lengths)
    n_trunc = sum(1 for x in lengths if x > seq_len)
    return TruncationReport(
        n_examples=n,
        n_truncated=n_trunc,
        pct_truncated=(100.0 * n_trunc / n) if n else 0.0,
        seq_len=seq_len,
        max_tokens=lengths[-1] if lengths else 0,
        median_tokens=lengths[n // 2] if lengths else 0,
        seq_len_for_zero_truncation=lengths[-1] if lengths else seq_len,
    )


def truncation_warning(report: TruncationReport) -> str | None:
    """The operator-facing warning for a truncating ``seq_len`` — or None when nothing is cut."""
    if not report.truncates:
        return None
    return (
        f"[WARNING] TRUNCATION: {report.n_truncated}/{report.n_examples} examples "
        f"({report.pct_truncated:.0f}%) exceed sequence_len={report.seq_len} and will be CUT — the end "
        f"of each (including the assistant/output) is lost, so the model learns incomplete outputs. "
        f"Longest example is {report.max_tokens} tokens; raise sequence_len to "
        f">= {report.seq_len_for_zero_truncation} to keep every example whole (costs more VRAM), or "
        "shorten/split the data."
    )


def _prepare_training_texts(
    rows: list[dict[str, Any]],
    config: TrainRunConfig,
    tokenizer: Any,
    *,
    stage_callback: StageCallback | None = None,
) -> tuple[list[str], TruncationReport]:
    """Format and tokenize every row with bounded, same-thread progress events.

    A callback fires only after actual rows complete. At most
    ``_MAX_PREFLIGHT_PROGRESS_EVENTS`` progress events are emitted per phase, so a hung formatter or
    tokenizer cannot be concealed by an independent liveness loop or unbounded event spam.
    """

    def _stage(name: str, message: str) -> None:
        if stage_callback is not None:
            stage_callback(name, message)

    def _interval(total: int) -> int:
        return max(
            1,
            (total + _MAX_PREFLIGHT_PROGRESS_EVENTS - 1)
            // _MAX_PREFLIGHT_PROGRESS_EVENTS,
        )

    total_rows = len(rows)
    formatting_interval = _interval(total_rows)
    texts: list[str] = []
    _stage("dataset_formatting", f"formatting {total_rows} sealed dataset rows")
    try:
        for index, row in enumerate(rows, start=1):
            texts.append(format_example_text(row, config.dataset_format, tokenizer))
            if index % formatting_interval == 0 or index == total_rows:
                _stage("dataset_formatting", f"formatted {index}/{total_rows} dataset rows")
    except TrainerError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize formatter failures.
        raise TrainerError(f"full-dataset formatting failed: {exc}") from exc
    _stage("dataset_formatting", f"formatted all {total_rows} dataset rows")

    rendered = [text for text in texts if text]
    rendered_count = len(rendered)
    truncation_interval = _interval(rendered_count)
    lengths: list[int] = []
    _stage(
        "truncation_analysis",
        f"tokenizing all {rendered_count} rendered rows for truncation analysis",
    )
    try:
        for index, text in enumerate(rendered, start=1):
            lengths.append(len(tokenizer(text)["input_ids"]))
            if index % truncation_interval == 0 or index == rendered_count:
                _stage(
                    "truncation_analysis",
                    f"tokenized {index}/{rendered_count} rendered rows",
                )
        report = analyze_truncation(lengths, config.sequence_len)
        warning = truncation_warning(report)
        if warning:
            if not config.truncation_allowed:
                raise TrainerError(warning.removeprefix("[WARNING] "))
            print(warning, file=sys.stderr)
    except TrainerError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize tokenizer failures.
        raise TrainerError(f"full-dataset truncation analysis failed: {exc}") from exc
    _stage(
        "truncation_analysis",
        f"verified {rendered_count} rendered rows; maximum {report.max_tokens} tokens",
    )
    return texts, report


def resolve_run_plan(config: TrainRunConfig, report: Any) -> dict[str, Any]:
    """Decide device + quantization from the runtime report, or raise a clean :class:`TrainerError`.

    CPU toy → CPU, no quantization (needs only ``cpu_toy_ready``). Real run → CUDA + 4-bit
    (needs the full ``ready`` set: deps + GPU + bitsandbytes)."""
    if config.cpu_toy:
        if not report.cpu_toy_ready:
            raise TrainerEnvironmentError(
                "CPU toy training needs torch + transformers + peft + trl + datasets + accelerate. "
                f"{INSTALL_HINT}"
            )
        return {"device": "cpu", "quantize": False}

    if not report.ready:
        raise TrainerEnvironmentError(
            "A real QLoRA run needs the full [train] runtime + a CUDA GPU + bitsandbytes. "
            f"Run 'corpus-studio train-check' to see what's missing. {INSTALL_HINT} "
            "(or pass --cpu-toy to smoke-test the pipeline on CPU)."
        )
    return {"device": "cuda", "quantize": True}


# Blackwell (RTX 50-series) is sm_120 → compute-capability major 12. The fused FLASH SDPA backward
# deadlocks there ONLY under the native-Windows WDDM driver model — the canonical rule lives in
# corpus_studio.platform.host_platform.flash_sdpa_deadlocks; kept as a plain check here so the trainer
# stays decoupled from the platform layer (which sits above it).
_BLACKWELL_CAPABILITY_MAJOR = 12


def resolve_attention_implementation(
    explicit: str | None,
    device_capability_major: int | None,
    *,
    native_windows: bool = False,
) -> tuple[str | None, bool]:
    """Decide the attention backend for ``from_pretrained``. PURE + unit-tested.

    Returns ``(attn_implementation, disable_flash_sdp)``:
    * an **explicit** choice (config / CLI ``--attn-implementation``) always wins, with no SDP toggling
      — the user owns it;
    * else on **native Windows (WDDM) + Blackwell** (sm_120) → ``(None, True)``: keep transformers'
      default SDPA but signal the caller to DISABLE the fused flash SDP backend — the only kernel that
      deadlocks there (mem-efficient + math are safe), so SDPA uses a non-deadlocking kernel;
    * else ``(None, False)`` - no forced change. This includes WSL and bare Linux because the known
      refusal is specific to Windows WDDM. Leaving a kernel enabled is not proof it works: the
      environment capability probe remains authoritative. WSL has separate passing evidence;
      bare-Linux RTX 5070 behavior is unverified.
    """
    if explicit:
        return explicit, False
    blackwell = device_capability_major is not None and device_capability_major >= _BLACKWELL_CAPABILITY_MAJOR
    if native_windows and blackwell:
        return None, True
    return None, False


def _list_checkpoints(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    return sorted(str(p) for p in output_dir.glob("checkpoint-*") if p.is_dir())


# The dependency-light unit gate deliberately omits torch/Transformers/PEFT/TRL.  This orchestration
# body is exercised only by the managed worker smoke/integration lane; its pure configuration,
# immutable-input, placement, precision, and policy helpers are unit-tested independently.  Keeping
# it out of unit-coverage accounting also avoids implying that a passing core gate verified training
# hardware or the optional stack.
def run_training(  # pragma: no cover - optional training-stack integration
    config: TrainRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    stage_callback: StageCallback | None = None,
    token_callback: TokenCallback | None = None,
) -> TrainResult:
    """Run the training. Lazy-imports the heavy stack; verified via the CPU toy path (a real GPU QLoRA
    can only be user-smoke-tested). Raises :class:`TrainerError` if the runtime can't run the request.
    ``stage_callback(name, message)`` fires for stable dataset verification, formatting, full-corpus
    tokenization, tokenizer/model-load boundaries, and later setup milestones. Row and byte progress
    is emitted synchronously by the work thread. Model loading has true start/end events and a bounded
    supervisor deadline; no independent heartbeat can make a stuck load look healthy.
    """

    def _stage(name: str, message: str) -> None:
        if stage_callback is not None:
            stage_callback(name, message)

    _require_checkpoint_free_execution(config)

    dataset_progress_bucket = 0

    def _dataset_progress(completed: int, total: int) -> None:
        nonlocal dataset_progress_bucket
        if total <= 0:
            return
        bucket = min(
            _MAX_PREFLIGHT_PROGRESS_EVENTS,
            max(1, completed * _MAX_PREFLIGHT_PROGRESS_EVENTS // total),
        )
        if bucket <= dataset_progress_bucket:
            return
        dataset_progress_bucket = bucket
        _stage(
            "dataset_verification",
            f"read and hashed {completed}/{total} sealed dataset bytes",
        )

    if config.execution_configuration_hash is not None:
        _stage("dataset_verification", "reading and hashing the sealed dataset once")
    dataset_bytes = verify_sealed_runtime(
        config,
        dataset_progress_callback=(
            _dataset_progress if config.execution_configuration_hash is not None else None
        ),
    )
    if config.execution_configuration_hash is not None and config.export_format != "adapter_peft":
        raise TrainerError("the first-party resolved executor can emit only a PEFT adapter")
    if config.execution_configuration_hash is not None:
        if dataset_bytes is None:  # pragma: no cover - sealed verification always returns bytes.
            raise TrainerError("sealed dataset verification returned no stable bytes")
        try:
            rows = list(read_jsonl_bytes(dataset_bytes))
        except ValueError as exc:
            raise TrainerError(f"sealed dataset is invalid: {exc}") from exc
        finally:
            del dataset_bytes
        _stage(
            "dataset_verification",
            f"verified and parsed {len(rows)} sealed dataset rows",
        )
        _stage(
            "execution_config_verified",
            f"verified resolved execution {config.execution_configuration_hash}",
        )
    else:
        rows = list(read_jsonl(Path(config.dataset_path)))
    if config.dataset_format == "trace":
        from corpus_studio.platform.trace_records import (  # noqa: PLC0415
            check_trace_dataset_for_training,
        )
        from corpus_studio.providers.overrides import load_overrides  # noqa: PLC0415

        trace_check = check_trace_dataset_for_training(
            rows,
            provider_overrides=load_overrides(Path(config.dataset_path).parent),
        )
        if not trace_check.ready:
            preview = "; ".join(trace_check.blocked[:10])
            remainder = len(trace_check.blocked) - min(len(trace_check.blocked), 10)
            suffix = f"; plus {remainder} more issue(s)" if remainder else ""
            raise TrainerError(f"Trace dataset is not training-ready: {preview}{suffix}")
        if trace_check.legacy_rows:
            print(
                f"[WARNING] {trace_check.legacy_rows} legacy trace row(s) are unsealed and have no "
                "review provenance; migrate and review them for the versioned safety gate.",
                file=sys.stderr,
            )
    runtime_report = probe_training_runtime()
    if config.execution_configuration_hash is not None:
        quantize = config.quantization_mode != "none"
        if config.cpu_toy:
            if not runtime_report.cpu_toy_ready:
                raise TrainerEnvironmentError(
                    "the sealed CPU-toy runtime packages are not usable in this worker"
                )
        else:
            if not runtime_report.gpu.available:
                raise TrainerEnvironmentError(
                    "the sealed CUDA device is not available in this worker"
                )
            if quantize and not runtime_report.bitsandbytes_ok:
                raise TrainerEnvironmentError(
                    "the sealed quantized path is not available in this worker"
                )
    else:
        plan = resolve_run_plan(config, runtime_report)
        quantize = bool(plan["quantize"])

    import torch  # noqa: PLC0415 - intentionally lazy heavy imports.
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    _stage("env_loaded", "loaded the optional training runtime")
    set_seed(config.seed)

    # SECURITY: never execute a downloaded model repo's custom code. trust_remote_code defaults False,
    # but we set it explicitly (defence-in-depth; guards against a future default change). A model that
    # genuinely needs custom code is a deliberate, separate decision — not something a fetched repo can
    # trigger silently.
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if config.tokenizer_revision is not None:
        tokenizer_kwargs["revision"] = config.tokenizer_revision
    _stage("tokenizer_load", "loading the sealed tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_location or config.base_model, **tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.execution_configuration_hash is not None:
        from corpus_studio.platform.execution_config import formatter_identity  # noqa: PLC0415

        expected_formatter = formatter_identity(config.dataset_format)
        if (config.formatter_id, config.formatter_sha256) != expected_formatter:
            raise TrainerError("the sealed formatter identity does not match this worker implementation")
        if config.dataset_format == "chat":
            template = getattr(tokenizer, "chat_template", None)
            if not isinstance(template, str) or not template:
                raise TrainerError("the pinned tokenizer has no usable chat template")
            if not callable(getattr(tokenizer, "apply_chat_template", None)):
                raise TrainerError("the pinned tokenizer cannot apply its chat template")
            observed_template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
            if observed_template_hash != config.chat_template_sha256:
                raise TrainerError("the tokenizer chat template changed after planning")
    _stage("tokenizer_load", "loaded and verified the sealed tokenizer")

    # Complete formatting and truncation analysis before allocating model weights. A sealed refusing
    # policy never spends GPU memory before this preflight has passed for every pinned row.
    texts, _truncation_report = _prepare_training_texts(
        rows,
        config,
        tokenizer,
        stage_callback=_stage,
    )
    del rows

    effective_kernel: str | None = None
    if config.execution_configuration_hash is not None:
        _stage(
            "attention_policy_applied",
            "applying the sealed SDP toggles before model allocation",
        )
        effective_kernel = apply_attention_execution_policy(torch, config)
        probe_effective_attention_kernel(torch, config)

    _stage("model_load", "loading the sealed model weights")
    if config.cpu_toy:
        # Build from config (random weights) — no weights download, so the smoke test runs offline.
        model_config_kwargs = dict(tokenizer_kwargs)
        if config.execution_configuration_hash is not None:
            model_config_kwargs["attn_implementation"] = config.attn_implementation
        model_config = AutoConfig.from_pretrained(config.base_model, **model_config_kwargs)
        if config.execution_configuration_hash is not None:
            model_config._attn_implementation = config.attn_implementation
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        bitsandbytes_config_cls = None
        if quantize:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            bitsandbytes_config_cls = BitsAndBytesConfig
        model_kwargs = build_model_load_kwargs(
            config,
            torch,
            quantize=quantize,
            bitsandbytes_config_cls=bitsandbytes_config_cls,
        )

        if config.execution_configuration_hash is not None:
            attn_impl = config.attn_implementation
        else:
            # Legacy, explicitly unsealed train-run compatibility path. New RunPlans never use it.
            try:
                capability_major = (
                    torch.cuda.get_device_capability()[0]
                    if torch.cuda.is_available()
                    else None
                )
            except Exception:  # noqa: BLE001 - a probe failure means no special case.
                capability_major = None
            attn_impl, disable_flash_sdp = resolve_attention_implementation(
                config.attn_implementation,
                capability_major,
                native_windows=sys.platform == "win32",
            )
            if disable_flash_sdp:
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    attn_impl = "sdpa"
                except Exception as exc:  # noqa: BLE001
                    raise TrainerError(
                        "could not enforce the native-Windows Blackwell math attention path"
                    ) from exc
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    _stage("model_load", "materialized the sealed model weights")
    if config.execution_configuration_hash is not None:
        verify_local_inputs_after_load(config)
        try:
            verify_loaded_model_execution(model, config)
        except ExecutionPlacementDeviation as exc:
            _stage("placement_deviation", str(exc))
            raise
        _stage("placement_verified", f"observed exact device map {config.device_map}")
        _stage(
            "attention_policy_applied",
            "applied exact SDP toggles and observed model attention API; "
            f"required kernel={effective_kernel}",
        )
    _stage("model_loaded", f"loaded {config.base_model}")  # the end of the long silent load window
    if quantize:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
        _stage("quantized", "prepared for 4-bit k-bit training")
    model = get_peft_model(model, LoraConfig(**build_lora_kwargs(config)))
    gradient_tracker = enforce_trainable_precision(model, torch, config)
    verify_model_state_execution(model, torch, config, quantize=quantize)
    if config.execution_configuration_hash is not None:
        _stage(
            "placement_verified",
            f"observed post-adapter parameter placement on {(config.device_map or {}).get('', '')}",
        )
        _stage(
            "precision_verified",
            "observed base storage, dequantization, adapter master-weight, and gradient policies",
        )
    _stage("adapter_attached", "LoRA adapter attached")

    dataset = Dataset.from_list([{"text": text} for text in texts if text])
    del texts
    if len(dataset) == 0:
        raise TrainerError("The dataset produced no usable training rows.")

    # New sealed plans bind the exact field surface observed during capability probing. No semantic
    # field may disappear through version-dependent filtering.
    import dataclasses  # noqa: PLC0415

    raw_kwargs = build_training_kwargs(config)
    valid_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    if config.execution_configuration_hash is not None:
        required = set(config.required_sft_config_fields)
        if set(raw_kwargs) != required:
            missing = sorted(required - set(raw_kwargs))
            unexpected = sorted(set(raw_kwargs) - required)
            raise TrainerError(
                f"sealed trainer argument drift (missing={missing}, unexpected={unexpected})"
            )
        unavailable = sorted(required - valid_fields)
        if unavailable:
            raise TrainerError(
                "the installed SFTConfig no longer exposes sealed fields: "
                + ", ".join(unavailable)
            )
        sft_config = SFTConfig(**raw_kwargs)
    else:
        if "max_seq_length" not in valid_fields and "max_seq_length" in raw_kwargs:
            seq_len = raw_kwargs.pop("max_seq_length")
            if "max_length" in valid_fields:
                raw_kwargs["max_length"] = seq_len
        sft_config = SFTConfig(**{k: v for k, v in raw_kwargs.items() if k in valid_fields})

    execution_tracker = (
        TrainingExecutionTracker(
            config=config,
            torch_module=torch,
            gradients=gradient_tracker,
            progress_callback=progress_callback,
            stage_callback=_stage,
        )
        if gradient_tracker is not None
        else None
    )

    # Token accounting is fully observational: the accumulator only reads collated batches, and its
    # per-step total is reported through ``token_callback`` at the step boundary. Any fault degrades to
    # no token evidence (null downstream) - it can never fail or alter training.
    token_accumulator = _TokenAccumulator() if token_callback is not None else None

    class _ProgressCallback(TrainerCallback):  # type: ignore[misc]
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            optimizer = kwargs.get("optimizer")
            if execution_tracker is not None:
                execution_tracker.on_train_begin(optimizer)
            elif optimizer is not None:
                _stage("optimizer_created", "observed the optimizer at on_train_begin")

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            optimizer = kwargs.get("optimizer")
            if execution_tracker is not None:
                execution_tracker.on_step_end(int(state.global_step), optimizer)
            elif optimizer is not None:
                verify_optimizer_state_precision(optimizer, torch, config)
            if token_accumulator is not None and token_callback is not None:
                nonpadding, supervised, observed = token_accumulator.flush()
                try:
                    token_callback(int(state.global_step), nonpadding, supervised, observed)
                except Exception:  # noqa: BLE001 - token accounting must never affect training
                    pass

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            step = int(state.global_step)
            total = int(state.max_steps or 0)
            if execution_tracker is not None:
                execution_tracker.on_log(step, total, logs)
            elif progress_callback is not None and logs and "loss" in logs:
                loss = float(logs["loss"])
                progress_callback(step, total, loss)

    # The tokenizer arg was renamed `tokenizer` -> `processing_class`; pass it under whichever exists.
    import inspect  # noqa: PLC0415

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "callbacks": [_ProgressCallback()],
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if config.execution_configuration_hash is not None:
        if config.tokenizer_parameter not in trainer_params:
            raise TrainerError(
                f"SFTTrainer no longer exposes sealed tokenizer parameter "
                f"{config.tokenizer_parameter!r}"
            )
        trainer_kwargs[config.tokenizer_parameter] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    class _TokenObservingSFTTrainer(SFTTrainer):  # type: ignore[misc,valid-type]
        """SFTTrainer that observes the exact collated batch each optimizer microstep consumes, for
        token accounting, without altering it. It reads ``inputs`` at ``training_step`` - the trainer's
        own consumption boundary - because the accelerate-prepared dataloader (``accelerator.prepare``)
        does NOT honor a ``collate_fn`` reassignment on the returned shard, so a dataloader-level wrap is
        silently bypassed on this pinned stack (trl 1.8.0 / transformers 5.13.1 / accelerate 1.14.0) and
        was the cause of the v6 token-throughput=0.0 gap. The observation is a pure read that forwards
        every training object unchanged; when there is no accumulator it is byte-identical to
        ``SFTTrainer``; any counting fault degrades to no token evidence and never disturbs training."""

        def training_step(self, *args: Any, **kwargs: Any) -> Any:
            if token_accumulator is not None:
                # inputs is the collated, device-placed batch: positionally args[1], else kwargs.
                inputs = args[1] if len(args) > 1 else kwargs.get("inputs")
                if isinstance(inputs, Mapping):
                    token_accumulator.observe(inputs)
            return super().training_step(*args, **kwargs)

    trainer = _TokenObservingSFTTrainer(**trainer_kwargs)
    evidence_model = getattr(trainer, "model", model)
    if gradient_tracker is not None:
        restored_trainable = reassert_trainable_precision(
            evidence_model,
            torch,
            config,
            gradient_tracker,
        )
        # SFTTrainer initialization is part of the semantic execution boundary: it may not replace,
        # move, or silently recast sealed adapter/base state before the first backward pass.
        verify_model_state_execution(evidence_model, torch, config, quantize=quantize)
        _stage(
            "precision_verified",
            "reasserted and reverified sealed trainable precision after trainer "
            f"initialization; restored {len(restored_trainable)} tensor(s)",
        )
    before_trainable_state = (
        capture_trainable_state(
            evidence_model,
            torch,
            stage=StageMarker.adapter_attached,
        )
        if execution_tracker is not None
        else None
    )
    before_export_state = (
        capture_adapter_export_state(
            get_peft_model_state_dict(evidence_model),
            torch,
            stage=StageMarker.adapter_attached,
        )
        if execution_tracker is not None
        else None
    )
    # Training frameworks print metrics/tqdm to STDOUT — and transformers can log to stdout during
    # SAVE too — so redirect the whole train+save block to stderr. Only the CLI's final JSON echo then
    # writes to stdout, keeping it the pure JSON result the desktop/WBG parses. Progress reaches stderr.
    with contextlib.redirect_stdout(sys.stderr), enforced_attention_training_kernel(
        torch,
        config,
    ):
        train_output = trainer.train()

        steps = int(getattr(trainer.state, "global_step", 0) or 0)
        verify_completed_step_count(config, steps)
        execution_evidence = None
        if execution_tracker is not None:
            if (
                before_trainable_state is None or before_export_state is None
            ):  # pragma: no cover - constructed together above.
                raise TrainingEvidenceError(
                    "missing pre-training adapter/export state snapshot",
                    taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                    stage=StageMarker.adapter_attached,
                )
            trained_model = getattr(trainer, "model", evidence_model)
            after_export_state = capture_adapter_export_state(
                get_peft_model_state_dict(trained_model),
                torch,
                stage=StageMarker.optimizer_step,
            )
            execution_evidence = execution_tracker.finalize(
                steps=steps,
                before=before_trainable_state,
                before_export=before_export_state,
                after_export=after_export_state,
                adapter_config_semantic_sha256=expected_saved_adapter_config_sha256(
                    trained_model,
                    config,
                ),
                model=trained_model,
            )

        output_dir = Path(config.output_dir)
        _stage("export", "serializing the final adapter and tokenizer")
        try:
            trainer.save_model(str(output_dir))  # saves the LoRA adapter
            tokenizer.save_pretrained(str(output_dir))
            checkpoints = _list_checkpoints(output_dir)
        except Exception as exc:  # noqa: BLE001 - normalize framework/filesystem export failures.
            raise TrainingEvidenceError(
                f"final adapter serialization failed: {exc}",
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
                remediation="preserve the run directory and inspect storage and serializer errors",
            ) from exc

        metrics = getattr(train_output, "metrics", {}) or {}

        # Self-documenting run: write a model card next to the adapter (recipe + honesty + base-model
        # license reminder). Best-effort — the card is a convenience and must NEVER fail a completed run.
        try:
            from corpus_studio.training.model_card import build_model_card, write_model_card  # noqa: PLC0415

            write_model_card(
                output_dir,
                build_model_card(
                    output_dir,
                    base_model=config.base_model,
                    training_config={
                        "format": config.dataset_format,
                        "sequence_len": config.sequence_len,
                        "learning_rate": config.learning_rate,
                        "seed": config.seed,
                    },
                    train_result={"steps": steps, "final_loss": metrics.get("train_loss"), "cpu_toy": config.cpu_toy},
                ),
            )
        except Exception:  # noqa: BLE001 - never let card-writing break a finished training run.
            pass

    return TrainResult(
        output_dir=str(output_dir),
        adapter_path=str(output_dir),
        base_model=config.base_model,
        cpu_toy=config.cpu_toy,
        steps=steps,
        final_loss=(
            execution_evidence.step_losses[-1].loss
            if execution_evidence is not None
            else metrics.get("train_loss")
        ),
        checkpoints=checkpoints,
        execution_evidence=execution_evidence,
    )
