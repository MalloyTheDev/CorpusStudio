"""The import-light trainer boundary for a from-scratch / continued PRETRAINING run - the sibling of
``TrainRunConfig`` for full-parameter causal-LM pretraining (no adapter, no 4-bit, a packed corpus).

Torch-free: this maps the sealed :class:`ResolvedPretrainingExecutionConfiguration` to a flat config the
(torch) ``run_pretraining`` worker consumes, WITHOUT importing torch, so the mapping is unit-tested in the
base gate. The dense-SFT seal is untouched - pretraining has its OWN resolved config and its OWN mapping.

FAITHFULNESS: the flat config must carry every sealed field the worker acts on, or the worker would run
something different from what was sealed (loss, precision dtypes, optimizer betas, device map, attention
kernel, export format, custom-code binding, continued-reset intent, corpus mixture/packing, tokenizer
threshold). A lossy lowering is a silent divergence, so this mirrors them all.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from corpus_studio.platform.contracts import ResolvedPretrainingExecutionConfiguration


def _enum_value(value: object) -> str | None:
    """The string of a str-Enum sealed field, or None (keeps the mapping torch-free + total)."""
    if value is None:
        return None
    return value.value if isinstance(value, Enum) else str(value)


def _ref_sha256(ref: object) -> str:
    """A Ref's pinned digest (custom-code refs are validator-guaranteed pinned; "" only if absent)."""
    hash_ref = getattr(ref, "hash", None)
    return getattr(hash_ref, "value", None) or ""


class CorpusShardBinding(BaseModel):
    """One content-hashed corpus shard the worker streams + packs (location + integrity + token count)."""

    model_config = ConfigDict(extra="forbid")

    location: str = Field(min_length=1)
    content_sha256: str = Field(min_length=1)
    token_count: int = Field(ge=0)


class CustomCodeBinding(BaseModel):
    """The admitted custom-block bundle sealed on a mode-3 custom_decoder run (carried so the worker can
    load + re-verify the exact bytes; execution stays gated by the sandbox slice)."""

    model_config = ConfigDict(extra="forbid")

    bundle_location: str = Field(min_length=1)
    bundle_sha256: str = Field(min_length=1)
    entry_symbol: str = Field(min_length=1)
    interface_version: str = Field(min_length=1)
    vetting_location: str = Field(min_length=1)
    vetting_sha256: str = Field(min_length=1)


class PretrainRunConfig(BaseModel):
    """A flat, torch-free, FAITHFUL description of a pretraining run for the worker."""

    model_config = ConfigDict(extra="forbid")

    # model init
    init_mode: Literal["random", "continued"]
    architecture_ref_location: str | None = None
    architecture_ref_sha256: str | None = None
    source_checkpoint_location: str | None = None
    source_checkpoint_sha256: str | None = None
    init_vocab_size: int | None = Field(default=None, ge=1)
    init_seed: int | None = Field(default=None, ge=0)
    initializer_range: float | None = Field(default=None, gt=0)
    reset_optimizer: bool = True
    reset_lr_scheduler: bool = True
    reset_data_cursor: bool = True
    custom_code: CustomCodeBinding | None = None
    # tokenizer
    tokenizer_mode: Literal["train", "import", "freeze"]
    tokenizer_content_sha256: str | None = None
    tokenizer_algorithm: str | None = None
    tokenizer_vocab_size: int | None = None
    tokenizer_special_tokens: list[str] | None = None
    tokenizer_min_frequency: int | None = None
    # packed corpus policy
    corpus_shards: list[CorpusShardBinding]
    corpus_streaming: bool = True
    corpus_mixture_weights: dict[str, float] = Field(default_factory=dict)
    corpus_document_boundaries: bool = True
    corpus_packing: str = "concat_and_split"
    corpus_data_seed: int = Field(default=42, ge=0)
    corpus_token_budget: int | None = Field(default=None, ge=1)
    corpus_epochs: int | None = Field(default=None, ge=1)
    global_batch_size: int = Field(ge=1)
    # training
    sequence_len: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.999, ge=0, lt=1)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, ge=0)
    lr_scheduler: str | None = None
    warmup_ratio: float | None = Field(default=None, ge=0, le=1)
    max_steps: int | None = Field(default=None, ge=1)
    num_train_epochs: float | None = Field(default=None, gt=0)
    optim: str = "adamw_torch"
    loss_impl: str = "cross_entropy"
    # precision (full-parameter: every sealed dtype matters)
    forward_compute_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    weight_storage_dtype: str | None = None
    dequantization_dtype: str | None = None
    optimizer_state_dtype: str | None = None
    optimizer_auxiliary_dtype: str | None = None
    master_weight_dtype: str | None = None
    # attention
    attn_implementation: str | None = None
    attention_kernel: str | None = None
    flash_sdp_enabled: bool | None = None
    mem_efficient_sdp_enabled: bool | None = None
    math_sdp_enabled: bool | None = None
    # placement / checkpoint / export / runtime
    device_map: dict[str, str] = Field(default_factory=dict)
    checkpoint_impl: str | None = None
    checkpoint_cadence_optimizer_steps: int | None = None
    checkpoint_keep_last: int | None = None
    checkpoint_reload_verify: bool = False
    export_format: str = "merged_safetensors"
    output_layout: str = "run_scoped_v1"
    output_dir: str = Field(min_length=1)
    gradient_checkpointing: bool = True
    cpu_toy: bool = False
    seed: int = Field(default=42, ge=0)
    data_seed: int = Field(default=42, ge=0)
    execution_configuration_hash: str | None = None


def pretrain_config_from_resolved(
    execution: ResolvedPretrainingExecutionConfiguration,
) -> PretrainRunConfig:
    """Map the sealed config to the trainer boundary faithfully (no torch, no invented defaults - every
    value comes from the sealed config so the worker runs exactly what was sealed)."""
    init = execution.init
    arch = init.architecture_ref
    checkpoint = init.source_checkpoint_ref
    tokenizer = execution.tokenizer_source
    corpus = execution.data
    optimizer = execution.optimizer
    precision = execution.precision
    attention = execution.attention
    checkpoint_policy = execution.checkpoint_policy

    custom_code = None
    if init.custom_code is not None:
        cc = init.custom_code
        custom_code = CustomCodeBinding(
            bundle_location=cc.code_bundle_ref.id,
            bundle_sha256=_ref_sha256(cc.code_bundle_ref),
            entry_symbol=cc.entry_symbol,
            interface_version=cc.interface_version,
            vetting_location=cc.vetting_ref.id,
            vetting_sha256=_ref_sha256(cc.vetting_ref),
        )

    return PretrainRunConfig(
        init_mode=init.mode,
        architecture_ref_location=arch.id if arch else None,
        architecture_ref_sha256=arch.hash.value if arch and arch.hash else None,
        source_checkpoint_location=checkpoint.id if checkpoint else None,
        source_checkpoint_sha256=checkpoint.hash.value if checkpoint and checkpoint.hash else None,
        init_vocab_size=init.vocab_size,
        init_seed=init.init_seed,
        initializer_range=init.initializer_range,
        reset_optimizer=init.reset_optimizer,
        reset_lr_scheduler=init.reset_lr_scheduler,
        reset_data_cursor=init.reset_data_cursor,
        custom_code=custom_code,
        tokenizer_mode=tokenizer.mode,
        tokenizer_content_sha256=tokenizer.tokenizer_content_sha256,
        tokenizer_algorithm=tokenizer.algorithm,
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_special_tokens=list(tokenizer.special_tokens) if tokenizer.special_tokens else None,
        tokenizer_min_frequency=tokenizer.min_frequency,
        corpus_shards=[
            CorpusShardBinding(
                location=shard.location,
                content_sha256=shard.content_sha256,
                token_count=shard.token_count,
            )
            for shard in corpus.shards
        ],
        corpus_streaming=corpus.streaming,
        corpus_mixture_weights=dict(corpus.mixture_weights),
        corpus_document_boundaries=corpus.document_boundaries,
        corpus_packing=corpus.packing,
        corpus_data_seed=corpus.data_seed,
        corpus_token_budget=corpus.token_budget,
        corpus_epochs=corpus.epochs,
        global_batch_size=corpus.global_batch_size,
        sequence_len=execution.sequence.max_sequence_len,
        micro_batch_size=execution.batching.micro_batch_size,
        gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps,
        learning_rate=optimizer.learning_rate,
        weight_decay=optimizer.weight_decay,
        adam_beta1=optimizer.adam_beta1,
        adam_beta2=optimizer.adam_beta2,
        adam_epsilon=optimizer.adam_epsilon,
        max_grad_norm=optimizer.max_grad_norm,
        lr_scheduler=optimizer.lr_scheduler,
        warmup_ratio=optimizer.warmup_ratio,
        max_steps=execution.schedule.max_steps,
        num_train_epochs=execution.schedule.num_train_epochs,
        optim=optimizer.impl.value,
        loss_impl=execution.loss_impl.value,
        forward_compute_dtype=precision.forward_compute_dtype.value,
        gradient_dtype=precision.gradient_dtype.value,
        weight_storage_dtype=_enum_value(precision.weight_storage_dtype),
        dequantization_dtype=_enum_value(precision.dequantization_dtype),
        optimizer_state_dtype=_enum_value(precision.optimizer_state_dtype),
        optimizer_auxiliary_dtype=_enum_value(precision.optimizer_auxiliary_dtype),
        master_weight_dtype=_enum_value(precision.master_weight_dtype),
        attn_implementation=attention.model_attention_api.value,
        attention_kernel=attention.effective_backend_required.value,
        flash_sdp_enabled=attention.flash_sdp_enabled,
        mem_efficient_sdp_enabled=attention.mem_efficient_sdp_enabled,
        math_sdp_enabled=attention.math_sdp_enabled,
        device_map={entry.module: entry.device for entry in execution.device_map},
        checkpoint_impl=_enum_value(checkpoint_policy.impl),
        checkpoint_cadence_optimizer_steps=checkpoint_policy.cadence_optimizer_steps,
        checkpoint_keep_last=checkpoint_policy.keep_last,
        checkpoint_reload_verify=checkpoint_policy.reload_verify,
        export_format=execution.export_format.value,
        output_layout=execution.output_layout,
        output_dir=execution.output_dir,
        gradient_checkpointing=execution.gradient_checkpointing,
        cpu_toy=execution.runtime_mode == "cpu_toy",
        seed=execution.seed,
        data_seed=execution.data_seed,
        execution_configuration_hash=execution.configuration_hash,
    )
