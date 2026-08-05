"""The import-light trainer boundary for a from-scratch / continued PRETRAINING run - the sibling of
``TrainRunConfig`` for full-parameter causal-LM pretraining (no adapter, no 4-bit, a packed corpus).

Torch-free: this maps the sealed :class:`ResolvedPretrainingExecutionConfiguration` to a flat config the
(torch) ``run_pretraining`` worker consumes, WITHOUT importing torch, so the mapping is unit-tested in the
base gate. The dense-SFT seal is untouched - pretraining has its OWN resolved config and its OWN mapping.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CorpusShardBinding(BaseModel):
    """One content-hashed corpus shard the worker streams + packs (location + integrity + token count)."""

    model_config = ConfigDict(extra="forbid")

    location: str = Field(min_length=1)
    content_sha256: str = Field(min_length=1)
    token_count: int = Field(ge=0)


class PretrainRunConfig(BaseModel):
    """A flat, torch-free description of a pretraining run for the worker: how to init the model (random
    from an architecture config, or continued from a checkpoint), the tokenizer, the packed corpus, and
    the full-parameter training hyperparameters."""

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
    # tokenizer (import/freeze pin a content digest; train builds one - a later worker slice)
    tokenizer_mode: Literal["train", "import", "freeze"]
    tokenizer_content_sha256: str | None = None
    tokenizer_algorithm: str | None = None
    tokenizer_vocab_size: int | None = None
    tokenizer_special_tokens: list[str] | None = None
    # packed corpus
    corpus_shards: list[CorpusShardBinding]
    # training
    sequence_len: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    adam_epsilon: float = Field(default=1e-8, gt=0)
    max_grad_norm: float = Field(default=1.0, ge=0)
    lr_scheduler: str | None = None
    warmup_ratio: float | None = Field(default=None, ge=0, le=1)
    max_steps: int | None = Field(default=None, ge=1)
    num_train_epochs: float | None = Field(default=None, gt=0)
    optim: str = "adamw_torch"
    # runtime
    cpu_toy: bool = False
    gradient_checkpointing: bool = True
    attn_implementation: str | None = None
    forward_compute_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    seed: int = Field(default=42, ge=0)
    data_seed: int = Field(default=42, ge=0)
    output_dir: str = Field(min_length=1)
    execution_configuration_hash: str | None = None


def pretrain_config_from_resolved(execution: Any) -> PretrainRunConfig:
    """Map the sealed ``ResolvedPretrainingExecutionConfiguration`` to the trainer boundary (no torch, no
    defaults invented - every value comes from the sealed config)."""
    init = execution.init
    arch = init.architecture_ref
    checkpoint = init.source_checkpoint_ref
    tokenizer = execution.tokenizer_source
    return PretrainRunConfig(
        init_mode=init.mode,
        architecture_ref_location=arch.id if arch else None,
        architecture_ref_sha256=arch.hash.value if arch and arch.hash else None,
        source_checkpoint_location=checkpoint.id if checkpoint else None,
        source_checkpoint_sha256=checkpoint.hash.value if checkpoint and checkpoint.hash else None,
        init_vocab_size=init.vocab_size,
        init_seed=init.init_seed,
        initializer_range=init.initializer_range,
        tokenizer_mode=tokenizer.mode,
        tokenizer_content_sha256=tokenizer.tokenizer_content_sha256,
        tokenizer_algorithm=tokenizer.algorithm,
        tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_special_tokens=list(tokenizer.special_tokens) if tokenizer.special_tokens else None,
        corpus_shards=[
            CorpusShardBinding(
                location=shard.location,
                content_sha256=shard.content_sha256,
                token_count=shard.token_count,
            )
            for shard in execution.data.shards
        ],
        sequence_len=execution.sequence.max_sequence_len,
        micro_batch_size=execution.batching.micro_batch_size,
        gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps,
        learning_rate=execution.optimizer.learning_rate,
        weight_decay=execution.optimizer.weight_decay,
        adam_epsilon=execution.optimizer.adam_epsilon,
        max_grad_norm=execution.optimizer.max_grad_norm,
        lr_scheduler=execution.optimizer.lr_scheduler,
        warmup_ratio=execution.optimizer.warmup_ratio,
        max_steps=execution.schedule.max_steps,
        num_train_epochs=execution.schedule.num_train_epochs,
        optim=execution.optimizer.impl.value,
        cpu_toy=execution.runtime_mode == "cpu_toy",
        gradient_checkpointing=execution.gradient_checkpointing,
        attn_implementation=execution.attention.model_attention_api.value,
        forward_compute_dtype=execution.precision.forward_compute_dtype.value,
        gradient_dtype=execution.precision.gradient_dtype.value,
        seed=execution.seed,
        data_seed=execution.data_seed,
        output_dir=execution.output_dir,
        execution_configuration_hash=execution.configuration_hash,
    )
