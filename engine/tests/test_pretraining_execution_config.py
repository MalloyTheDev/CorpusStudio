"""The pretraining execution seal (S3a-1): ``ResolvedPretrainingExecutionConfiguration`` is the sibling
of the byte-locked dense SFT seal for the ``pretraining`` execution variant. Unlike SFT/DPO it is a
FULL-PARAMETER causal-LM run - no adapter, no 4-bit base, no single dataset file: the model comes from a
``ModelInitializationSpec`` (random-from-config or a continued checkpoint), the tokenizer from a
``TokenizerSourceSpec`` (train/import/freeze), and the corpus from the sharded ``PretrainingDataPolicy``.
It must be hash-sealable, fail closed on the pretraining-specific invariants, and NEVER perturb the dense
SFT seal. Execution stays gated - the pretraining worker loop + a workload-verified run + the milestone
wheel are the retained-human slice (RunPlan carry + resolver + CLI are S3a-2)."""

import pytest

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    CheckpointPolicy,
    ModelInitializationSpec,
    PretrainingDataPolicy,
    PretrainingShard,
    ResolvedPretrainingExecutionConfiguration,
    TokenizerSourceSpec,
    TrainingDataPolicy,
)
from corpus_studio.platform.enums import (
    CheckpointImpl,
    ExportFormat,
    LossImpl,
    QuantizationMode,
)
from corpus_studio.platform.execution_config import (
    pretraining_execution_configuration_hash_for,
    verify_execution_configuration_hash,
    verify_pretraining_execution_configuration_hash,
)
from corpus_studio.platform.objectives import get_objective
from corpus_studio.platform.runners import demo_training_plan


def _pretraining_objective_ref(continued: bool = False) -> Ref:
    obj_id = "continued_pretraining" if continued else "pretraining"
    obj = get_objective(obj_id)
    assert obj is not None
    return Ref(id=obj_id, hash=HashRef(value=obj.objective_hash))


def _pretraining_data(**over) -> PretrainingDataPolicy:
    base = dict(
        shards=(
            PretrainingShard(
                shard_id="shard-0",
                location="corpus/shard-0.jsonl",
                source="web",
                row_count=100,
                token_count=100_000,
                content_sha256="d" * 64,
            ),
        ),
        data_seed=42,
        global_batch_size=8,
        token_budget=1_000_000,
    )
    base.update(over)
    return PretrainingDataPolicy(**base)


def _quantized_4bit(sft_precision):
    return sft_precision.model_copy(
        update={
            "quantized_storage_format": QuantizationMode.nf4,
            "weight_storage_dtype": None,
            "dequantization_dtype": sft_precision.forward_compute_dtype,
        }
    )


def _pretrain_fields(**over) -> dict:
    """A valid FULL-PARAMETER pretraining execution config built by reusing the dense SFT demo config's
    shared execution sub-specs (placement / precision / attention / optimizer / sequence / batching /
    schedule / trainer interface) and swapping in the pretraining seals: the model init, the tokenizer
    source, the sharded corpus, a full-model (not adapter-only) checkpoint, and a full-model export.
    Reusing the proven sub-specs is exactly how a resolver will lower a pretraining plan."""
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    fields = dict(
        configuration_id=sft.configuration_id,
        configuration_hash="0" * 64,  # placeholder; the caller reseals a valid config
        backend_ref=sft.backend_ref,
        environment_ref=sft.environment_ref,
        environment_binding=sft.environment_binding,
        capability_report_ref=sft.capability_report_ref,
        objective_ref=_pretraining_objective_ref(),
        runtime_mode=sft.runtime_mode,
        init=ModelInitializationSpec(
            mode="random",
            architecture_ref=Ref(id="arch:demo-small", hash=HashRef(value="c" * 64)),
            vocab_size=32000,
            init_seed=42,
        ),
        tokenizer_source=TokenizerSourceSpec(
            mode="train",
            algorithm="bpe",
            vocab_size=32000,
            special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"],
        ),
        precision=sft.precision,  # the demo SFT precision is unquantized, weight==forward (full-param ok)
        attention=sft.attention,
        device_map=sft.device_map,
        optimizer=sft.optimizer,
        loss_impl=LossImpl.cross_entropy,
        sequence=sft.sequence,
        batching=sft.batching,
        checkpoint_policy=CheckpointPolicy(impl=CheckpointImpl.full_state),
        schedule=sft.schedule,
        data=_pretraining_data(data_seed=sft.data_seed),
        trainer_interface=sft.trainer_interface,
        export_format=ExportFormat.merged_safetensors,
        gradient_checkpointing=sft.gradient_checkpointing,
        output_dir=sft.output_dir,
        seed=sft.seed,
        data_seed=sft.data_seed,
    )
    fields.update(over)
    return fields


def _pretrain_config(**over) -> ResolvedPretrainingExecutionConfiguration:
    cfg = ResolvedPretrainingExecutionConfiguration(**_pretrain_fields(**over))
    return cfg.model_copy(
        update={"configuration_hash": pretraining_execution_configuration_hash_for(cfg)}
    )


# ---- expressible + sealed --------------------------------------------------------------------------


def test_pretraining_config_round_trips_and_seals_self_consistently():
    cfg = _pretrain_config()
    assert cfg.init.mode == "random"
    assert cfg.tokenizer_source.mode == "train"
    assert isinstance(cfg.data, PretrainingDataPolicy)  # pretraining data policy, never the SFT one
    assert verify_pretraining_execution_configuration_hash(cfg)
    assert (
        ResolvedPretrainingExecutionConfiguration.model_validate_json(cfg.model_dump_json()) == cfg
    )


def test_continued_pretraining_config_seals():
    cfg = _pretrain_config(
        objective_ref=_pretraining_objective_ref(continued=True),
        init=ModelInitializationSpec(
            mode="continued",
            source_checkpoint_ref=Ref(id="ckpt:base", hash=HashRef(value="e" * 64)),
        ),
        tokenizer_source=TokenizerSourceSpec(mode="freeze", tokenizer_content_sha256="f" * 64),
    )
    assert cfg.init.mode == "continued"
    assert verify_pretraining_execution_configuration_hash(cfg)


def test_seal_changes_when_the_init_vocab_changes():
    base = _pretrain_config()
    # A different vocab is a different sealed identity - and the model + tokenizer vocab move together
    # (the config seal enforces they agree).
    other = _pretrain_config(
        init=ModelInitializationSpec(
            mode="random",
            architecture_ref=Ref(id="arch:demo-small", hash=HashRef(value="c" * 64)),
            vocab_size=50000,
            init_seed=42,
        ),
        tokenizer_source=TokenizerSourceSpec(
            mode="train",
            algorithm="bpe",
            vocab_size=50000,
            special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"],
        ),
    )
    assert base.configuration_hash != other.configuration_hash
    assert verify_pretraining_execution_configuration_hash(other)


def test_model_vocab_must_match_the_trained_tokenizer():
    # A from-scratch model whose embedding vocab disagrees with its freshly-trained tokenizer would fail
    # at train time; the seal refuses it fail-closed rather than sealing a guaranteed mismatch.
    with pytest.raises(ValueError, match="must equal the trained tokenizer's vocab_size"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(
                tokenizer_source=TokenizerSourceSpec(
                    mode="train",
                    algorithm="bpe",
                    vocab_size=16000,  # != the init/model vocab (32000)
                    special_tokens=["<eos>"],
                )
            )
        )


# ---- fails closed on the pretraining-specific invariants -------------------------------------------


def test_pretraining_must_bind_a_pretraining_objective():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="'pretraining' or 'continued_pretraining' objective"):
        ResolvedPretrainingExecutionConfiguration(**_pretrain_fields(objective_ref=sft.objective_ref))


def test_pretraining_init_mode_must_match_the_objective():
    # random init cannot bind the continued_pretraining objective (and vice versa).
    with pytest.raises(ValueError, match="init.mode must match the objective"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(objective_ref=_pretraining_objective_ref(continued=True))
        )


def test_continued_pretraining_cannot_train_a_new_tokenizer():
    with pytest.raises(ValueError, match="cannot train a new tokenizer"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(
                objective_ref=_pretraining_objective_ref(continued=True),
                init=ModelInitializationSpec(
                    mode="continued",
                    source_checkpoint_ref=Ref(id="ckpt:base", hash=HashRef(value="e" * 64)),
                ),
            )
        )


def test_pretraining_refuses_a_quantized_base():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="does not quantize the base"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(precision=_quantized_4bit(sft.precision))
        )


def test_pretraining_refuses_adapter_only_checkpoints():
    with pytest.raises(ValueError, match="does not use adapter-only checkpoints"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(checkpoint_policy=CheckpointPolicy(impl=CheckpointImpl.adapter_only))
        )


def test_pretraining_refuses_a_peft_adapter_export():
    with pytest.raises(ValueError, match="not a PEFT adapter"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(export_format=ExportFormat.adapter_peft)
        )


def test_pretraining_refuses_a_non_cross_entropy_loss():
    with pytest.raises(ValueError, match="next-token cross entropy"):
        ResolvedPretrainingExecutionConfiguration(**_pretrain_fields(loss_impl=LossImpl.dpo))


def test_pretraining_requires_matching_data_seeds():
    with pytest.raises(ValueError, match="must match for one reproducible"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(data=_pretraining_data(data_seed=1), data_seed=2)
        )


def test_pretraining_requires_hash_pinned_refs():
    with pytest.raises(ValueError, match="must be hash-pinned"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(backend_ref=Ref(id="corpus_studio"))
        )


def test_pretraining_requires_the_first_party_backend():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="corpus_studio worker backend"):
        ResolvedPretrainingExecutionConfiguration(
            **_pretrain_fields(backend_ref=sft.backend_ref.model_copy(update={"id": "echo"}))
        )


# ---- the method sub-spec validators ----------------------------------------------------------------


def test_random_init_requires_an_architecture_and_seed():
    with pytest.raises(ValueError, match="hash-pinned architecture_ref"):
        ModelInitializationSpec(mode="random", vocab_size=32000, init_seed=42)
    with pytest.raises(ValueError, match="vocab_size and init_seed"):
        ModelInitializationSpec(
            mode="random", architecture_ref=Ref(id="a", hash=HashRef(value="c" * 64))
        )


def test_continued_init_requires_a_source_checkpoint():
    with pytest.raises(ValueError, match="hash-pinned source_checkpoint_ref"):
        ModelInitializationSpec(mode="continued")


def test_train_tokenizer_requires_algorithm_and_special_tokens():
    with pytest.raises(ValueError, match="algorithm and a vocab_size"):
        TokenizerSourceSpec(mode="train", special_tokens=["<eos>"])
    with pytest.raises(ValueError, match="declare its special tokens"):
        TokenizerSourceSpec(mode="train", algorithm="bpe", vocab_size=32000)


def test_train_tokenizer_requires_an_eos_token_for_a_usable_base():
    # A from-scratch tokenizer without an eos produces a base model that cannot be fine-tuned (the pretrain
    # -> SFT/DPO handoff crashes on a padless tokenizer). Fail closed at planning with a clear reason.
    with pytest.raises(ValueError, match="end-of-sequence token"):
        TokenizerSourceSpec(mode="train", algorithm="bpe", vocab_size=1024, special_tokens=["<unk>"])
    # </s> is accepted as the eos spelling; a full BOS/EOS/PAD/UNK set is fine.
    assert TokenizerSourceSpec(
        mode="train", algorithm="bpe", vocab_size=1024, special_tokens=["</s>", "<pad>"]
    ).special_tokens == ["</s>", "<pad>"]
    assert TokenizerSourceSpec(
        mode="train", algorithm="bpe", vocab_size=1024,
        special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"],
    ).special_tokens == ["<bos>", "<eos>", "<pad>", "<unk>"]


def test_import_tokenizer_requires_a_content_digest():
    with pytest.raises(ValueError, match="pinned tokenizer_content_sha256"):
        TokenizerSourceSpec(mode="import")


# ---- the dense SFT seal stays byte-decoupled -------------------------------------------------------


def test_sft_execution_seal_is_unperturbed_by_the_pretraining_sibling():
    # Adding the pretraining sibling must not add a field to, or reshape, the byte-locked dense SFT seal.
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    assert verify_execution_configuration_hash(sft)
    dump = sft.model_dump(mode="json")
    assert "init" not in dump and "tokenizer_source" not in dump  # no pretraining field leaked
    assert isinstance(sft.data, TrainingDataPolicy)  # the SFT seal keeps the SFT data policy
