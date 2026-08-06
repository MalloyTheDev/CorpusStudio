"""The full-parameter fine-tune execution seal (dense_full_finetune slice 1a):
``ResolvedFullFinetuneExecutionConfiguration`` is the FULL-MODEL sibling of the byte-locked adapter-only
dense SFT seal. Same SFT data + objective path, but ALL parameters train and the artifact is a full model
(merged safetensors + full-state checkpoints), which the SFT config's validator hard-refuses. It must be
hash-sealable, fail closed on its full-parameter invariants, NEVER perturb the dense SFT seal, and it is
still refused at execution (the worker + full-model evidence + a workload-verified run are later slices)."""

import pytest
from pydantic import ValidationError

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    AdapterSpec,
    CheckpointPolicy,
    ResolvedFullFinetuneExecutionConfiguration,
)
from corpus_studio.platform.enums import (
    AdapterMethod,
    CheckpointImpl,
    ExportFormat,
    QuantizationMode,
)
from corpus_studio.platform.execution_config import (
    full_finetune_execution_configuration_hash_for,
    verify_execution_configuration_hash,
    verify_full_finetune_execution_configuration_hash,
)
from corpus_studio.platform.objectives import get_objective
from corpus_studio.platform.runners import demo_training_plan


def _full_finetune_objective_ref() -> Ref:
    obj = get_objective("full_parameter_sft")
    assert obj is not None
    return Ref(id="full_parameter_sft", hash=HashRef(value=obj.objective_hash))


def _ff_fields(**over) -> dict:
    """A valid full-parameter fine-tune config built by reusing the dense SFT demo config's shared execution
    sub-specs (the demo SFT precision is unquantized, weight==forward - exactly what full-param needs) and
    swapping in the full-model seals: adapter.method=full_finetune (no LoRA fields), a full-state checkpoint,
    and a merged full-model export. This is exactly how the resolver (slice 1b) will lower a full-param plan."""
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    fields = dict(
        configuration_id=sft.configuration_id,
        configuration_hash="0" * 64,  # placeholder; the caller reseals a valid config
        backend_ref=sft.backend_ref,
        environment_ref=sft.environment_ref,
        environment_binding=sft.environment_binding,
        capability_report_ref=sft.capability_report_ref,
        inputs=sft.inputs,
        objective_ref=_full_finetune_objective_ref(),
        runtime_mode=sft.runtime_mode,
        precision=sft.precision,
        attention=sft.attention,
        device_map=sft.device_map,
        adapter=AdapterSpec(method=AdapterMethod.full_finetune),  # no LoRA fields
        optimizer=sft.optimizer,
        loss_impl=sft.loss_impl,
        sequence=sft.sequence,
        batching=sft.batching,
        checkpoint_policy=sft.checkpoint_policy.model_copy(update={"impl": CheckpointImpl.full_state}),
        schedule=sft.schedule,
        data=sft.data,
        trainer_interface=sft.trainer_interface,
        export_format=ExportFormat.merged_safetensors,
        bnb_4bit_use_double_quant=sft.bnb_4bit_use_double_quant,
        save_strategy=sft.save_strategy,
        gradient_checkpointing=sft.gradient_checkpointing,
        output_dir=sft.output_dir,
        seed=sft.seed,
        data_seed=sft.data_seed,
    )
    fields.update(over)
    return fields


def _ff_config(**over) -> ResolvedFullFinetuneExecutionConfiguration:
    cfg = ResolvedFullFinetuneExecutionConfiguration(**_ff_fields(**over))
    return cfg.model_copy(
        update={"configuration_hash": full_finetune_execution_configuration_hash_for(cfg)}
    )


def test_full_finetune_config_round_trips_and_seals_self_consistently():
    cfg = _ff_config()
    assert cfg.adapter.method == AdapterMethod.full_finetune
    assert cfg.export_format == ExportFormat.merged_safetensors
    assert cfg.checkpoint_policy.impl == CheckpointImpl.full_state
    assert cfg.precision.quantized_storage_format == QuantizationMode.none
    assert verify_full_finetune_execution_configuration_hash(cfg)
    assert (
        ResolvedFullFinetuneExecutionConfiguration.model_validate_json(cfg.model_dump_json()) == cfg
    )


def test_seal_changes_when_an_execution_field_changes():
    base = _ff_config()
    other = _ff_config(seed=1234)
    assert base.configuration_hash != other.configuration_hash


def test_full_finetune_seal_is_decoupled_from_the_sft_seal():
    # The full-finetune hash function is its OWN; the byte-locked SFT seal is never perturbed.
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    assert verify_execution_configuration_hash(sft)  # SFT seal still self-consistent
    cfg = _ff_config()
    assert full_finetune_execution_configuration_hash_for(cfg) != sft.configuration_hash


@pytest.mark.parametrize(
    "over, match",
    [
        (dict(adapter=AdapterSpec(method=AdapterMethod.lora, lora_r=8, lora_alpha=16,
                                  lora_dropout=0.0, target_modules=["all-linear"], bias="none")),
         "requires adapter.method='full_finetune'"),
        (dict(export_format=ExportFormat.adapter_peft), "emits a merged full model"),
        (dict(checkpoint_policy=CheckpointPolicy(impl=CheckpointImpl.adapter_only)),
         "checkpoints the full model state"),
    ],
)
def test_full_finetune_validator_fails_closed(over, match):
    with pytest.raises(ValidationError, match=match):
        ResolvedFullFinetuneExecutionConfiguration(**_ff_fields(**over))


def test_full_finetune_refuses_stray_lora_fields():
    # method=full_finetune but a LoRA field leaked in -> refused (a full fine-tune carries no adapter deltas).
    with pytest.raises(ValidationError, match="carries no LoRA adapter fields"):
        ResolvedFullFinetuneExecutionConfiguration(
            **_ff_fields(adapter=AdapterSpec(method=AdapterMethod.full_finetune, lora_r=8))
        )
