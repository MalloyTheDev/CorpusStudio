"""The reward-model execution seal (RL slice S5a-1): ``ResolvedRewardExecutionConfiguration`` is the
sibling of the byte-locked dense SFT / DPO seals for the ``reward_model`` variant, carried on
``RunPlan.resolved_reward_execution``. It must be hash-sealable and fail closed on the reward-specific
invariants (the first-party ``corpus_studio`` backend, the ``reward_model`` objective, the
``reward_model`` export family, a ``SEQ_CLS`` head), and never perturb the SFT/DPO seals. Execution stays
gated - the reward-head trainer branch + a workload-verified run + the promoting wheel are later slices."""

import pytest

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    PreferenceDataPolicy,
    ResolvedRewardExecutionConfiguration,
    RewardModelingSpec,
    _canonical_contract_sha256,
)
from corpus_studio.platform.enums import AdapterMethod, ExportFormat, QuantizationMode
from corpus_studio.platform.objectives import get_objective
from corpus_studio.platform.runners import demo_training_plan


def _reward_objective_ref() -> Ref:
    obj = get_objective("reward_model")
    assert obj is not None
    return Ref(id="reward_model", hash=HashRef(value=obj.objective_hash))


def _reward_fields(**over) -> dict:
    """A valid reward-model execution config built by reusing the dense SFT demo config's shared execution
    sub-specs and swapping in the reward data + modeling seals (exactly how a resolver will lower a reward
    plan). A reward model trains on the SAME chosen/rejected pairs, so it reuses ``PreferenceDataPolicy``."""
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    fields = dict(
        configuration_id=sft.configuration_id,
        configuration_hash="0" * 64,  # placeholder; _reward_config reseals a valid one
        backend_ref=sft.backend_ref,
        environment_ref=sft.environment_ref,
        environment_binding=sft.environment_binding,
        capability_report_ref=sft.capability_report_ref,
        inputs=sft.inputs,
        objective_ref=_reward_objective_ref(),
        runtime_mode=sft.runtime_mode,
        precision=sft.precision.model_copy(
            update={
                "quantized_storage_format": QuantizationMode.nf4,
                "weight_storage_dtype": None,
                "dequantization_dtype": sft.precision.forward_compute_dtype,
            }
        ),
        attention=sft.attention,
        device_map=sft.device_map,
        adapter=sft.adapter.model_copy(update={"method": AdapterMethod.qlora}),
        optimizer=sft.optimizer,
        sequence=sft.sequence,
        batching=sft.batching,
        checkpoint_policy=sft.checkpoint_policy,
        schedule=sft.schedule,
        data=PreferenceDataPolicy(
            schema_id="preference",
            schema_version="0.1.0",
            schema_sha256="e" * 64,
            formatter_id="corpus-studio:preference-chat-v1",
            formatter_sha256="a" * 64,
            max_prompt_length=sft.sequence.max_sequence_len // 2,
            max_length=sft.sequence.max_sequence_len,
            data_seed=sft.data_seed,
        ),
        reward=RewardModelingSpec(),
        trainer_interface=sft.trainer_interface,
        export_format=ExportFormat.reward_model,
        bnb_4bit_use_double_quant=sft.bnb_4bit_use_double_quant,
        save_strategy=sft.save_strategy,
        gradient_checkpointing=sft.gradient_checkpointing,
        output_dir=sft.output_dir,
        seed=sft.seed,
        data_seed=sft.data_seed,
    )
    fields.update(over)
    return fields


def _reward_config(**over) -> ResolvedRewardExecutionConfiguration:
    cfg = ResolvedRewardExecutionConfiguration(**_reward_fields(**over))
    body_hash = _canonical_contract_sha256(cfg.model_dump(mode="json", exclude={"configuration_hash"}))
    return cfg.model_copy(update={"configuration_hash": body_hash})


def test_reward_execution_config_round_trips_and_is_a_seq_cls_reward_family():
    cfg = _reward_config()
    expected = _canonical_contract_sha256(cfg.model_dump(mode="json", exclude={"configuration_hash"}))
    assert cfg.configuration_hash == expected
    # A reward model is a scalar SEQ_CLS score head exporting the reward_model artifact family - never the
    # CAUSAL_LM policy adapter every other config locks.
    assert cfg.adapter_task_type == "SEQ_CLS"
    assert cfg.export_format == ExportFormat.reward_model
    assert cfg.reward.family == "pairwise" and cfg.reward.loss_type == "bradley_terry"


def test_reward_seal_must_bind_the_reward_model_objective():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="must bind the 'reward_model' objective"):
        ResolvedRewardExecutionConfiguration(**_reward_fields(objective_ref=sft.objective_ref))


def test_reward_seal_requires_the_reward_model_export_family():
    with pytest.raises(ValueError, match="reward_model' artifact family"):
        ResolvedRewardExecutionConfiguration(**_reward_fields(export_format=ExportFormat.adapter_peft))


def test_reward_seal_requires_the_first_party_backend():
    with pytest.raises(ValueError, match="corpus_studio worker backend"):
        ResolvedRewardExecutionConfiguration(
            **_reward_fields(backend_ref=Ref(id="echo", hash=HashRef(value="b" * 64)))
        )


def test_reward_requires_the_qlora_adapter_method():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="requires the qlora adapter method"):
        ResolvedRewardExecutionConfiguration(
            **_reward_fields(adapter=sft.adapter.model_copy(update={"method": AdapterMethod.lora}))
        )


def test_reward_refuses_a_length_budget_that_overflows_the_sequence_window():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    overflow = PreferenceDataPolicy(
        schema_id="preference",
        schema_version="0.1.0",
        schema_sha256="e" * 64,
        formatter_id="corpus-studio:preference-chat-v1",
        formatter_sha256="a" * 64,
        max_prompt_length=sft.sequence.max_sequence_len,
        max_length=sft.sequence.max_sequence_len + 128,  # overflows the sealed window
        data_seed=sft.data_seed,
    )
    with pytest.raises(ValueError, match="must fit within the sealed sequence length"):
        ResolvedRewardExecutionConfiguration(**_reward_fields(data=overflow))
