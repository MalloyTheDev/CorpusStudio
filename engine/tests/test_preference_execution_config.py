"""The DPO execution seal (S2b-2 step 3): ``ResolvedPreferenceExecutionConfiguration`` is the sibling of
the byte-locked dense SFT seal for the ``preference_dpo`` variant, carried on
``RunPlan.resolved_preference_execution``. It must be hash-sealable, fail closed on the DPO-specific
invariants, be carried on a RunPlan (at most one execution authority, byte-safe), and NEVER perturb the
dense SFT seal. Execution stays gated - the DPOTrainer branch + a workload-verified run + the milestone
wheel are the retained-human slice."""

import pytest

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    AdapterSpec,
    PreferenceDataPolicy,
    PreferenceOptimizationSpec,
    ReferenceModelBinding,
    ResolvedPreferenceExecutionConfiguration,
    RunPlan,
    TrainingDataPolicy,
)
from corpus_studio.platform.enums import AdapterMethod, QuantizationMode
from corpus_studio.platform.execution_config import (
    preference_execution_configuration_hash_for,
    verify_execution_configuration_hash,
    verify_preference_execution_configuration_hash,
)
from corpus_studio.platform.objectives import get_objective
from corpus_studio.platform.runners import demo_training_plan


def _dpo_qlora_objective_ref() -> Ref:
    obj = get_objective("dpo_qlora")
    assert obj is not None
    return Ref(id="dpo_qlora", hash=HashRef(value=obj.objective_hash))


def _qlora_4bit_precision(sft_precision):
    # the demo SFT precision is unquantized LoRA; make it the QLoRA 4-bit form the dpo_qlora seal needs.
    return sft_precision.model_copy(
        update={
            "quantized_storage_format": QuantizationMode.nf4,
            "weight_storage_dtype": None,
            "dequantization_dtype": sft_precision.forward_compute_dtype,
        }
    )


def _preference_data(**over) -> PreferenceDataPolicy:
    base = dict(
        schema_id="preference",
        schema_version="0.1.0",
        schema_sha256="e" * 64,
        formatter_id="corpus-studio:preference-chat-v1",
        formatter_sha256="a" * 64,
        max_prompt_length=512,
        max_length=1024,
    )
    base.update(over)
    return PreferenceDataPolicy(**base)


def _dpo_fields(**over) -> dict:
    """A valid DPO execution config built by reusing the dense SFT demo config's shared execution
    sub-specs (placement / precision / attention / adapter / optimizer / sequence / batching /
    checkpoint / schedule / trainer interface) and swapping in the DPO data + loss seals. Reusing the
    proven sub-specs is exactly how a resolver will lower a preference plan."""
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
        objective_ref=_dpo_qlora_objective_ref(),
        runtime_mode=sft.runtime_mode,
        precision=_qlora_4bit_precision(sft.precision),
        attention=sft.attention,
        device_map=sft.device_map,
        adapter=sft.adapter.model_copy(update={"method": AdapterMethod.qlora}),
        optimizer=sft.optimizer,
        sequence=sft.sequence,
        batching=sft.batching,
        checkpoint_policy=sft.checkpoint_policy,
        schedule=sft.schedule,
        # size the DPO length budget to the demo's small sealed sequence window, and keep the data seed
        # equal to the top-level execution seed so the base fixture is internally consistent.
        data=_preference_data(
            max_prompt_length=sft.sequence.max_sequence_len // 2,
            max_length=sft.sequence.max_sequence_len,
            data_seed=sft.data_seed,
        ),
        preference=PreferenceOptimizationSpec(
            reference_model=ReferenceModelBinding(mode="frozen_base"),
        ),
        trainer_interface=sft.trainer_interface,
        export_format=sft.export_format,
        bnb_4bit_use_double_quant=sft.bnb_4bit_use_double_quant,
        save_strategy=sft.save_strategy,
        gradient_checkpointing=sft.gradient_checkpointing,
        output_dir=sft.output_dir,
        seed=sft.seed,
        data_seed=sft.data_seed,
    )
    fields.update(over)
    return fields


def _dpo_config(**over) -> ResolvedPreferenceExecutionConfiguration:
    cfg = ResolvedPreferenceExecutionConfiguration(**_dpo_fields(**over))
    return cfg.model_copy(
        update={"configuration_hash": preference_execution_configuration_hash_for(cfg)}
    )


# ---- expressible + sealed --------------------------------------------------------------------------


def test_dpo_execution_config_round_trips_and_seals_self_consistently():
    cfg = _dpo_config()
    assert cfg.preference.objective == "dpo"
    assert cfg.preference.reference_model.mode == "frozen_base"
    assert isinstance(cfg.data, PreferenceDataPolicy)  # DPO data policy, never the SFT one
    assert verify_preference_execution_configuration_hash(cfg)
    assert ResolvedPreferenceExecutionConfiguration.model_validate_json(cfg.model_dump_json()) == cfg


def test_dpo_seal_changes_when_a_loss_hyperparameter_changes():
    # the DPO loss params are inside the seal, so a beta/loss change is a different sealed identity.
    base = _dpo_config()
    other = _dpo_config(
        preference=PreferenceOptimizationSpec(
            beta=0.5, reference_model=ReferenceModelBinding(mode="frozen_base")
        )
    )
    assert base.configuration_hash != other.configuration_hash
    assert verify_preference_execution_configuration_hash(other)


# ---- fails closed on the DPO-specific invariants ---------------------------------------------------


def test_dpo_requires_the_qlora_adapter_method():
    # the only admitted preference objective (dpo_qlora) requires qlora; full-parameter and plain LoRA
    # have no admitting objective and are refused.
    with pytest.raises(ValueError, match="requires the qlora adapter method"):
        ResolvedPreferenceExecutionConfiguration(
            **_dpo_fields(adapter=AdapterSpec(method=AdapterMethod.full_finetune))
        )


def test_dpo_seal_must_bind_the_dpo_qlora_objective():
    # a config whose objective lineage points elsewhere (here the SFT lora objective) is refused.
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="must bind the 'dpo_qlora' objective"):
        ResolvedPreferenceExecutionConfiguration(**_dpo_fields(objective_ref=sft.objective_ref))


def test_dpo_refuses_to_pack_sequences():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="does not pack sequences"):
        ResolvedPreferenceExecutionConfiguration(
            **_dpo_fields(sequence=sft.sequence.model_copy(update={"packing": True}))
        )


def test_dpo_refuses_a_length_budget_that_overflows_the_sequence_window():
    with pytest.raises(ValueError, match="must fit within the sealed sequence length"):
        # PreferenceDataPolicy already guarantees max_prompt_length < max_length; here max_length
        # exceeds the sealed 4096 sequence window.
        ResolvedPreferenceExecutionConfiguration(
            **_dpo_fields(data=_preference_data(max_prompt_length=4096, max_length=8192))
        )


def test_dpo_refuses_a_silent_truncation_contradiction():
    # the same no-silent-truncation contradiction the SFT sibling refuses.
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    seq = sft.sequence.max_sequence_len
    with pytest.raises(ValueError, match="would silently truncate"):
        ResolvedPreferenceExecutionConfiguration(
            **_dpo_fields(
                sequence=sft.sequence.model_copy(update={"truncation_allowed": False}),
                data=_preference_data(
                    max_prompt_length=seq // 2, max_length=seq, truncation_policy="allow"
                ),
            )
        )


def test_dpo_requires_matching_data_seeds():
    # the preference-data seed and the top-level execution data seed must agree (one sample order).
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    seq = sft.sequence.max_sequence_len
    with pytest.raises(ValueError, match="must match for one reproducible"):
        ResolvedPreferenceExecutionConfiguration(
            **_dpo_fields(
                data=_preference_data(max_prompt_length=seq // 2, max_length=seq, data_seed=1),
                data_seed=2,
            )
        )


def test_reference_model_defaults_to_the_frozen_base():
    ref = ReferenceModelBinding()
    assert ref.mode == "frozen_base"
    assert ref.precompute_ref_log_probs is False


def test_dpo_execution_config_requires_hash_pinned_refs():
    with pytest.raises(ValueError, match="must be hash-pinned"):
        ResolvedPreferenceExecutionConfiguration(**_dpo_fields(backend_ref=Ref(id="corpus_studio")))


def test_dpo_requires_the_first_party_backend():
    # the seal enforces the corpus_studio backend itself (a pinned non-corpus_studio backend refuses),
    # so a standalone config can never name a wrong backend.
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="corpus_studio worker backend"):
        ResolvedPreferenceExecutionConfiguration(
            **_dpo_fields(backend_ref=sft.backend_ref.model_copy(update={"id": "echo"}))
        )


def test_qlora_dpo_requires_a_4bit_base():
    # QLoRA-DPO must be over a 4-bit base; an unquantized base contradicts the sealed quantization.
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="QLoRA-DPO requires a 4-bit quantized base"):
        ResolvedPreferenceExecutionConfiguration(**_dpo_fields(precision=sft.precision))


# ---- carried on RunPlan ----------------------------------------------------------------------------


def _dpo_plan_payload():
    """A RunPlan payload carrying the DPO config, with its summaries reconciled to the sealed config
    (task type, quantization, adapter) - exactly what a resolver would emit consistently."""
    sft_plan = demo_training_plan()
    dpo = _dpo_config()  # reuses the demo SFT config's refs/sub-specs (backend/env/dataset/model + more)
    payload = sft_plan.model_dump(mode="json")
    payload["resolved_execution"] = None
    payload["resolved_preference_execution"] = dpo.model_dump(mode="json")
    payload["task_type"] = "preference"
    payload["quantization"] = dpo.precision.quantized_storage_format.value
    payload["adapter"] = dpo.adapter.model_dump(mode="json")
    payload["loss_impl"] = "dpo"
    return sft_plan, dpo, payload


def test_run_plan_carries_the_dpo_config_at_most_one_and_byte_safe():
    sft_plan, dpo, payload = _dpo_plan_payload()
    # A plan carrying the DPO config (and NOT the SFT config), with matching summaries, validates.
    plan = RunPlan.model_validate(payload)
    assert plan.resolved_preference_execution is not None
    assert verify_preference_execution_configuration_hash(plan.resolved_preference_execution)
    # at-most-one: carrying BOTH the SFT and the DPO config is refused.
    both = dict(payload)
    assert sft_plan.resolved_execution is not None
    both["resolved_execution"] = sft_plan.resolved_execution.model_dump(mode="json")
    with pytest.raises(ValueError, match="never both"):
        RunPlan.model_validate(both)
    # byte-safe: an SFT plan (no preference config) omits the field from its serialization entirely.
    assert "resolved_preference_execution" not in sft_plan.model_dump(mode="json")


def test_run_plan_refuses_a_dpo_config_on_a_non_preference_plan():
    # the DPO seal must sit on a preference plan - a summary that contradicts the seal is refused.
    _sft_plan, _dpo, payload = _dpo_plan_payload()
    payload["task_type"] = "sft"
    with pytest.raises(ValueError, match="requires a preference RunPlan"):
        RunPlan.model_validate(payload)


# ---- the dense SFT seal stays byte-decoupled -------------------------------------------------------


def test_sft_execution_seal_is_unperturbed_by_the_dpo_sibling():
    # Adding the DPO sibling must not add a field to, or otherwise reshape, the byte-locked dense SFT
    # seal - the two are fully decoupled. (The absolute SFT semantic golden is guarded in
    # test_execution_variants.py; this asserts the seal stays self-consistent and DPO-free.)
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    assert verify_execution_configuration_hash(sft)
    dump = sft.model_dump(mode="json")
    assert "preference" not in dump  # no DPO field leaked into the SFT seal
    assert isinstance(sft.data, TrainingDataPolicy)  # the SFT seal keeps the SFT data policy
