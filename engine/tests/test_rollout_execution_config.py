"""The on-policy RL execution seal (RL slice S5b, gated L1 design #839):
``ResolvedRolloutExecutionConfiguration`` is the sibling of the byte-locked SFT / DPO / reward seals for the
``on_policy_rl`` variant, carried on ``RunPlan.resolved_rollout_execution``. It must be hash-sealable and
fail closed on the rollout-specific invariants (the first-party ``corpus_studio`` backend, the ``grpo``
objective, a CAUSAL_LM policy adapter exporting ``adapter_peft``, the GRPO shape, a served reward source,
the prompt+generation budget fitting the window), and never perturb the other seals. Execution stays gated -
the rollout+reward+GRPO worker + a workload-verified run + the promoting wheel are later slices."""

import pytest

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    ExperienceSource,
    PolicyOptimizationSpec,
    ResolvedRolloutExecutionConfiguration,
    RewardSourceRef,
    RolloutSpec,
    StabilityController,
    _canonical_contract_sha256,
)
from corpus_studio.platform.enums import AdapterMethod, ExportFormat, QuantizationMode
from corpus_studio.platform.runners import demo_training_plan


def _grpo_objective_ref() -> Ref:
    # The 'grpo' objective is registered in the resolver slice (S5b-2); the seal only checks the id + pin.
    return Ref(id="grpo", hash=HashRef(value="f" * 64))


def _served_reward_ref() -> Ref:
    return Ref(id="reward-model-x", hash=HashRef(value="c" * 64))


def _served_reward_source(**over) -> RewardSourceRef:
    """A valid served reward source: the loadable identity (base + adapter dir) + the hash-pinned reward
    RunManifest that proves it came from an admitted reward run (the provenance binding)."""
    fields = dict(
        kind="served_reward_model",
        reward_ref=_served_reward_ref(),
        reward_base_model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_adapter_location="/runs/reward/artifacts/adapter",
        provenance_manifest_ref=Ref(id="reward-run-manifest", hash=HashRef(value="d" * 64)),
    )
    fields.update(over)
    return RewardSourceRef(**fields)


def _rollout_fields(**over) -> dict:
    """A valid on-policy RL execution config built by reusing the dense SFT demo config's shared execution
    sub-specs and swapping in the rollout/experience/reward-source/stability/policy seals (exactly how a
    resolver will lower an on-policy plan). The policy is a CAUSAL_LM QLoRA adapter over a 4-bit base."""
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    window = sft.sequence.max_sequence_len
    fields = dict(
        configuration_id=sft.configuration_id,
        configuration_hash="0" * 64,  # placeholder; _rollout_config reseals a valid one
        backend_ref=sft.backend_ref,
        environment_ref=sft.environment_ref,
        environment_binding=sft.environment_binding,
        capability_report_ref=sft.capability_report_ref,
        inputs=sft.inputs,
        objective_ref=_grpo_objective_ref(),
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
        experience=ExperienceSource(
            schema_id="prompt",
            schema_version="0.1.0",
            schema_sha256="e" * 64,
            formatter_id="corpus-studio:prompt-chat-v1",
            formatter_sha256="a" * 64,
            max_prompt_length=window // 4,
            data_seed=sft.data_seed,
        ),
        rollout=RolloutSpec(
            sampling_temperature=1.0,
            sampling_top_p=0.95,
            max_new_tokens=window // 4,
            rollouts_per_prompt=4,
        ),
        reward_source=_served_reward_source(),
        stability=StabilityController(kl_coefficient=0.05, clip_range=0.2),
        policy_optimization=PolicyOptimizationSpec(algorithm="grpo"),
        trainer_interface=sft.trainer_interface,
        export_format=ExportFormat.adapter_peft,
        bnb_4bit_use_double_quant=sft.bnb_4bit_use_double_quant,
        save_strategy=sft.save_strategy,
        gradient_checkpointing=sft.gradient_checkpointing,
        output_dir=sft.output_dir,
        seed=sft.seed,
        data_seed=sft.data_seed,
    )
    fields.update(over)
    return fields


def _rollout_config(**over) -> ResolvedRolloutExecutionConfiguration:
    cfg = ResolvedRolloutExecutionConfiguration(**_rollout_fields(**over))
    body_hash = _canonical_contract_sha256(cfg.model_dump(mode="json", exclude={"configuration_hash"}))
    return cfg.model_copy(update={"configuration_hash": body_hash})


def test_rollout_execution_config_round_trips_and_is_a_causal_lm_policy():
    cfg = _rollout_config()
    expected = _canonical_contract_sha256(cfg.model_dump(mode="json", exclude={"configuration_hash"}))
    assert cfg.configuration_hash == expected
    # On-policy RL trains a CAUSAL_LM POLICY adapter (adapter_peft), never a SEQ_CLS score head.
    assert cfg.adapter_task_type == "CAUSAL_LM"
    assert cfg.export_format == ExportFormat.adapter_peft
    assert cfg.policy_optimization.algorithm == "grpo" and cfg.reward_source.kind == "served_reward_model"
    assert cfg.rollout.rollouts_per_prompt >= 2  # a GRPO group


def test_rollout_seal_must_bind_the_grpo_objective():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="must bind the 'grpo' objective"):
        ResolvedRolloutExecutionConfiguration(**_rollout_fields(objective_ref=sft.objective_ref))


def test_rollout_seal_requires_the_adapter_peft_export():
    with pytest.raises(ValueError, match="PEFT policy adapter"):
        ResolvedRolloutExecutionConfiguration(
            **_rollout_fields(export_format=ExportFormat.reward_model)
        )


def test_rollout_seal_requires_the_first_party_backend():
    with pytest.raises(ValueError, match="corpus_studio worker backend"):
        ResolvedRolloutExecutionConfiguration(
            **_rollout_fields(backend_ref=Ref(id="echo", hash=HashRef(value="b" * 64)))
        )


def test_rollout_requires_the_qlora_adapter_method():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    with pytest.raises(ValueError, match="requires the qlora adapter method"):
        ResolvedRolloutExecutionConfiguration(
            **_rollout_fields(adapter=sft.adapter.model_copy(update={"method": AdapterMethod.lora}))
        )


def test_rollout_admits_only_the_grpo_shape():
    # PPO is the S5c slice - a PPO policy-optimization spec is refused at the seal today.
    with pytest.raises(ValueError, match="only the GRPO shape"):
        ResolvedRolloutExecutionConfiguration(
            **_rollout_fields(policy_optimization=PolicyOptimizationSpec(algorithm="ppo", use_critic=True))
        )


def test_rollout_admits_only_a_served_reward_model_source():
    with pytest.raises(ValueError, match="served reward-model source"):
        ResolvedRolloutExecutionConfiguration(
            **_rollout_fields(
                reward_source=RewardSourceRef(kind="rlaif_judge", reward_ref=_served_reward_ref())
            )
        )


def test_rollout_refuses_a_generation_budget_that_overflows_the_window():
    sft = demo_training_plan().resolved_execution
    assert sft is not None
    window = sft.sequence.max_sequence_len
    with pytest.raises(ValueError, match="must fit the sealed sequence length"):
        ResolvedRolloutExecutionConfiguration(
            **_rollout_fields(
                rollout=RolloutSpec(
                    sampling_temperature=1.0,
                    sampling_top_p=0.95,
                    max_new_tokens=window,  # prompt budget + this overflows the window
                    rollouts_per_prompt=4,
                )
            )
        )


def test_reward_source_ref_must_be_hash_pinned():
    with pytest.raises(ValueError, match="reward source reference must be hash-pinned"):
        RewardSourceRef(kind="served_reward_model", reward_ref=Ref(id="unpinned"))


def test_served_reward_source_requires_provenance_and_loadable_identity():
    # the chosen binding: a served reward model must PROVE provenance (a pinned reward RunManifest) AND
    # carry the loadable identity the worker reconstructs the scorer from.
    _served_reward_source()  # the valid form round-trips
    with pytest.raises(ValueError, match="hash-pinned provenance RunManifest"):
        _served_reward_source(provenance_manifest_ref=None)
    with pytest.raises(ValueError, match="hash-pinned provenance RunManifest"):
        _served_reward_source(provenance_manifest_ref=Ref(id="unpinned-manifest"))
    with pytest.raises(ValueError, match="reward_base_model"):
        _served_reward_source(reward_base_model=None)
    with pytest.raises(ValueError, match="reward_adapter_location"):
        _served_reward_source(reward_adapter_location=None)


def test_non_served_reward_source_omits_the_served_model_fields():
    # a verifier / RLAIF reward source needs no base model or provenance manifest (those are served-only).
    verifier = RewardSourceRef(kind="verifier", reward_ref=_served_reward_ref())
    assert verifier.reward_base_model is None and verifier.provenance_manifest_ref is None


def test_grpo_policy_optimization_refuses_a_critic():
    with pytest.raises(ValueError, match="must not carry a critic"):
        PolicyOptimizationSpec(algorithm="grpo", use_critic=True)
