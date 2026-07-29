"""Execution variants of the ONE canonical training harness (P0d, #484).

The load-bearing gates: the sealed dense-QLoRA-SFT ResolvedExecutionConfiguration is byte-identical
(serialization + configuration_hash + field set unchanged), and only the workload_verified
dense_qlora_sft variant is admitted - every non-SFT variant is refused fail-closed.
"""

from __future__ import annotations

import hashlib

import pytest

from corpus_studio.platform.contracts import (
    BackendExecutionVariant,
    ResolvedExecutionConfiguration,
)
from corpus_studio.platform.enums import ExecutionVariantKind, ExecutionVariantSupport
from corpus_studio.platform.execution_config import execution_configuration_hash_for
from corpus_studio.platform.execution_variants import (
    ExecutionVariantRefused,
    admit_execution_variant,
    reference_execution_variants,
    variant_envelope,
)
from corpus_studio.platform.runners import demo_training_plan

# GOLDEN baseline of the sealed dense-QLoRA-SFT config, captured from the UNMODIFIED
# ResolvedExecutionConfiguration before this P0d slice. If any of these change, the sealed worker
# lineage (configuration_hash across RunPlan / checkpoint identities) has been broken.
_SFT_CONFIG_HASH = "8153f2020b026d8a9a23193ac508236408e6b65b04aa1bca3aebcb2eb79688cb"
_SFT_SERIALIZATION_SHA = "8c76dd9f6eee82c13579ff3bfcf3c3ecbe23de4c5e6214e61906061488c02ef7"
_SFT_FIELD_COUNT = 33


def _sft_config() -> ResolvedExecutionConfiguration:
    execution = demo_training_plan().resolved_execution
    assert execution is not None
    return execution


# ---- Gate 1/2: byte + hash identity of the sealed SFT configuration --------------------------------


def test_sealed_sft_configuration_serialization_is_byte_identical():
    serialization = _sft_config().model_dump_json()
    assert hashlib.sha256(serialization.encode("utf-8")).hexdigest() == _SFT_SERIALIZATION_SHA


def test_sealed_sft_configuration_hash_is_unchanged():
    cfg = _sft_config()
    assert execution_configuration_hash_for(cfg) == _SFT_CONFIG_HASH
    assert cfg.configuration_hash == _SFT_CONFIG_HASH  # the fixture's own seal agrees


# ---- Gate 3: schema identity (no field added / removed / relaxed on the sealed config) -------------


def test_sealed_sft_configuration_schema_and_locks_are_unchanged():
    fields = ResolvedExecutionConfiguration.model_fields
    assert len(fields) == _SFT_FIELD_COUNT
    assert fields["adapter_task_type"].default == "CAUSAL_LM"
    assert fields["trust_remote_code"].default is False
    assert fields["use_safetensors"].default is True
    for smuggled in ("variant_kind", "execution_variant", "support"):
        assert smuggled not in fields  # no variant field smuggled onto the sealed config


# ---- Variant envelope (derived, not free Booleans) -------------------------------------------------


def test_variant_envelopes_are_derived_and_coherent():
    sft = variant_envelope(ExecutionVariantKind.dense_qlora_sft)
    assert sft.requires_peft_adapter and not sft.allows_full_parameter  # the SFT locks
    for kind in (
        ExecutionVariantKind.dense_full_finetune,
        ExecutionVariantKind.pretraining,
        ExecutionVariantKind.moe,
    ):
        env = variant_envelope(kind)
        assert env.allows_full_parameter and not env.requires_peft_adapter  # full-parameter modes
    # the descriptor carries NO free envelope Booleans - the envelope is derived from the kind, so a
    # contradictory combination cannot be expressed.
    assert set(BackendExecutionVariant.model_fields) == {
        "contract_version",
        "backend_id",
        "variant_kind",
        "support",
    }


# ---- Admission (fail-closed; only workload_verified dense_qlora_sft) --------------------------------


def _variant(kind: ExecutionVariantKind, support: ExecutionVariantSupport) -> BackendExecutionVariant:
    return BackendExecutionVariant(backend_id="corpus_studio", variant_kind=kind, support=support)


def test_only_workload_verified_dense_qlora_sft_is_admitted():
    reference = {v.variant_kind: v for v in reference_execution_variants()}
    admit_execution_variant(reference[ExecutionVariantKind.dense_qlora_sft])  # the proven path
    for kind in (
        ExecutionVariantKind.dense_full_finetune,
        ExecutionVariantKind.pretraining,
        ExecutionVariantKind.moe,
    ):
        with pytest.raises(ExecutionVariantRefused, match="refused"):
            admit_execution_variant(reference[kind])


def test_dense_qlora_sft_below_required_support_is_refused():
    # support is not a casual Boolean: a dense_qlora_sft descriptor that has NOT reached
    # workload_verified is refused even though it is the right kind.
    weak = _variant(ExecutionVariantKind.dense_qlora_sft, ExecutionVariantSupport.worker_implemented)
    with pytest.raises(ExecutionVariantRefused):
        admit_execution_variant(weak)


def test_admission_refuses_an_unsupported_schema_version():
    variant = _variant(ExecutionVariantKind.dense_qlora_sft, ExecutionVariantSupport.workload_verified)
    tampered = variant.model_copy(update={"contract_version": "9.9.9"})
    with pytest.raises(ExecutionVariantRefused, match="schema version"):
        admit_execution_variant(tampered)


def test_reference_variants_only_mark_sft_workload_verified():
    by_kind = {v.variant_kind: v.support for v in reference_execution_variants()}
    assert by_kind[ExecutionVariantKind.dense_qlora_sft] == ExecutionVariantSupport.workload_verified
    assert all(
        by_kind[kind] != ExecutionVariantSupport.workload_verified
        for kind in (
            ExecutionVariantKind.dense_full_finetune,
            ExecutionVariantKind.pretraining,
            ExecutionVariantKind.moe,
        )
    )


def test_descriptor_is_not_the_sealed_worker_configuration():
    # the capability descriptor must never be usable as a worker configuration, and admission returns
    # None (admit/refuse) - never a ResolvedExecutionConfiguration.
    variant = _variant(ExecutionVariantKind.dense_qlora_sft, ExecutionVariantSupport.workload_verified)
    assert not isinstance(variant, ResolvedExecutionConfiguration)
    assert admit_execution_variant(variant) is None
