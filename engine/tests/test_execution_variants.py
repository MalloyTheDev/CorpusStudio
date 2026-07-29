"""Execution variants of the ONE canonical training harness (P0d, #484).

The load-bearing gates: the sealed dense-QLoRA-SFT ResolvedExecutionConfiguration is unchanged by this
control-plane slice - its execution-affecting semantics are byte-identical and its hash-seal is
self-consistent - and only the workload_verified dense_qlora_sft variant is admitted (every non-SFT
variant is refused fail-closed).

The semantic pin deliberately excludes the environment-captured provenance: the exact installed
trainer package versions and the capability/probe refs whose hashes derive from them. Those vary by
machine - the dependency-light engine venv reports the trainer packages as "not-installed" while CI
has them installed - and this control-plane slice cannot touch them, so pinning them would break CI
for reasons unrelated to the sealed config. The real configuration_hash still seals those versions in
production; here we prove the config is internally sealed rather than pin a machine-specific value.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from corpus_studio.platform.contracts import (
    BackendExecutionVariant,
    ResolvedExecutionConfiguration,
)
from corpus_studio.platform.enums import ExecutionVariantKind, ExecutionVariantSupport
from corpus_studio.platform.execution_config import (
    canonical_sha256,
    execution_configuration_hash_for,
)
from corpus_studio.platform.execution_variants import (
    ExecutionVariantRefused,
    admit_execution_variant,
    reference_execution_variants,
    variant_envelope,
)
from corpus_studio.platform.runners import demo_training_plan

# The environment-captured provenance on the sealed config: the installed trainer package versions and
# the capability/probe refs whose hashes are derived from them. Excluded from the semantic pin so the
# guard is deterministic across machines (proven env-invariant by
# test_sealed_sft_configuration_semantics_are_env_invariant). Every execution-affecting field stays in.
_ENV_PROVENANCE_EXCLUDE = {
    "configuration_hash": True,
    "capability_report_ref": True,
    "attention": {"kernel_probe_ref": True},
    "trainer_interface": {"package_versions": True},
}

# The trainer packages demo_training_plan() version-stamps via importlib.metadata; used only to prove
# env-invariance of the semantic pin.
_TRAINER_PACKAGES = frozenset({"accelerate", "datasets", "peft", "torch", "transformers", "trl"})

# GOLDEN semantic baseline of the sealed dense-QLoRA-SFT config, captured before this P0d slice with the
# environment provenance excluded. If this changes, the sealed worker lineage's execution semantics
# (adapter / optimizer / loss / precision / sequence / data / task-type locks) have been altered.
_SFT_SEMANTIC_SHA = "9b79711c5081f818951502ea970cebf8e84f29f3e3fcd7e751acd85072e1f25c"
_SFT_FIELD_COUNT = 33


def _sft_config() -> ResolvedExecutionConfiguration:
    execution = demo_training_plan().resolved_execution
    assert execution is not None
    return execution


def _semantic_dump(cfg: ResolvedExecutionConfiguration) -> dict[str, object]:
    """The sealed config's execution semantics with environment-captured provenance removed."""
    return cfg.model_dump(mode="json", exclude=_ENV_PROVENANCE_EXCLUDE)


# ---- Gate 1/2: the sealed SFT configuration's semantics + seal are unchanged -----------------------


def test_sealed_sft_configuration_semantics_are_byte_identical():
    # every execution-affecting field (adapter / optimizer / loss / precision / sequence / data / the
    # task-type locks) is pinned; only the machine-specific package provenance is excluded.
    assert canonical_sha256(_semantic_dump(_sft_config())) == _SFT_SEMANTIC_SHA


def test_sealed_sft_configuration_hash_is_self_consistent():
    # the config carries its own hash-seal and a fresh recomputation agrees - the seal is neither stale
    # nor forged. Its absolute value embeds installed package versions (env-specific), so it is
    # deliberately not pinned to a cross-machine golden here.
    cfg = _sft_config()
    assert cfg.configuration_hash == execution_configuration_hash_for(cfg)


def test_sealed_sft_configuration_semantics_are_env_invariant(monkeypatch):
    # the semantic pin must not depend on which trainer packages happen to be installed: the
    # dependency-light engine venv reports them "not-installed" while CI has real versions. Simulate a
    # different install set and confirm the excluded dump is identical - this is exactly what keeps the
    # golden from breaking CI for reasons unrelated to the sealed config.
    baseline = canonical_sha256(_semantic_dump(_sft_config()))
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "9.9.9-test" if name in _TRAINER_PACKAGES else real_version(name),
    )
    assert canonical_sha256(_semantic_dump(_sft_config())) == baseline == _SFT_SEMANTIC_SHA


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
