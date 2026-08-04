"""Execution variants of the ONE canonical training harness (P0d, #484).

The load-bearing gates: the sealed dense-QLoRA-SFT ResolvedExecutionConfiguration is unchanged by this
control-plane slice - its execution-affecting semantics are byte-identical and its hash-seal is
self-consistent - and only the workload_verified dense_qlora_sft variant is admitted (every non-SFT
variant is refused fail-closed).

The semantic pin deliberately excludes the environment-captured provenance - every value whose bytes
depend on the machine rather than on the sealed SFT semantics: the installed trainer package versions
(the dependency-light engine venv reports them "not-installed" while CI has them installed), the input
POINTERS (an absolute checkout path plus a raw-bytes dataset digest), and the formatter identity
(inspect.getsource of trainer.py) - the latter two are CRLF in a dev checkout but LF on CI. This
control-plane slice cannot touch any of them, so pinning them would break CI for reasons unrelated to
the sealed config. The real configuration_hash still seals them in production; here we prove the config
is internally sealed rather than pin a machine-specific value.
"""

from __future__ import annotations

import importlib.metadata
import re

import pytest

from corpus_studio.platform.contracts import (
    BackendExecutionVariant,
    ResolvedExecutionConfiguration,
)
from corpus_studio.platform.enums import ExecutionVariantKind, ExecutionVariantSupport, TaskType
from corpus_studio.platform.execution_config import (
    canonical_sha256,
    execution_configuration_hash_for,
)
from corpus_studio.platform.execution_variants import (
    ExecutionVariantRefused,
    admit_execution_variant,
    admit_task_execution_variant,
    execution_variant_kind_for_task,
    reference_execution_variants,
    variant_envelope,
)
from corpus_studio.platform.runners import demo_training_plan

# The environment-captured provenance on the sealed config - everything whose bytes depend on the
# machine rather than on the sealed SFT semantics, so the pin holds across a CRLF dev checkout and CI's
# LF checkout regardless of which trainer packages are installed:
#   - trainer_interface.package_versions : importlib.metadata versions (installed vs "not-installed")
#   - capability_report_ref / attention.kernel_probe_ref : refs whose hashes derive from those versions
#   - inputs : dataset/model/tokenizer POINTERS - an absolute checkout path + a raw-bytes file digest
#     (CRLF vs LF); the dataset FORMAT semantics live in the separate `data` field, kept below
#   - data.formatter_sha256 : hashes inspect.getsource(format_example_text), i.e. trainer.py's bytes
#   - configuration_hash : derived from all of the above
# Every execution-affecting field (adapter / optimizer / loss / precision / sequence / data-format /
# batching / schedule / the task-type locks) stays in the pin; that no env-specific value leaks in is
# asserted by test_semantic_pin_has_no_environment_specific_values.
_ENV_PROVENANCE_EXCLUDE = {
    "configuration_hash": True,
    "capability_report_ref": True,
    "attention": {"kernel_probe_ref": True},
    "trainer_interface": {"package_versions": True},
    "inputs": True,
    "data": {"formatter_sha256": True},
}

# The trainer packages demo_training_plan() version-stamps via importlib.metadata; used only to prove
# env-invariance of the semantic pin.
_TRAINER_PACKAGES = frozenset({"accelerate", "datasets", "peft", "torch", "transformers", "trl"})

# In-code model refs whose hashes are over pydantic content (not bytes on disk or package versions), so
# they are stable across CRLF/LF checkouts and install sets - the only 64-hex values allowed to survive
# into the semantic pin.
_STABLE_HASH_ROOTS = frozenset({"backend_ref", "objective_ref", "environment_ref", "environment_binding"})

# GOLDEN semantic baseline of the sealed dense-QLoRA-SFT config, captured before this P0d slice with the
# environment provenance excluded. If this changes, the sealed worker lineage's execution semantics
# (adapter / optimizer / loss / precision / sequence / data / task-type locks) have been altered.
_SFT_SEMANTIC_SHA = "ba48d66986b94ad1e941fdf2ff0b18de1042fdf10bc42ce2d9db0e9e8cfeb6f1"
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


def test_semantic_pin_has_no_environment_specific_values():
    # the strongest guard on the exclusion set: whatever fields the sealed config carries, the semantic
    # pin must hold no value whose bytes depend on the machine - no absolute path, and no file/source
    # digest outside the in-code model refs that hash pydantic content rather than bytes on disk. If a
    # future field leaks a path, a raw-bytes file digest, or an inspect.getsource hash into the pin,
    # this fails loudly here instead of silently breaking CI on the next CRLF/LF or version skew.
    def _leaves(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield from _leaves(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                yield from _leaves(value, f"{path}[{index}]")
        else:
            yield path, obj

    offenders = [
        (path, value)
        for path, value in _leaves(_semantic_dump(_sft_config()))
        if isinstance(value, str)
        and (
            value.startswith("/")
            or ":\\" in value
            or (re.fullmatch(r"[0-9a-f]{64}", value) and path.split(".")[0] not in _STABLE_HASH_ROOTS)
        )
    ]
    assert offenders == [], f"environment-specific values leaked into the semantic pin: {offenders}"


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


def test_task_maps_to_an_execution_variant_shape_and_admits_fail_closed():
    declared = reference_execution_variants()
    # sft -> the workload_verified dense-QLoRA-SFT shape: admitted, returns that variant.
    assert execution_variant_kind_for_task(TaskType.sft) == ExecutionVariantKind.dense_qlora_sft
    assert admit_task_execution_variant(TaskType.sft, declared_variants=declared).variant_kind == (
        ExecutionVariantKind.dense_qlora_sft
    )
    # pretraining -> a declared-only shape: refused below workload_verified (no fallback).
    assert execution_variant_kind_for_task(TaskType.pretraining) == ExecutionVariantKind.pretraining
    with pytest.raises(ExecutionVariantRefused, match="below the required"):
        admit_task_execution_variant(TaskType.pretraining, declared_variants=declared)
    # preference WITHOUT a recognized objective -> no shape: refused (it resolves by objective, not task).
    assert execution_variant_kind_for_task(TaskType.preference) is None
    with pytest.raises(ExecutionVariantRefused, match="no executable execution variant"):
        admit_task_execution_variant(TaskType.preference, declared_variants=declared)


def test_preference_resolves_by_its_specific_objective_not_the_task():
    declared = reference_execution_variants()
    # Only the QLoRA-DPO objective has a built (PEFT) shape; DPO/IPO/KTO/ORPO are NOT interchangeable, so
    # a task-level 'preference -> DPO' mapping would mis-claim the others.
    assert (
        execution_variant_kind_for_task(TaskType.preference, objective_id="dpo_qlora")
        == ExecutionVariantKind.preference_dpo
    )
    for oid in ("dpo", "ipo", "kto", "orpo", "unknown"):
        assert execution_variant_kind_for_task(TaskType.preference, objective_id=oid) is None
    # a MoE preference request has no built shape: it is refused, NOT routed to the generic moe variant
    # (the preference-objective rule is applied before MoE routing).
    assert (
        execution_variant_kind_for_task(
            TaskType.preference, objective_id="dpo_qlora", is_moe=True
        )
        is None
    )
    # dpo_qlora is declared at contract_validated: admitted at that bar, refused below workload_verified.
    admitted = admit_task_execution_variant(
        TaskType.preference,
        objective_id="dpo_qlora",
        declared_variants=declared,
        required_support=ExecutionVariantSupport.contract_validated,
    )
    assert admitted.variant_kind == ExecutionVariantKind.preference_dpo
    with pytest.raises(ExecutionVariantRefused, match="below the required 'workload_verified'"):
        admit_task_execution_variant(
            TaskType.preference, objective_id="dpo_qlora", declared_variants=declared
        )
    # an unrecognized/unbuilt preference objective is refused as no-variant.
    with pytest.raises(ExecutionVariantRefused, match="no executable execution variant"):
        admit_task_execution_variant(
            TaskType.preference, objective_id="ipo", declared_variants=declared
        )


def test_moe_topology_routes_to_the_declared_only_shape_regardless_of_task():
    # Shape is objective x TOPOLOGY: a MoE model routes to the moe shape even for an sft task, and moe is
    # declared-only -> refused fail-closed. It is never misclassified as the workload_verified dense
    # shape (the sealed registry marks MoE declared-only).
    declared = reference_execution_variants()
    assert execution_variant_kind_for_task(TaskType.sft, is_moe=True) == ExecutionVariantKind.moe
    with pytest.raises(ExecutionVariantRefused, match="below the required 'workload_verified'"):
        admit_task_execution_variant(TaskType.sft, is_moe=True, declared_variants=declared)
