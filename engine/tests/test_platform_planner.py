"""Platform slice 6 — the run planner. Pure tests (no torch): synthetic EnvironmentProfile +
CapabilityReport drive every resolution path and the honesty invariants (Blackwell→math from
cc_major, proven-only precision/quant, sequence_len flows, cpu_toy never a silent downgrade, a real
sha256 plan-hash that excludes the volatile stamp). The rendered snapshot is round-tripped through
the actual TrainRunConfig the runner replays."""

import pytest
from pydantic import ValidationError

import corpus_studio.platform as P
from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvironmentProfile,
    EnvHost,
    ExecutionCapabilityCombination,
    GpuDevice,
    ParameterAccountingReport,
    ParameterEvidenceGap,
    ParameterScope,
    ParameterWindow,
    PhysicalExecutionSpec,
    ProbeResult,
    PretrainingDataPolicy,
    PretrainingShard,
    StorageProfile,
    StorageRoleAssessment,
)
from corpus_studio.platform.planner import (
    PlannerConstraints,
    PlannerError,
    _offload_summary,
    _validate_parameter_accounting,
    build_run_plan,
    compute_plan_hash,
    is_trivial_physical_execution,
    run_plan_hash_payload,
    storage_profile_ref_for,
    verify_run_plan_hash,
)
from corpus_studio.platform.parameter_accounting import parameter_accounting_hash_for
from corpus_studio.training.trainer import train_config_from_resolved

_SIG = "a" * 64
_NOW = "2026-07-11T00:00:00+00:00"


def _profile(*, cc_major=None, os="linux"):
    gpus = []
    if cc_major is not None:
        gpus = [
            GpuDevice(
                index=0, kind="cuda", name="GPU", vram_total_bytes=12_000_000_000,
                compute_capability=f"{cc_major}.0", compute_capability_major=cc_major,
            )
        ]
    return EnvironmentProfile(environment_signature=_SIG, host=EnvHost(os=os), gpus=gpus)


def _report(
    *,
    readiness="ready",
    bnb=True,
    precisions=("bf16", "fp32"),
    attn=("math", "sdpa"),
    kernels=("torch_sdpa_flash", "torch_sdpa_math"),
    missing=(),
    physical=False,
    backend_id="corpus_studio",
    extra_combinations=(),
):
    # extra_combinations: (quantization, adapter_method) pairs to ALSO prove as complete bf16/math
    # training tuples - lets a test prove the int8/none precision-ladder combos the GPU probes demonstrate.
    from corpus_studio.platform.backends import get_backend

    backend = get_backend(backend_id)
    trainer_backend = get_backend("corpus_studio")
    assert trainer_backend is not None
    precision_values = sorted(set(precisions))
    attention_values = sorted(set(attn))
    kernel_values = sorted(set(kernels))
    adapter_values = ["lora", "qlora"] if bnb else ["lora"]
    optimizer_values = ["adamw_torch", "paged_adamw_8bit"]
    loss_values = ["cross_entropy", "liger_fused_ce"]
    if readiness == "cpu_toy_only":
        precision_values = sorted(set((*precision_values, "fp32")))
        attention_values = sorted(set((*attention_values, "eager")))
        kernel_values = sorted(set((*kernel_values, "eager")))

    exact: ExecutionCapabilityCombination | None = None
    if readiness != "not_ready":
        precision = "bf16" if "bf16" in precision_values else "fp32" if "fp32" in precision_values else None
        if readiness == "cpu_toy_only":
            exact_attention = ("eager", "eager")
        elif "math" in attention_values and "torch_sdpa_math" in kernel_values:
            exact_attention = ("math", "torch_sdpa_math")
        elif "sdpa" in attention_values and "torch_sdpa_flash" in kernel_values:
            exact_attention = ("sdpa", "torch_sdpa_flash")
        elif "sdpa" in attention_values and "torch_sdpa_math" in kernel_values:
            exact_attention = ("sdpa", "torch_sdpa_math")
        elif "eager" in attention_values and "eager" in kernel_values:
            exact_attention = ("eager", "eager")
        else:
            exact_attention = None
        if precision is not None and exact_attention is not None:
            exact = ExecutionCapabilityCombination.model_validate(
                {
                    "runtime_mode": "cpu_toy" if readiness == "cpu_toy_only" else "training",
                    "device": "cpu" if readiness == "cpu_toy_only" else "cuda",
                    "precision": "fp32" if readiness == "cpu_toy_only" else precision,
                    "quantization": "none" if readiness == "cpu_toy_only" or not bnb else "nf4",
                    "adapter_method": "lora" if readiness == "cpu_toy_only" or not bnb else "qlora",
                    "attention_impl": exact_attention[0],
                    "attention_kernel": exact_attention[1],
                    "optimizer": "adamw_torch",
                    "loss_impl": "cross_entropy",
                    "checkpoint_impl": "adapter_only",
                    "export_format": "adapter_peft",
                    "execution_contract_version": "1.0.0",
                    "probe": "synthetic_execution",
                }
            )
    axis_proofs = {
        "adapter": adapter_values,
        "attention": attention_values,
        "attention_kernel": kernel_values,
        "checkpoint": ["adapter_only"],
        "loss": loss_values,
        "optimizer": optimizer_values,
        "precision": precision_values,
    }
    if physical:
        axis_proofs.update(
            {"placement_mode": ["single_resource"], "placement_tier": ["gpu"]}
        )
    probe_results = [
        ProbeResult(probe="synthetic_axes", outcome="PASS", proves=axis_proofs),
        ProbeResult(
            probe="trainer_contract",
            outcome="PASS",
            proves={
                "trainer_field": trainer_backend.trainer_fields,
                "trainer_init_field": trainer_backend.trainer_init_fields,
            },
        ),
    ]
    if bnb:
        probe_results.append(
            ProbeResult(
                probe="bnb_4bit_load",
                outcome="PASS",
                proves={"quantization": ["nf4"]},
            )
        )
    if exact is not None:
        probe_results.append(
            ProbeResult(
                probe="synthetic_execution",
                outcome="PASS",
                execution_combinations=[exact],
            )
        )
    extra_combos = [
        ExecutionCapabilityCombination.model_validate(
            {
                "runtime_mode": "training", "device": "cuda", "precision": "bf16",
                "quantization": quant, "adapter_method": adapter,
                "attention_impl": "math", "attention_kernel": "torch_sdpa_math",
                "optimizer": "adamw_torch", "loss_impl": "cross_entropy",
                "checkpoint_impl": "adapter_only", "export_format": "adapter_peft",
                "execution_contract_version": "1.0.0", "probe": "synthetic_precision_ladder",
            }
        )
        for quant, adapter in extra_combinations
    ]
    if extra_combos:
        probe_results.append(
            ProbeResult(
                probe="synthetic_precision_ladder",
                outcome="PASS",
                # Mirror how the real int8/none probes emit proves={"quantization": [...]}, so the report's
                # effective quantization_modes equals the passing probe evidence (a CapabilityReport invariant).
                proves={"quantization": sorted({quant for quant, _ in extra_combinations})},
                execution_combinations=extra_combos,
            )
        )
    actual_readiness = (
        "ready"
        if exact is not None and exact.runtime_mode == "training"
        else "cpu_toy_only"
        if exact is not None
        else "not_ready"
    )
    eff = EffectiveCapabilities(
        precision_modes=precision_values,
        quantization_modes=sorted(
            set((["nf4"] if bnb else []) + [quant for quant, _ in extra_combinations])
        ),
        attention_impls=attention_values,
        attention_kernels=kernel_values,
        adapter_methods=adapter_values,
        optimizers=optimizer_values,
        loss_impls=loss_values,
        checkpoint_impls=["adapter_only"],
        execution_contract_versions=["1.0.0"] if exact is not None or extra_combos else [],
        execution_combinations=sorted(
            (([exact] if exact is not None else []) + extra_combos),
            key=lambda item: item.canonical_key(),
        ),
        trainer_fields=trainer_backend.trainer_fields,
        trainer_init_fields=trainer_backend.trainer_init_fields,
        placement_tiers=["gpu"] if physical else [],
        placement_modes=["single_resource"] if physical else [],
    )
    return CapabilityReport(
        backend_id=backend_id,
        backend_version=backend.backend_version if backend is not None else None,
        environment_ref=Ref(id=_SIG), readiness=actual_readiness,
        bitsandbytes_ok=bnb, effective_capabilities=eff, missing_packages=list(missing),
        probe_results=probe_results,
        installed_packages=[
            P.PackageLock(
                name=name,
                normalized_name=name,
                version="1.0",
                hash=P.HashRef(value="1" * 64),
                artifact=f"{name}-1.0-py3-none-any.whl",
                artifact_hash=P.HashRef(value="2" * 64),
                record_integrity="verified",
                record_count_semantics="all_record_rows_v2",
                record_entries=1,
                record_verified_entries=1,
                installed_files_hash=P.HashRef(value="3" * 64),
                installed_file_count=1,
            )
            for name in [
                "accelerate",
                "bitsandbytes",
                "datasets",
                "liger-kernel",
                "peft",
                "torch",
                "transformers",
                "trl",
            ]
        ],
    )


def _plan(
    profile,
    report,
    *,
    now=_NOW,
    parameter_accounting=None,
    physical_execution=None,
    storage_profile=None,
    allow_marginal_storage=False,
    allow_unknown_storage=False,
    **kw,
):
    kw.setdefault("base_model", "Qwen/Qwen2.5-7B-Instruct")
    kw.setdefault("dataset_path", "data/examples.jsonl")
    kw.setdefault("model_revision", "1" * 40)
    kw.setdefault("dataset_content_sha256", "d" * 64)
    constraints = PlannerConstraints(**kw)
    return build_run_plan(
        profile=profile,
        capabilities=report,
        dataset_ref=Ref(id="ds-1", hash=P.HashRef(value="d" * 64)),
        constraints=constraints, plan_id="p1", now=now,
        parameter_accounting=parameter_accounting,
        physical_execution=physical_execution,
        storage_profile=storage_profile,
        allow_marginal_storage=allow_marginal_storage,
        allow_unknown_storage=allow_unknown_storage,
    )


def _accounting_report(*, scope_id="model"):
    model_ref = Ref(id="model", hash=P.HashRef(value="c" * 64))
    scope = ParameterScope(
        scope_id=scope_id,
        kind="model",
        model_ref=model_ref,
        coordinate_universe_id="model-coordinates",
        coordinate_universe_sha256="c" * 64,
        definition="Exact model coordinate universe.",
    )
    gap = ParameterEvidenceGap(
        gap_id="logical-gap",
        kind="logical",
        scope=scope,
        window=ParameterWindow(
            window_id="static-model",
            kind="static_snapshot",
            definition="One static model snapshot.",
        ),
        reason="missing_observation",
        explanation="Logical evidence is deliberately absent in this planner fixture.",
        resolution="Supply a measured logical observation.",
    )
    draft = ParameterAccountingReport(
        report_id="parameter-report",
        report_hash="0" * 64,
        generated_at=_NOW,
        profile="model_static",
        status="incomplete",
        model_ref=model_ref,
        gaps=[gap],
    )
    return draft.model_copy(update={"report_hash": parameter_accounting_hash_for(draft)})


def _scoped_physical(scope_id="model"):
    return PhysicalExecutionSpec.model_validate(
        {
            "resources": [
                {
                    "resource_id": "compute-0",
                    "tier": "gpu",
                    "device_kind": "cuda",
                    "device_id": "cuda:0",
                }
            ],
            "placements": [
                {
                    "placement_id": "parameters-authoritative",
                    "state": "parameters",
                    "selector": {"parameter_scope_ids": [scope_id]},
                    "resource_id": "compute-0",
                    "role": "authoritative",
                }
            ],
            "parallelism": {
                "world_size": 1,
                "ranks": [{"rank": 0, "resource_id": "compute-0"}],
            },
        }
    )


def _offload_physical(*states):
    return PhysicalExecutionSpec.model_validate(
        {
            "resources": [
                {
                    "resource_id": "compute-0",
                    "tier": "gpu",
                    "device_kind": "cuda",
                    "device_id": "cuda:0",
                },
                {
                    "resource_id": "host-ram",
                    "tier": "pageable_ram",
                    "device_kind": "cpu",
                    "device_id": "cpu:0",
                },
            ],
            "placements": [
                {
                    "placement_id": f"{state}-authoritative",
                    "state": state,
                    "selector": {"whole_model": True},
                    "resource_id": "compute-0",
                    "role": "authoritative",
                }
                for state in states
            ],
            "offload_rules": [
                {
                    "rule_id": f"{state}-offload",
                    "state": state,
                    "selector": {"whole_model": True},
                    "source_resource_id": "compute-0",
                    "target_resource_id": "host-ram",
                    "mechanism": "cpu_copy",
                    "trigger": "memory_pressure",
                }
                for state in states
            ],
            "parallelism": {
                "world_size": 1,
                "ranks": [{"rank": 0, "resource_id": "compute-0"}],
            },
        }
    )


def _storage_physical(assessment, storage):
    return PhysicalExecutionSpec.model_validate(
        {
            "storage_profile_ref": storage_profile_ref_for(storage).model_dump(mode="json"),
            "resources": [
                {
                    "resource_id": "compute-0",
                    "tier": "gpu",
                    "device_kind": "cuda",
                    "device_id": "cuda:0",
                },
                {
                    "resource_id": "nvme-offload",
                    "tier": "nvme",
                    "storage": {
                        "role": "parameter_offload",
                        "path": "C:/offload",
                        "assessment": assessment.model_dump(mode="json"),
                        "accepted_suitability": assessment.suitability.value,
                    },
                },
            ],
            "placements": [
                {
                    "placement_id": "parameters-authoritative",
                    "state": "parameters",
                    "selector": {"whole_model": True},
                    "resource_id": "compute-0",
                    "role": "authoritative",
                }
            ],
            "offload_rules": [
                {
                    "rule_id": "parameter-offload",
                    "state": "parameters",
                    "selector": {"whole_model": True},
                    "source_resource_id": "compute-0",
                    "target_resource_id": "nvme-offload",
                    "mechanism": "nvme_io",
                    "trigger": "after_use",
                }
            ],
            "parallelism": {
                "world_size": 1,
                "ranks": [{"rank": 0, "resource_id": "compute-0"}],
            },
        }
    )


# ---- resolution paths -------------------------------------------------------


def test_planner_admits_the_dense_qlora_sft_task_and_refuses_unexecutable_variants():
    # #484 wired into planning: only the dense-QLoRA-SFT execution shape is workload_verified. sft maps
    # to it and the plan builds; a task whose shape the first-party harness cannot execute is refused
    # FAIL-CLOSED at planning - never silently downgraded to dense_qlora_sft.
    assert _plan(_profile(cc_major=8), _report(), task_type="sft").resolved_execution is not None
    # pretraining maps to a worker_implemented shape -> ADMITTED at planning (below workload_verified, so
    # still refused at execution); its dedicated builder then requires a corpus, so without one it fails
    # THERE, not at the variant admission gate.
    with pytest.raises(PlannerError, match="requires a corpus"):
        _plan(_profile(cc_major=8), _report(), task_type="pretraining")
    # a preference request must name its objective; without one it maps to no shape -> refused.
    with pytest.raises(PlannerError, match="no executable execution variant"):
        _plan(_profile(cc_major=8), _report(), task_type="preference")
    # preference + the dpo_qlora objective maps to preference_dpo (workload_verified) -> ADMITTED at
    # planning: the resolver seals a ResolvedPreferenceExecutionConfiguration and the plan carries it
    # (not the SFT resolved_execution), so the plan's loss summary is the DPO loss and it binds the
    # dpo_qlora objective. Execution now routes to the first-party PreferenceRunner lane (GPU bring-up).
    dpo_plan = _plan(_profile(cc_major=8), _report(), task_type="preference", objective_id="dpo_qlora")
    assert dpo_plan.resolved_preference_execution is not None
    assert dpo_plan.resolved_execution is None
    assert dpo_plan.loss_impl.value == "dpo"
    assert dpo_plan.resolved_preference_execution.objective_ref.id == "dpo_qlora"


def test_preference_dpo_resolves_to_a_sealed_config_and_routes_to_the_preference_lane():
    from corpus_studio.platform.execution_config import preference_execution_configuration_hash_for

    plan = _plan(_profile(cc_major=8), _report(), task_type="preference", objective_id="dpo_qlora")
    pref = plan.resolved_preference_execution
    assert pref is not None and plan.resolved_execution is None
    # the resolver sealed a self-consistent config bound to the dpo_qlora objective + the DPO loss/data
    assert pref.configuration_hash == preference_execution_configuration_hash_for(pref)
    assert pref.objective_ref.id == "dpo_qlora"
    assert pref.preference.objective == "dpo" and pref.data.schema_id == "preference"
    # the sealed data policy binds the PREFERENCE-pair formatter, not the SFT instruction formatter
    assert pref.data.formatter_id == "corpus-studio:preference-pair-v1"
    assert pref.data.max_prompt_length < pref.data.max_length  # room for the response
    # deferred #779 finding: device_map reconciles to exactly the one sealed compute device
    assert len(pref.device_map) == 1
    # preference_dpo is workload_verified (GPU bring-up), so the dispatch gate admits it and routes to the
    # first-party PreferenceRunner lane - never the SFT/pretraining lane (which never runs the DPO
    # reference / log-prob path).
    from corpus_studio.platform.execution_config import required_runner_lane

    assert required_runner_lane(plan) == "preference"


def test_full_parameter_sft_seals_a_full_model_config_and_routes_to_the_full_finetune_lane():
    from corpus_studio.platform.execution_config import (
        full_finetune_execution_configuration_hash_for,
        required_runner_lane,
    )

    # --adapter-method full_finetune (task=sft) seals the full-MODEL sibling config, NOT the adapter one.
    plan = _plan(
        _profile(cc_major=8), _report(), task_type="sft",
        adapter_method="full_finetune", export_format="merged_safetensors",
    )
    ff = plan.resolved_full_finetune_execution
    assert ff is not None and plan.resolved_execution is None
    assert ff.configuration_hash == full_finetune_execution_configuration_hash_for(ff)
    assert ff.adapter.method.value == "full_finetune"
    assert ff.export_format.value == "merged_safetensors"
    assert ff.checkpoint_policy.impl.value == "full_state"
    assert ff.precision.quantized_storage_format.value == "none"  # full-param is unquantized
    assert ff.objective_ref.id == "full_parameter_sft"
    # dense_full_finetune is workload_verified (the GPU bring-up), so the dispatch gate admits it and routes
    # to the first-party full-finetune lane - never the adapter SFT lane (which never trains a full model).
    assert required_runner_lane(plan) == "full_finetune"


def test_a_dense_qlora_sft_plan_is_unchanged_by_the_full_finetune_path():
    # the byte-locked adapter path stays: no full_finetune constraint -> the SFT adapter config, executable.
    plan = _plan(_profile(cc_major=8), _report(), task_type="sft")
    assert plan.resolved_execution is not None
    assert plan.resolved_full_finetune_execution is None
    assert plan.resolved_execution.adapter.method.value in {"lora", "qlora"}


def test_preference_resolver_threads_the_dpo_knobs():
    # beta / label_smoothing / max_prompt_length are operator knobs, not fixed defaults - they flow from
    # PlannerConstraints into the sealed preference config.
    plan = _plan(
        _profile(cc_major=8), _report(), task_type="preference", objective_id="dpo_qlora",
        preference_beta=0.3, preference_label_smoothing=0.2, preference_max_prompt_length=1234)
    pref = plan.resolved_preference_execution
    assert pref.preference.beta == 0.3 and pref.preference.label_smoothing == 0.2
    assert pref.data.max_prompt_length == 1234
    # unset -> documented defaults (half the window for the prompt cap)
    dpref = _plan(
        _profile(cc_major=8), _report(), task_type="preference", objective_id="dpo_qlora"
    ).resolved_preference_execution
    assert dpref.preference.beta == 0.1 and dpref.preference.label_smoothing == 0.0
    assert dpref.data.max_prompt_length == dpref.data.max_length // 2


def test_preference_resolver_rejects_invalid_dpo_knobs():
    # Fail-closed on out-of-range / non-finite knobs (a clean PlannerError, never a silent clamp).
    base = dict(task_type="preference", objective_id="dpo_qlora")
    for bad_beta in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(PlannerError, match="preference_beta"):
            _plan(_profile(cc_major=8), _report(), **base, preference_beta=bad_beta)
    with pytest.raises(PlannerError, match="label_smoothing"):
        _plan(_profile(cc_major=8), _report(), **base, preference_label_smoothing=0.5)
    for bad_cap in (0, -5, 999999):  # < 1 or >= the sequence window
        with pytest.raises(PlannerError, match="max_prompt_length"):
            _plan(_profile(cc_major=8), _report(), **base, preference_max_prompt_length=bad_cap)


def test_preference_resolver_refuses_an_incompatible_project_local_schema(monkeypatch):
    # A project-local 'preference' schema that makes a required pair field optional (or retypes it) is
    # rejected - the sealed chosen_rejected formatter needs prompt/chosen/rejected as required text.
    import corpus_studio.platform.planner as planner_mod
    from corpus_studio.schemas.registry import load_builtin_schema

    good = load_builtin_schema("preference")
    incompatible = good.model_copy(
        update={
            "fields": [
                field.model_copy(update={"required": False}) if field.name == "chosen" else field
                for field in good.fields
            ]
        }
    )
    monkeypatch.setattr(planner_mod, "resolve_schema", lambda _project_dir, _schema_id: (incompatible, "project"))
    with pytest.raises(PlannerError, match="incompatible with the chosen_rejected pair formatter"):
        _plan(_profile(cc_major=8), _report(), task_type="preference", objective_id="dpo_qlora")


def test_planner_refuses_a_preference_objective_on_a_non_preference_task():
    # A preference objective (dpo_qlora) with task_type=sft would silently lower to a dense-SFT run while
    # retaining a misleading DPO identity - refuse the objective/task contradiction fail-closed.
    with pytest.raises(PlannerError, match="requires task_type='preference'"):
        _plan(_profile(cc_major=8), _report(), task_type="sft", objective_id="dpo_qlora")


def test_native_windows_blackwell_host_forces_math_bf16_nf4_qlora():
    plan = _plan(_profile(cc_major=12, os="windows"), _report())
    assert plan.attention_backend.value == "math"  # native-Windows Blackwell (WDDM) mandate
    assert plan.precision.value == "bf16"
    assert plan.quantization.value == "nf4"
    assert plan.adapter.method.value == "qlora"
    assert plan.resolved_execution is not None
    attention = plan.resolved_execution.attention
    assert attention.model_attention_api.value == "sdpa"
    assert attention.effective_backend_required.value == "torch_sdpa_math"
    assert (attention.flash_sdp_enabled, attention.mem_efficient_sdp_enabled) == (False, False)
    assert attention.math_sdp_enabled is True
    assert attention.safety_mandate == "native_windows_blackwell_math_or_eager_only"


def test_native_windows_blackwell_refuses_math_without_a_passing_math_probe():
    report = _report(attn=("sdpa",), kernels=("torch_sdpa_flash",))
    with pytest.raises(PlannerError, match="no passing functional probe"):
        _plan(_profile(cc_major=12, os="windows"), report)


def test_managed_environment_lock_reference_is_sealed_into_plan():
    environment_ref = Ref(id="managed-env", hash=P.HashRef(value="b" * 64))
    plan = build_run_plan(
        profile=_profile(cc_major=8),
        capabilities=_report(),
        dataset_ref=Ref(id="ds-1", hash=P.HashRef(value="d" * 64)),
        constraints=PlannerConstraints(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            model_revision="1" * 40,
            dataset_path="data/examples.jsonl",
            dataset_content_sha256="d" * 64,
        ),
        plan_id="p-managed",
        environment_ref=environment_ref,
        now=_NOW,
    )
    assert plan.environment_ref == environment_ref


def test_managed_environment_refuses_version_only_trainer_package_evidence():
    environment_ref = Ref(id="managed-env", hash=P.HashRef(value="b" * 64))
    report = _report()
    downgraded = [
        item.model_copy(
            update={
                "hash": None,
                "artifact_hash": None,
                "record_integrity": "unknown",
                "record_entries": None,
                "record_verified_entries": None,
                "installed_files_hash": None,
                "installed_file_count": None,
            }
        )
        if item.name == "torch"
        else item
        for item in report.installed_packages
    ]
    with pytest.raises(
        PlannerError,
        match="managed trainer packages lack verified artifact, RECORD, or installed-file "
        "integrity evidence: torch",
    ):
        build_run_plan(
            profile=_profile(cc_major=8),
            capabilities=report.model_copy(update={"installed_packages": downgraded}),
            dataset_ref=Ref(id="ds-1", hash=P.HashRef(value="d" * 64)),
            constraints=PlannerConstraints(
                base_model="Qwen/Qwen2.5-7B-Instruct",
                model_revision="1" * 40,
                dataset_path="data/examples.jsonl",
                dataset_content_sha256="d" * 64,
            ),
            plan_id="p-managed",
            environment_ref=environment_ref,
            now=_NOW,
        )


def test_new_plans_seal_an_explicit_single_rank_physical_execution():
    plan = _plan(_profile(cc_major=8), _report())
    assert plan.physical_execution is not None
    assert is_trivial_physical_execution(plan.physical_execution)
    assert plan.physical_execution.evidence_status == "planned_not_measured"
    assert plan.physical_execution.resources[0].device_id == "cuda:0"
    assert plan.physical_execution.parallelism.world_size == 1
    assert verify_run_plan_hash(plan)


def test_cpu_toy_plan_resolves_an_explicit_cpu_resource():
    plan = _plan(
        _profile(),
        _report(readiness="cpu_toy_only", bnb=False),
        allow_cpu_toy=True,
    )
    assert plan.physical_execution is not None
    resource = plan.physical_execution.resources[0]
    assert resource.tier.value == "pageable_ram"
    assert resource.device_id == "cpu:0"


def test_scoped_physical_plan_consumes_a_verified_parameter_report_by_hash():
    report = _accounting_report()
    pinned = _validate_parameter_accounting(report, _scoped_physical())
    assert pinned.id == report.report_id
    assert pinned.hash.value == report.report_hash
    with pytest.raises(PlannerError, match="identity_scoped"):
        _plan(
            _profile(cc_major=8),
            _report(physical=True),
            physical_execution=_scoped_physical(),
            parameter_accounting=report,
        )


def test_scoped_physical_plan_refuses_missing_or_tampered_accounting_evidence():
    with pytest.raises(PlannerError, match="parameter-accounting"):
        _plan(
            _profile(cc_major=8),
            _report(physical=True),
            physical_execution=_scoped_physical(),
        )
    report = _accounting_report()
    with pytest.raises(PlannerError, match="absent from the sealed report"):
        _plan(
            _profile(cc_major=8),
            _report(physical=True),
            physical_execution=_scoped_physical("missing-scope"),
            parameter_accounting=report,
        )
    with pytest.raises(PlannerError, match="hash mismatch"):
        _plan(
            _profile(cc_major=8),
            _report(physical=True),
            physical_execution=_scoped_physical(),
            parameter_accounting=report.model_copy(update={"report_hash": "0" * 64}),
        )


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (("activations",), "controlled_activation_offload"),
        (("optimizer_state",), "controlled_optimizer_offload"),
        (("parameters",), "controlled_parameter_offload"),
        (("activations", "optimizer_state"), "cpu_offload"),
    ],
)
def test_offload_summary_preserves_the_planned_state_kind(states, expected):
    assert _offload_summary(_offload_physical(*states)).value == expected


def test_storage_and_accelerator_evidence_must_match_the_physical_spec():
    empty_storage = StorageProfile(captured_at=_NOW)
    with pytest.raises(PlannerError, match="uses no storage"):
        _plan(_profile(cc_major=8), _report(), storage_profile=empty_storage)

    marginal = StorageRoleAssessment(
        role="parameter_offload",
        path="C:/offload",
        suitability="marginal",
        interface="hdd",
        reasons=["rotational storage can bottleneck offload"],
    )
    marginal_profile = StorageProfile(captured_at=_NOW, assessments=[marginal])
    with pytest.raises(PlannerError, match="requires the exact StorageProfile"):
        _plan(
            _profile(cc_major=8),
            _report(),
            physical_execution=_storage_physical(marginal, marginal_profile),
        )

    with pytest.raises(PlannerError, match="assessment absent"):
        _plan(
            _profile(cc_major=8),
            _report(),
            physical_execution=_storage_physical(marginal, empty_storage),
            storage_profile=empty_storage,
        )

    unknown = StorageRoleAssessment.model_validate(
        {
            **marginal.model_dump(mode="json"),
            "suitability": "unknown",
            "interface": "unknown",
        }
    )
    unknown_profile = StorageProfile(captured_at=_NOW, assessments=[unknown])
    with pytest.raises(PlannerError, match="allow_unknown_storage"):
        _plan(
            _profile(cc_major=8),
            _report(),
            physical_execution=_storage_physical(unknown, unknown_profile),
            storage_profile=unknown_profile,
        )

    wrong_gpu_body = _scoped_physical().model_dump(mode="json")
    wrong_gpu_body["resources"][0]["device_id"] = "cuda:1"
    with pytest.raises(PlannerError, match="accelerator absent"):
        _plan(
            _profile(cc_major=8),
            _report(physical=True),
            physical_execution=PhysicalExecutionSpec.model_validate(wrong_gpu_body),
        )


def test_storage_backed_plan_requires_profile_match_and_explicit_marginal_acceptance():
    assessment = StorageRoleAssessment(
        role="parameter_offload",
        path="C:/offload",
        suitability="marginal",
        interface="hdd",
        reasons=["rotational storage can bottleneck offload"],
    )
    storage = StorageProfile(captured_at=_NOW, assessments=[assessment])
    physical = _storage_physical(assessment, storage)
    with pytest.raises(PlannerError, match="allow_marginal_storage"):
        _plan(
            _profile(cc_major=8),
            _report(),
            physical_execution=physical,
            storage_profile=storage,
        )
    with pytest.raises(PlannerError, match="can't run the physical plan"):
        _plan(
            _profile(cc_major=8),
            _report(),
            physical_execution=physical,
            storage_profile=storage,
            allow_marginal_storage=True,
        )
    changed = storage.model_copy(update={"captured_at": "2026-07-12T00:00:00Z"})
    with pytest.raises(PlannerError, match="does not match"):
        _plan(
            _profile(cc_major=8),
            _report(),
            physical_execution=physical,
            storage_profile=changed,
            allow_marginal_storage=True,
        )


def test_wsl_blackwell_host_keeps_sdpa_not_math():
    # WSL is its own platform: the flash deadlock is Windows-WDDM-only, so a WSL Blackwell host does
    # NOT force math — it seals the proven sdpa (→ flash on Linux CUDA). The whole reason to run under
    # WSL (verified on a real 5070 under WSL2).
    plan = _plan(_profile(cc_major=12, os="wsl"), _report(attn=("sdpa",)))
    assert plan.attention_backend.value == "sdpa"


def test_non_blackwell_with_proven_sdpa_uses_sdpa():
    plan = _plan(_profile(cc_major=8), _report(attn=("sdpa",)))
    assert plan.attention_backend.value == "sdpa"


# ---- memory / spill-avoidance levers flow through the platform ---------------


def test_independently_proven_optimizer_and_loss_are_refused_without_an_exact_tuple():
    with pytest.raises(PlannerError, match="complete requested execution tuple"):
        _plan(_profile(cc_major=8), _report(), optim="paged_adamw_8bit", use_liger=True)


def test_default_optim_and_no_liger():
    plan = _plan(_profile(cc_major=8), _report())
    assert plan.optimizer.impl.value == "adamw_torch"
    assert plan.loss_impl.value == "cross_entropy"
    assert plan.resolved_execution is not None
    assert plan.resolved_execution.optimizer.impl.value == "adamw_torch"
    assert plan.training_config_snapshot == {}


def test_new_plans_explicitly_disable_unresumable_intermediate_checkpoints():
    plan = _plan(_profile(cc_major=8), _report())
    execution = plan.resolved_execution
    assert execution is not None
    assert execution.save_strategy == "no"
    assert execution.checkpoint_policy.cadence_optimizer_steps is None
    assert execution.checkpoint_policy.keep_last is None
    assert "save_strategy" in execution.trainer_interface.required_sft_config_fields
    assert "save_steps" not in execution.trainer_interface.required_sft_config_fields
    assert "save_total_limit" not in execution.trainer_interface.required_sft_config_fields
    cfg = train_config_from_resolved(execution)
    assert cfg.save_strategy == "no"
    assert cfg.save_steps is None and cfg.save_total_limit is None


def test_allocator_max_split_size_is_sealed_into_the_plan():
    plan = _plan(
        _profile(cc_major=8), _report(),
        allocator_policy="max_split_size", allocator_max_split_size_mb=128,
    )
    assert plan.allocator_policy.value == "max_split_size"
    assert plan.allocator_max_split_size_mb == 128


def test_allocator_max_split_size_without_its_parameter_is_refused():
    with pytest.raises(PlannerError, match="requires a max_split_size_mb"):
        _plan(_profile(cc_major=8), _report(), allocator_policy="max_split_size")


def test_max_split_size_mb_without_the_matching_policy_is_refused():
    with pytest.raises(PlannerError, match="only valid with allocator_policy 'max_split_size'"):
        _plan(_profile(cc_major=8), _report(), allocator_max_split_size_mb=128)


def test_expandable_segments_with_a_paged_optimizer_is_refused():
    # the measured seq-4096 collision: expandable_segments + paged managed memory -> illegal access.
    with pytest.raises(PlannerError, match="collides with a paged optimizer"):
        _plan(
            _profile(cc_major=8), _report(),
            allocator_policy="expandable_segments", optim="paged_adamw_8bit",
        )


def test_resolved_execution_rejects_a_no_truncation_config_that_permits_truncation():
    # The no-silent-truncation cross-check: a config that declares no truncation
    # (sequence.truncation_allowed False) yet permits it at runtime (data.truncation_policy 'allow')
    # would silently truncate - the contract refuses it. Take a valid planner-built config and flip it.
    plan = _plan(_profile(cc_major=8), _report())
    execution = plan.resolved_execution
    assert execution is not None
    body = execution.model_dump(mode="json")
    body["sequence"]["truncation_allowed"] = False
    body["data"]["truncation_policy"] = "allow"
    with pytest.raises(ValidationError, match="silently truncate"):
        type(execution).model_validate(body)


@pytest.mark.parametrize(
    "checkpoint_overrides",
    [{"checkpoint_steps": 50}, {"checkpoint_keep_last": 3}],
)
def test_planner_refuses_checkpoint_requests_without_resume_lineage(checkpoint_overrides):
    with pytest.raises(PlannerError, match="resume compatibility and lineage"):
        _plan(_profile(cc_major=8), _report(), **checkpoint_overrides)


def test_invalid_optim_is_rejected():
    # optim is sealed as an Optimizer enum; a bogus value → a clean PlannerError, not a raw pydantic error.
    with pytest.raises(PlannerError, match="unsupported optimizer"):
        _plan(_profile(cc_major=8), _report(), optim="not_a_real_optimizer")


def test_resolved_execution_round_trips_as_a_trainrunconfig():
    plan = _plan(_profile(cc_major=8), _report())
    assert plan.resolved_execution is not None
    cfg = train_config_from_resolved(plan.resolved_execution)
    assert cfg.optim == "adamw_torch" and cfg.use_liger is False
    assert cfg.optimizer_state_dtype == "fp32"
    assert cfg.optimizer_auxiliary_dtype == "fp32"
    assert cfg.master_weight_dtype == "fp32" and cfg.gradient_dtype == "fp32"


def test_no_proven_attention_is_refused():
    with pytest.raises(PlannerError, match="not ready"):
        _plan(_profile(cc_major=8), _report(attn=(), kernels=()))


def test_bf16_not_proven_falls_back_to_fp32():
    plan = _plan(_profile(cc_major=8), _report(precisions=("fp16", "fp32")))
    assert plan.precision.value == "fp32"


def test_no_proven_training_precision_is_refused():
    with pytest.raises(PlannerError, match="not ready"):
        _plan(_profile(cc_major=8), _report(precisions=("fp16",)))


def test_no_bitsandbytes_gives_no_quant_and_lora():
    plan = _plan(_profile(cc_major=8), _report(bnb=False))
    assert plan.quantization.value == "none"
    assert plan.adapter.method.value == "lora"


def test_none_override_fails_closed_until_its_exact_combo_is_proven():
    # 'none' (16-bit on an unquantized base) needs no quantization proof, but the plan's COMPLETE tuple
    # (bf16, none, lora, ...) must still be demonstrated by a bounded probe. On a host that only proved the
    # nf4/qlora combo, a none override fails closed at the execution-tuple gate - the selector is reachable,
    # but honestly gated on a probe (exactly like int8), never sealing an un-demonstrated tuple.
    with pytest.raises(PlannerError, match="complete requested execution tuple"):
        _plan(_profile(cc_major=8), _report(), quantization="none")


def test_explicit_nf4_quantization_is_honored_when_proven():
    plan = _plan(_profile(cc_major=8), _report(), quantization="nf4")
    assert plan.quantization.value == "nf4"
    assert plan.adapter.method.value == "qlora"


def test_unproven_quantization_override_fails_closed():
    # int8 is DECLARED by the backend but NOT proven by the probe (only nf4 is): selecting it must fail
    # closed with a clear reason, never seal a plan that would break at execution ("declared" != "proven").
    with pytest.raises(PlannerError, match="not proven"):
        _plan(_profile(cc_major=8), _report(), quantization="int8")


def test_quantized_override_without_bitsandbytes_fails_closed():
    with pytest.raises(PlannerError, match="bitsandbytes"):
        _plan(_profile(cc_major=8), _report(bnb=False), quantization="nf4")


def test_full_finetune_rejects_a_quantized_override():
    with pytest.raises(PlannerError, match="cannot be quantized"):
        _plan(
            _profile(cc_major=8), _report(), task_type="sft",
            adapter_method="full_finetune", export_format="merged_safetensors",
            quantization="nf4",
        )


def test_none_override_is_honored_when_its_16bit_combo_is_proven():
    # Once a probe demonstrates the bf16/none/lora tuple (what the cuda_bf16_lora_math_execution GPU probe
    # proves), --quantization none seals a 16-bit LoRA plan on an unquantized base.
    plan = _plan(
        _profile(cc_major=8), _report(extra_combinations=(("none", "lora"),)), quantization="none"
    )
    assert plan.quantization.value == "none"
    assert plan.adapter.method.value == "lora"


def test_int8_override_is_honored_and_reads_as_qlora_when_proven():
    # Once a probe demonstrates the bf16/int8/qlora tuple (what cuda_int8_qlora_math_execution proves),
    # --quantization int8 seals - and the adapter resolves to qlora (a LoRA over a QUANTIZED base is
    # qlora for ANY quant type, not only nf4), matching the int8 execution probe's combination.
    plan = _plan(
        _profile(cc_major=8), _report(extra_combinations=(("int8", "qlora"),)), quantization="int8"
    )
    assert plan.quantization.value == "int8"
    assert plan.adapter.method.value == "qlora"


def test_explicit_unproven_attention_override_is_refused():
    with pytest.raises(PlannerError, match="not functionally proven"):
        _plan(_profile(cc_major=8), _report(), attention_backend="flash_attention_2")


def test_explicit_proven_sdpa_resolves_one_exact_kernel():
    plan = _plan(
        _profile(cc_major=8),
        _report(attn=("sdpa",)),
        attention_backend="sdpa",
    )
    assert plan.resolved_execution is not None
    policy = plan.resolved_execution.attention
    assert policy.effective_backend_required.value == "torch_sdpa_flash"
    assert (policy.flash_sdp_enabled, policy.mem_efficient_sdp_enabled, policy.math_sdp_enabled) == (
        True,
        False,
        False,
    )


def test_native_windows_blackwell_rejects_an_explicit_unsafe_attention_override():
    # The native-Windows Blackwell (WDDM) math mandate outranks the request. The fused/flash family
    # deadlocks outright there, and plain sdpa can DISPATCH to the flash kernel — only math/eager are
    # guaranteed safe under WDDM+sm_120.
    for unsafe in ("flash_attention_2", "mem_efficient", "sdpa"):
        with pytest.raises(PlannerError, match="deadlock"):
            _plan(_profile(cc_major=12, os="windows"), _report(), attention_backend=unsafe)


def test_wsl_blackwell_allows_an_explicit_sdpa_override():
    # On WSL the deadlock does not apply, so an explicit sdpa override is honored (not refused).
    plan = _plan(
        _profile(cc_major=12, os="wsl"),
        _report(attn=("sdpa",)),
        attention_backend="sdpa",
    )
    assert plan.attention_backend.value == "sdpa"


def test_native_windows_blackwell_allows_only_proven_math_and_eager_attention():
    math_plan = _plan(
        _profile(cc_major=12, os="windows"),
        _report(),
        attention_backend="math",
    )
    assert math_plan.attention_backend.value == "math"

    eager_report = _report(
        attn=("eager",),
        kernels=("eager",),
    )
    eager_plan = _plan(
        _profile(cc_major=12, os="windows"),
        eager_report,
        attention_backend="eager",
    )
    assert eager_plan.attention_backend.value == "eager"


def test_unsloth_refused_on_native_windows_blackwell_even_with_an_explicit_sdpa_override():
    # The "Unsloth refused on native-Windows sm_120" invariant must NOT be bypassable: an explicit sdpa
    # (which Unsloth declares) is itself refused there, so Unsloth can't be sealed by any attention path.
    with pytest.raises(PlannerError, match="deadlock"):
        _plan(_profile(cc_major=12, os="windows"), _report(), backend="unsloth", attention_backend="sdpa")


def test_unsupported_adapter_method_is_rejected():
    # dora / ia3 / full_finetune are in the enum but the corpus_studio backend declares only lora/qlora,
    # so the planner refuses rather than emit a plan that would be silently trained as plain LoRA.
    for method in ("dora", "ia3", "full_finetune"):
        with pytest.raises(PlannerError, match="adapter"):
            _plan(_profile(cc_major=12), _report(), adapter_method=method)


# ---- multi-backend selection ------------------------------------------------


def test_backend_without_resolved_execution_contract_is_refused():
    with pytest.raises(PlannerError, match="capability report belongs"):
        _plan(_profile(cc_major=8), _report(), backend="unsloth")


def test_unknown_backend_is_rejected():
    with pytest.raises(PlannerError, match="unknown backend"):
        _plan(_profile(cc_major=8), _report(), backend="megatron")


def test_backend_that_cannot_run_the_resolved_plan_is_rejected_with_alternatives():
    # Unsloth can't do the math attention a NATIVE-WINDOWS Blackwell plan requires → refused,
    # corpus_studio named. (On WSL the plan seals sdpa, which Unsloth CAN run — see the CLI test.)
    with pytest.raises(PlannerError, match="capability report belongs"):
        _plan(_profile(cc_major=12, os="windows"), _report(), backend="unsloth")


def test_default_backend_is_corpus_studio():
    plan = _plan(_profile(cc_major=12), _report())
    assert plan.backend_ref.id == "corpus_studio"


def test_lora_and_qlora_adapters_are_allowed():
    assert _plan(_profile(cc_major=8), _report(bnb=False), adapter_method="lora").adapter.method.value == "lora"
    assert _plan(_profile(cc_major=12), _report(), adapter_method="qlora").adapter.method.value == "qlora"


# ---- cpu-toy + readiness (honesty) ------------------------------------------


def test_cpu_toy_only_with_optin_yields_a_cpu_toy_plan():
    plan = _plan(_profile(), _report(readiness="cpu_toy_only", bnb=False), allow_cpu_toy=True)
    assert plan.precision.value == "fp32"
    assert plan.quantization.value == "none"
    assert plan.attention_backend.value == "eager"
    assert plan.resolved_execution is not None
    assert plan.resolved_execution.runtime_mode == "cpu_toy"
    assert plan.resolved_execution.schedule.max_steps == 3


def test_cpu_toy_only_without_optin_raises():
    with pytest.raises(PlannerError, match="cpu"):
        _plan(_profile(), _report(readiness="cpu_toy_only"))


def test_not_ready_raises_with_missing_packages():
    with pytest.raises(PlannerError, match="not ready"):
        _plan(_profile(), _report(readiness="not_ready", missing=["torch", "bitsandbytes"]))


def test_unsupported_task_type_raises():
    with pytest.raises(PlannerError, match="task_type"):
        _plan(_profile(cc_major=12), _report(), task_type="telepathy")


def test_unsupported_attention_override_raises():
    with pytest.raises(PlannerError, match="attention_backend"):
        _plan(_profile(cc_major=8), _report(), attention_backend="quantum")


# ---- sequence_len flows (no hardcoded calibration value) --------------------


def test_sequence_len_flows_verbatim():
    plan = _plan(_profile(cc_major=12), _report(), sequence_len=1792)
    assert plan.sequence.max_sequence_len == 1792
    assert plan.resolved_execution is not None
    assert plan.resolved_execution.sequence.max_sequence_len == 1792


# ---- the resolved configuration maps exactly to the trainer boundary --------


def test_resolved_execution_validates_as_a_trainrunconfig():
    plan = _plan(_profile(cc_major=12), _report())
    assert plan.resolved_execution is not None
    cfg = train_config_from_resolved(plan.resolved_execution)
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.dataset_format == "instruction"  # NOT silently defaulted from a wrong "format" key


def test_chat_format_requires_and_seals_exact_template_hash():
    with pytest.raises(ValueError, match="chat datasets require"):
        _plan(_profile(cc_major=12), _report(), dataset_format="chat")
    plan = _plan(
        _profile(cc_major=12),
        _report(),
        dataset_format="chat",
        chat_template_sha256="e" * 64,
    )
    assert plan.resolved_execution is not None
    assert plan.resolved_execution.data.chat_template_sha256 == "e" * 64


# ---- plan_hash (the immutability seal) --------------------------------------


def test_plan_hash_is_a_real_lowercase_sha256():
    plan = _plan(_profile(cc_major=12), _report())
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan
    assert len(plan.plan_hash) == 64
    assert plan.plan_hash == plan.plan_hash.lower()
    assert plan.plan_hash != "0" * 64


def test_plan_hash_excludes_the_volatile_created_at():
    a = _plan(_profile(cc_major=12), _report(), now="2026-01-01T00:00:00+00:00")
    b = _plan(_profile(cc_major=12), _report(), now="2027-12-31T23:59:59+00:00")
    assert a.created_at != b.created_at
    assert a.plan_hash == b.plan_hash  # identical plan body → identical seal


def test_plan_hash_changes_when_a_planned_field_changes():
    base = _plan(_profile(cc_major=12), _report())
    other = _plan(_profile(cc_major=12), _report(), learning_rate=1e-5)
    assert base.plan_hash != other.plan_hash


def test_plan_hash_seals_physical_execution_and_detects_tampering():
    plan = _plan(_profile(cc_major=8), _report())
    assert plan.physical_execution is not None
    changed_physical = plan.physical_execution.model_copy(
        update={"evidence_status": "planned_not_measured"}
    )
    # A semantic no-op copy stays valid; changing a real physical field does not.
    assert changed_physical == plan.physical_execution
    tampered_body = plan.model_dump(mode="json")
    tampered_body["physical_execution"]["resources"][0]["device_id"] = "cuda:9"
    with pytest.raises(ValueError, match="unplanned physical resource"):
        P.RunPlan.model_validate(tampered_body)


def test_legacy_hash_payload_omits_absent_physical_execution():
    from corpus_studio.platform.supervisor import demo_run_plan

    legacy = demo_run_plan()
    assert legacy.physical_execution is None
    assert "physical_execution" not in run_plan_hash_payload(legacy)
    assert verify_run_plan_hash(legacy)


def test_compute_plan_hash_is_order_independent():
    assert compute_plan_hash({"a": 1, "b": 2}) == compute_plan_hash({"b": 2, "a": 1})


# ---- linkage ----------------------------------------------------------------


def test_environment_ref_links_the_profile_signature():
    plan = _plan(_profile(cc_major=12), _report())
    assert plan.environment_ref.id == _SIG
    assert plan.dataset_ref.id == "ds-1"
    assert plan.backend_ref.id == "corpus_studio"


def test_default_clock_stamps_created_at():
    plan = _plan(_profile(cc_major=12), _report(), now=None)
    assert plan.created_at is not None
    assert plan.created_at.endswith("+00:00")


def test_an_invalid_resolved_field_becomes_planner_error():
    # sequence_len=0 fails SequenceSpec.max_sequence_len (ge=1) → a clean PlannerError, not a raw
    # pydantic ValidationError leaking out.
    with pytest.raises(PlannerError, match="invalid"):
        _plan(_profile(cc_major=12), _report(), sequence_len=0)


# ---- pretraining (S3a-2): from-scratch / continued planning through its own builder ----------------


def _pretraining_data(**over):
    base = dict(
        shards=(
            PretrainingShard(
                shard_id="s0",
                location="corpus/s0.jsonl",
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


def _pretraining_plan(profile, report, *, pretraining_data=None, **kw):
    kw.setdefault("base_model", "arch:demo-small")
    kw.setdefault("dataset_path", "corpus/manifest.json")
    kw.setdefault("task_type", "pretraining")
    kw.setdefault("init_mode", "random")
    kw.setdefault("architecture_ref_id", "arch:demo-small")
    kw.setdefault("architecture_ref_sha256", "c" * 64)
    kw.setdefault("init_vocab_size", 32000)
    kw.setdefault("tokenizer_source_mode", "train")
    kw.setdefault("tokenizer_algorithm", "bpe")
    kw.setdefault("tokenizer_vocab_size", 32000)
    kw.setdefault("tokenizer_special_tokens", ("<bos>", "<eos>", "<pad>", "<unk>"))
    kw.setdefault("export_format", "merged_safetensors")
    return build_run_plan(
        profile=profile,
        capabilities=report,
        dataset_ref=Ref(id="corpus", hash=P.HashRef(value="d" * 64)),
        constraints=PlannerConstraints(**kw),
        plan_id="pt1",
        now=_NOW,
        pretraining_data=_pretraining_data() if pretraining_data is None else pretraining_data,
    )


def test_pretraining_plan_seals_a_from_scratch_config():
    from corpus_studio.platform.execution_config import (
        verify_pretraining_execution_configuration_hash,
    )

    plan = _pretraining_plan(_profile(cc_major=8), _report())
    assert plan.task_type.value == "pretraining"
    assert plan.resolved_pretraining_execution is not None
    assert plan.resolved_execution is None and plan.resolved_preference_execution is None
    cfg = plan.resolved_pretraining_execution
    assert verify_pretraining_execution_configuration_hash(cfg)
    assert cfg.init.mode == "random" and cfg.objective_ref.id == "pretraining"
    assert cfg.tokenizer_source.mode == "train"
    # the base-model-less body synthesizes a full-parameter, unquantized summary
    assert plan.adapter.method.value == "full_finetune"
    assert plan.quantization.value == "none"
    assert plan.loss_impl.value in {"cross_entropy", "liger_fused_ce"}
    assert verify_run_plan_hash(plan)


def test_cpu_toy_pretraining_actually_trains_a_from_scratch_model(tmp_path):
    # S3b-1a inc 3a: the CPU proof - run_pretraining trains a random-init model with a from-scratch BPE
    # tokenizer over packed sequences. Skipped in the base gate (no torch); runs where the [train] libs are.
    import hashlib
    import json

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    from corpus_studio.training.pretraining_trainer import run_pretraining

    arch = tmp_path / "arch.json"
    arch.write_text(
        json.dumps(
            {"model_type": "gpt2", "n_embd": 32, "n_layer": 2, "n_head": 2, "n_inner": 64,
             "n_positions": 128, "vocab_size": 300}
        ),
        encoding="utf-8",
    )
    (tmp_path / "corpus").mkdir()
    shard = tmp_path / "corpus" / "s0.jsonl"
    shard.write_text(
        "\n".join(
            json.dumps({"text": "the quick brown fox jumps over the lazy dog . " * 6}) for _ in range(60)
        ),
        encoding="utf-8",
    )
    data = _pretraining_data(
        shards=(
            PretrainingShard(
                shard_id="s0", location="corpus/s0.jsonl", source="t", row_count=60, token_count=600,
                content_sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
            ),
        ),
        token_budget=600,
    )
    plan = _pretraining_plan(
        _profile(cc_major=8), _report(readiness="cpu_toy_only"), pretraining_data=data,
        architecture_ref_id=str(arch), architecture_ref_sha256=hashlib.sha256(arch.read_bytes()).hexdigest(),
        init_vocab_size=300, tokenizer_vocab_size=300,
        tokenizer_special_tokens=("<unk>", "<bos>", "<eos>", "<pad>"), sequence_len=32, allow_cpu_toy=True,
    )
    result = run_pretraining(
        plan.resolved_pretraining_execution, corpus_root=tmp_path, output_dir=str(tmp_path / "out")
    )
    assert result.cpu_toy and result.tokenizer_source == "trained"
    assert result.steps == plan.resolved_pretraining_execution.schedule.max_steps
    assert result.num_blocks > 0 and result.final_loss is not None
    assert (tmp_path / "out" / "model.safetensors").exists()


def test_pretraining_import_tokenizer_seals_the_location():
    # S3b-1a inc 3b: an import tokenizer is sealed with WHERE to load it (location) + its content digest.
    plan = _pretraining_plan(
        _profile(cc_major=8), _report(), tokenizer_source_mode="import",
        tokenizer_content_sha256="b" * 64, tokenizer_location="/tokenizers/mine",
    )
    ts = plan.resolved_pretraining_execution.tokenizer_source
    assert ts.mode == "import"
    assert ts.tokenizer_location == "/tokenizers/mine" and ts.tokenizer_content_sha256 == "b" * 64


def test_pretraining_plan_continued_binds_the_continued_objective():
    plan = _pretraining_plan(
        _profile(cc_major=8),
        _report(),
        init_mode="continued",
        source_checkpoint_ref_id="ckpt:base",
        source_checkpoint_ref_sha256="e" * 64,
        tokenizer_source_mode="freeze",
        tokenizer_content_sha256="f" * 64,
        architecture_ref_id=None,
        architecture_ref_sha256=None,
        init_vocab_size=None,
    )
    cfg = plan.resolved_pretraining_execution
    assert cfg.init.mode == "continued"
    assert cfg.objective_ref.id == "continued_pretraining"


def test_pretraining_plan_is_admitted_at_execution_and_routes_to_the_pretraining_lane():
    from corpus_studio.platform.execution_config import required_runner_lane

    # 'pretraining' is workload_verified, so the dispatch gate admits it and routes to the
    # first-party PretrainingRunner lane (never the SFT/DPO lane).
    plan = _pretraining_plan(_profile(cc_major=8), _report())
    assert required_runner_lane(plan) in ("pretraining", "pretraining_cpu_toy")


def test_pretraining_requires_a_corpus():
    with pytest.raises(PlannerError, match="requires a corpus"):
        build_run_plan(
            profile=_profile(cc_major=8),
            capabilities=_report(),
            dataset_ref=Ref(id="corpus", hash=P.HashRef(value="d" * 64)),
            constraints=PlannerConstraints(
                base_model="arch:demo",
                dataset_path="corpus/manifest.json",
                task_type="pretraining",
                init_mode="random",
                architecture_ref_id="arch:demo",
                architecture_ref_sha256="c" * 64,
                init_vocab_size=32000,
                tokenizer_source_mode="train",
                tokenizer_algorithm="bpe",
                tokenizer_vocab_size=32000,
                tokenizer_special_tokens=("<eos>",),
            ),
            plan_id="pt1",
            now=_NOW,
        )


def test_pretraining_random_init_requires_an_architecture():
    with pytest.raises(PlannerError, match="architecture ref"):
        _pretraining_plan(_profile(cc_major=8), _report(), architecture_ref_id=None)


def test_pretraining_train_tokenizer_requires_an_algorithm():
    with pytest.raises(PlannerError, match="algorithm"):
        _pretraining_plan(_profile(cc_major=8), _report(), tokenizer_algorithm=None)


def test_pretraining_fit_is_honestly_not_estimated_not_fabricated():
    # AUDIT fix: a full-parameter pretraining plan must NOT be run through the LoRA/QLoRA VRAM estimator
    # (which would fabricate a fit as if it were LoRA r16 over an HF base). The calibrator says so.
    from corpus_studio.platform.calibrator import classify_fit

    plan = _pretraining_plan(_profile(cc_major=8), _report())
    fit = classify_fit(plan, _profile(cc_major=8))
    assert fit.classification.name == "PLANNED_UNPROVEN"
    assert "pretraining" in fit.rationale and "not estimated" in fit.rationale


def test_pretraining_routes_to_its_lane_but_the_sft_runner_still_refuses_it():
    # The dispatch gate now routes pretraining to the PretrainingRunner lane; the SFT runner keeps a
    # defense-in-depth typed refusal so a pretraining plan can NEVER run the adapter path.
    from corpus_studio.platform.execution_config import required_runner_lane
    from corpus_studio.platform.runners import RunnerFailure, TrainingRunner

    plan = _pretraining_plan(_profile(cc_major=8), _report())
    assert required_runner_lane(plan) in ("pretraining", "pretraining_cpu_toy")
    with pytest.raises(RunnerFailure, match="PretrainingRunner lane"):
        TrainingRunner(cpu_toy=False)._resolve_trainer(plan)
    with pytest.raises(RunnerFailure, match="PretrainingRunner lane"):
        TrainingRunner(cpu_toy=False)._resolve_config(plan, "run-x")


def test_pretraining_model_vocab_defaults_to_the_trained_tokenizer_vocab():
    # AUDIT fix: a from-scratch model sizes its embedding to its tokenizer - the operator need not repeat
    # the vocab on both knobs, and a mismatch can never be sealed.
    plan = _pretraining_plan(
        _profile(cc_major=8), _report(), init_vocab_size=None, tokenizer_vocab_size=50000
    )
    cfg = plan.resolved_pretraining_execution
    assert cfg.init.vocab_size == 50000 and cfg.tokenizer_source.vocab_size == 50000
