"""Platform slice 6 — the run planner. Pure tests (no torch): synthetic EnvironmentProfile +
CapabilityReport drive every resolution path and the honesty invariants (Blackwell→math from
cc_major, proven-only precision/quant, sequence_len flows, cpu_toy never a silent downgrade, a real
sha256 plan-hash that excludes the volatile stamp). The rendered snapshot is round-tripped through
the actual TrainRunConfig the runner replays."""

import pytest

import corpus_studio.platform as P
from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvironmentProfile,
    EnvHost,
    GpuDevice,
    ParameterAccountingReport,
    ParameterEvidenceGap,
    ParameterScope,
    ParameterWindow,
    PhysicalExecutionSpec,
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
from corpus_studio.training.trainer import TrainRunConfig

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
    precisions=("bf16",),
    attn=("sdpa",),
    missing=(),
    physical=False,
):
    eff = EffectiveCapabilities(
        precision_modes=list(precisions),
        quantization_modes=["nf4"] if bnb else [],
        attention_impls=list(attn),
        adapter_methods=["qlora"],
        placement_tiers=["gpu"] if physical else [],
        placement_modes=["single_resource"] if physical else [],
    )
    return CapabilityReport(
        backend_id="corpus_studio", environment_ref=Ref(id=_SIG), readiness=readiness,
        bitsandbytes_ok=bnb, effective_capabilities=eff, missing_packages=list(missing),
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
    constraints = PlannerConstraints(**kw)
    return build_run_plan(
        profile=profile, capabilities=report, dataset_ref=Ref(id="ds-1"),
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


def test_native_windows_blackwell_host_forces_math_bf16_nf4_qlora():
    plan = _plan(_profile(cc_major=12, os="windows"), _report())
    assert plan.attention_backend.value == "math"  # native-Windows Blackwell (WDDM) mandate
    assert plan.precision.value == "bf16"
    assert plan.quantization.value == "nf4"
    assert plan.adapter.method.value == "qlora"
    # math is not a from_pretrained string → the snapshot leaves the trainer's own path in control.
    assert "attn_implementation" not in plan.training_config_snapshot


def test_managed_environment_lock_reference_is_sealed_into_plan():
    environment_ref = Ref(id="managed-env", hash=P.HashRef(value="b" * 64))
    plan = build_run_plan(
        profile=_profile(cc_major=8),
        capabilities=_report(),
        dataset_ref=Ref(id="ds-1"),
        constraints=PlannerConstraints(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            dataset_path="data/examples.jsonl",
        ),
        plan_id="p-managed",
        environment_ref=environment_ref,
        now=_NOW,
    )
    assert plan.environment_ref == environment_ref


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


def test_optim_and_liger_flow_into_the_plan_and_snapshot():
    # The avoid-spill levers reach the SEALED plan (validated against the backend) AND the training
    # snapshot the trainer replays — so `platform-run` (not just `train-run`) gets them.
    plan = _plan(_profile(cc_major=8), _report(), optim="paged_adamw_8bit", use_liger=True)
    assert plan.optimizer.impl.value == "paged_adamw_8bit"
    assert plan.loss_impl.value == "liger_fused_ce"
    assert plan.training_config_snapshot["optim"] == "paged_adamw_8bit"
    assert plan.training_config_snapshot["use_liger"] is True


def test_default_optim_and_no_liger():
    plan = _plan(_profile(cc_major=8), _report())
    assert plan.optimizer.impl.value == "adamw_torch"
    assert plan.loss_impl.value == "cross_entropy"
    assert plan.training_config_snapshot["optim"] == "adamw_torch"
    assert "use_liger" not in plan.training_config_snapshot  # opt-in — absent by default


def test_invalid_optim_is_rejected():
    # optim is sealed as an Optimizer enum; a bogus value → a clean PlannerError, not a raw pydantic error.
    with pytest.raises(PlannerError, match="invalid"):
        _plan(_profile(cc_major=8), _report(), optim="not_a_real_optimizer")


def test_snapshot_with_levers_round_trips_as_a_trainrunconfig():
    plan = _plan(_profile(cc_major=8), _report(), optim="paged_adamw_8bit", use_liger=True)
    cfg = TrainRunConfig.model_validate(plan.training_config_snapshot)
    assert cfg.optim == "paged_adamw_8bit" and cfg.use_liger is True


def test_no_proven_attention_falls_back_to_eager():
    plan = _plan(_profile(cc_major=8), _report(attn=()))
    assert plan.attention_backend.value == "eager"
    assert plan.training_config_snapshot["attn_implementation"] == "eager"


def test_bf16_not_proven_falls_back_to_fp32():
    plan = _plan(_profile(cc_major=8), _report(precisions=("fp16",)))
    assert plan.precision.value == "fp32"


def test_no_bitsandbytes_gives_no_quant_and_lora():
    plan = _plan(_profile(cc_major=8), _report(bnb=False))
    assert plan.quantization.value == "none"
    assert plan.adapter.method.value == "lora"


def test_explicit_attention_override_wins():
    plan = _plan(_profile(cc_major=8), _report(), attention_backend="flash_attention_2")
    assert plan.attention_backend.value == "flash_attention_2"
    assert plan.training_config_snapshot["attn_implementation"] == "flash_attention_2"


def test_native_windows_blackwell_rejects_an_explicit_unsafe_attention_override():
    # The native-Windows Blackwell (WDDM) math mandate outranks the request. The fused/flash family
    # deadlocks outright there, and plain sdpa can DISPATCH to the flash kernel — only math/eager are
    # guaranteed safe under WDDM+sm_120.
    for unsafe in ("flash_attention_2", "mem_efficient", "sdpa"):
        with pytest.raises(PlannerError, match="deadlock"):
            _plan(_profile(cc_major=12, os="windows"), _report(), attention_backend=unsafe)


def test_wsl_blackwell_allows_an_explicit_sdpa_override():
    # On WSL the deadlock does not apply, so an explicit sdpa override is honored (not refused).
    plan = _plan(_profile(cc_major=12, os="wsl"), _report(), attention_backend="sdpa")
    assert plan.attention_backend.value == "sdpa"


def test_native_windows_blackwell_allows_only_math_and_eager_explicit_attention():
    for safe in ("eager", "math"):
        plan = _plan(_profile(cc_major=12, os="windows"), _report(), attention_backend=safe)
        assert plan.attention_backend.value == safe


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


def test_backend_ref_reflects_the_chosen_backend():
    # Unsloth on a non-Blackwell host with a proven sdpa plan.
    plan = _plan(_profile(cc_major=8), _report(attn=("sdpa",)), backend="unsloth")
    assert plan.backend_ref.id == "unsloth"


def test_unknown_backend_is_rejected():
    with pytest.raises(PlannerError, match="unknown backend"):
        _plan(_profile(cc_major=8), _report(), backend="megatron")


def test_backend_that_cannot_run_the_resolved_plan_is_rejected_with_alternatives():
    # Unsloth can't do the math attention a NATIVE-WINDOWS Blackwell plan requires → refused,
    # corpus_studio named. (On WSL the plan seals sdpa, which Unsloth CAN run — see the CLI test.)
    with pytest.raises(PlannerError, match="can't run this plan"):
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
    assert plan.training_config_snapshot["cpu_toy"] is True


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
    assert plan.training_config_snapshot["sequence_len"] == 1792


# ---- the snapshot round-trips through the real trainer config ---------------


def test_snapshot_validates_as_a_trainrunconfig():
    plan = _plan(_profile(cc_major=12), _report())
    cfg = TrainRunConfig.model_validate(plan.training_config_snapshot)
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.dataset_format == "instruction"  # NOT silently defaulted from a wrong "format" key


def test_snapshot_uses_dataset_format_key_not_format():
    plan = _plan(_profile(cc_major=12), _report(), dataset_format="chat")
    assert plan.training_config_snapshot["dataset_format"] == "chat"
    assert "format" not in plan.training_config_snapshot


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
    tampered = P.RunPlan.model_validate(tampered_body)
    assert not verify_run_plan_hash(tampered)


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
