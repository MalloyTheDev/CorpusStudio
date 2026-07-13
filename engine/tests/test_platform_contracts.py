"""Tests for the language-neutral platform contracts (corpus_studio.platform).

Pure — these run in CI without torch/transformers. They lock the substrate the whole platform
migration hangs on: the contracts import torch-free, round-trip through JSON, forbid unknown fields,
enforce the id/hash patterns and the supervised-token accumulation target, carry the dedicated-vs-
shared memory split + the spill/failure taxonomies, and export to language-neutral JSON Schema.
"""

from __future__ import annotations

import json
import sys

import pytest
from pydantic import ValidationError

import corpus_studio.platform as P
from corpus_studio.platform.contracts import BatchingSpec, RunDispatchBody, WORKER_BODY_BY_TYPE
from corpus_studio.platform.enums import FailureTaxonomy, FitClass, StageMarker


# ---- import boundary ---------------------------------------------------------


def test_platform_import_is_torch_free():
    # The contracts substrate must never pull the heavy training stack (the dependency-light gate).
    for heavy in ("torch", "transformers", "trl", "peft", "bitsandbytes"):
        assert heavy not in sys.modules, f"platform import pulled {heavy}"


# ---- version + registry ------------------------------------------------------


def test_contract_version_is_pinned():
    assert P.CONTRACT_VERSION == "1.0.0"


def test_all_root_contracts_registered():
    assert len(P.ROOT_CONTRACTS) == 28
    assert "StorageProfile" in P.ROOT_CONTRACTS
    for expected in (
        "ModelDescriptor",
        "ParameterAccountingReport",
        "TokenizerDescriptor",
        "TraceRecord",
        "TrainingObjective",
        "ObjectiveCompatibilityReport",
        "PythonRuntime",
        "EnvironmentRecipe",
        "DependencyResolution",
        "EnvironmentInstallation",
        "EnvironmentLock",
        "EnvironmentDescriptor",
        "EnvironmentHealthReport",
        "ResolvedExecutionConfiguration",
    ):
        assert expected in P.ROOT_CONTRACTS
    for name, model in P.ROOT_CONTRACTS.items():
        if name == "WorkerMessage":
            # the wire envelope negotiates its own protocol_version, independent of any single
            # contract's contract_version (so the two sides can version separately).
            assert "protocol_version" in model.model_fields
            continue
        field = model.model_fields.get("contract_version")
        assert field is not None, f"{name} missing contract_version"
        assert field.default == "1.0.0"


# ---- taxonomies --------------------------------------------------------------


def test_failure_and_fit_taxonomies_complete():
    assert len(FailureTaxonomy) == 11
    assert len(FitClass) == 12
    assert len(StageMarker) == 20
    # the spill-vs-OOM distinction that is the whole point
    assert FailureTaxonomy.KERNEL_STALL.value == "KERNEL_STALL"
    assert FailureTaxonomy.ACCIDENTAL_SPILL.value == "ACCIDENTAL_SPILL"
    assert FitClass.ACCIDENTAL_WDDM_SPILL.value == "ACCIDENTAL_WDDM_SPILL"
    assert FitClass.PLANNED_UNPROVEN.value == "PLANNED_UNPROVEN"
    assert FitClass.NATIVE_SAFE.value == "NATIVE_SAFE"


# ---- memory + tokens ---------------------------------------------------------


def test_memory_metrics_splits_dedicated_and_shared():
    m = P.MemoryMetrics(dedicated_gpu_bytes=11_000_000_000, shared_gpu_bytes=3_000_000_000)
    assert m.shared_gpu_bytes and m.shared_gpu_bytes > 0  # the accidental-spill fingerprint
    assert P.MemoryMetrics.model_validate_json(m.model_dump_json()) == m


def test_token_stats_supervised_and_no_truncation_default():
    ts = P.TokenStats(example_count=450, supervised_tokens=900_000)
    assert ts.no_truncation is False
    assert ts.supervised_tokens == 900_000
    with pytest.raises(ValidationError):
        P.TokenStats(example_count=-1)  # ge=0


# ---- helpers -----------------------------------------------------------------


def _valid_run_plan(**over) -> P.RunPlan:
    base = dict(
        plan_id="plan-1",
        plan_hash="0" * 64,
        backend_ref=P.Ref(id="corpus_studio_trl"),
        environment_ref=P.Ref(id="a" * 64),
        dataset_ref=P.Ref(id="20260710-abc"),
        task_type="sft",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        precision="bf16",
        quantization="nf4",
        adapter={"method": "qlora", "lora_r": 16, "lora_alpha": 32},
        optimizer={"impl": "paged_adamw_8bit", "learning_rate": 2e-4},
        loss_impl="liger_fused_ce",
        attention_backend="math",
        sequence={"max_sequence_len": 2048},
        batching={"micro_batch_size": 1, "supervised_token_accumulation_target": 16384},
        checkpoint_policy={"impl": "adapter_only"},
        export={"format": "adapter_peft"},
    )
    base.update(over)
    return P.RunPlan(**base)


# ---- RunPlan -----------------------------------------------------------------


def test_run_plan_roundtrip():
    rp = _valid_run_plan()
    assert P.RunPlan.model_validate_json(rp.model_dump_json()) == rp
    assert rp.seed == 42  # default
    assert rp.gradient_checkpointing is True


def test_run_plan_requires_supervised_token_target():
    # the accumulation target is expressed in supervised tokens, not a microbatch count
    with pytest.raises(ValidationError):
        BatchingSpec(micro_batch_size=1)  # missing supervised_token_accumulation_target


def test_run_plan_hash_must_be_sha256():
    with pytest.raises(ValidationError):
        _valid_run_plan(plan_hash="not-a-hash")


def _singleton_physical(*, selector=None):
    return P.PhysicalExecutionSpec(
        resources=[
            P.PhysicalResource(
                resource_id="compute-0", tier="gpu", device_kind="cuda", device_id="cuda:0"
            )
        ],
        placements=[
            P.StatePlacement(
                placement_id="parameters-authoritative",
                state="parameters",
                selector=selector or {"whole_model": True},
                resource_id="compute-0",
                role="authoritative",
            )
        ],
        parallelism=P.ParallelismSpec(
            world_size=1,
            ranks=[P.RankBinding(rank=0, resource_id="compute-0")],
        ),
    )


def _optimizer_offload_physical(*, route_miss_action="fail", **over):
    body = {
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
                "placement_id": "optimizer-authoritative",
                "state": "optimizer_state",
                "selector": {"whole_model": True},
                "resource_id": "compute-0",
                "role": "authoritative",
            }
        ],
        "offload_rules": [
            {
                "rule_id": "optimizer-offload",
                "state": "optimizer_state",
                "selector": {"whole_model": True},
                "source_resource_id": "compute-0",
                "target_resource_id": "host-ram",
                "mechanism": "cpu_copy",
                "trigger": "memory_pressure",
                "route_miss_action": route_miss_action,
            }
        ],
        "parallelism": {
            "world_size": 1,
            "ranks": [{"rank": 0, "resource_id": "compute-0"}],
        },
        **over,
    }
    return P.PhysicalExecutionSpec.model_validate(body)


def test_run_plan_physical_execution_is_planned_not_residency_evidence():
    physical = _singleton_physical()
    plan = _valid_run_plan(physical_execution=physical)
    assert plan.physical_execution is not None
    assert plan.physical_execution.evidence_status == "planned_not_measured"
    assert plan.physical_execution.route_fidelity == "preserve_or_fail"
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan


def test_scope_specific_placement_requires_pinned_parameter_report():
    physical = _singleton_physical(selector={"parameter_scope_ids": ["experts.layer0"]})
    with pytest.raises(ValidationError, match="parameter-accounting"):
        _valid_run_plan(physical_execution=physical)
    plan = _valid_run_plan(
        physical_execution=physical,
        parameter_accounting_ref={"id": "parameter-report", "hash": {"value": "a" * 64}},
    )
    assert plan.parameter_accounting_ref is not None


def test_explicit_offload_rules_must_agree_with_compatibility_summary():
    physical = _optimizer_offload_physical()
    with pytest.raises(ValidationError, match="offload_strategy"):
        _valid_run_plan(physical_execution=physical)
    plan = _valid_run_plan(
        physical_execution=physical,
        offload_strategy="controlled_optimizer_offload",
    )
    assert len(plan.physical_execution.offload_rules) == 1


def test_semantic_route_fallback_requires_a_pinned_model_policy():
    with pytest.raises(ValidationError, match="semantic route fallback"):
        _optimizer_offload_physical(route_miss_action="semantic_fallback")
    physical = _optimizer_offload_physical(
        route_miss_action="semantic_fallback",
        route_fidelity="declared_semantic_fallback",
        semantic_fallback_policy_ref={"id": "router-policy", "hash": {"value": "b" * 64}},
    )
    assert physical.route_fidelity == "declared_semantic_fallback"


def test_storage_resource_refuses_unsuitable_and_requires_explicit_risk_acceptance():
    assessment = {
        "role": "parameter_offload",
        "path": "C:/offload",
        "suitability": "unsuitable",
        "reasons": ["inside source repository"],
    }
    with pytest.raises(ValidationError, match="unsuitable"):
        P.PhysicalResource(
            resource_id="nvme",
            tier="nvme",
            storage={
                "role": "parameter_offload",
                "path": "C:/offload",
                "assessment": assessment,
                "accepted_suitability": "unsuitable",
            },
        )
    assessment["suitability"] = "marginal"
    with pytest.raises(ValidationError, match="accepted_suitability"):
        P.PhysicalResource(
            resource_id="nvme",
            tier="nvme",
            storage={
                "role": "parameter_offload",
                "path": "C:/offload",
                "assessment": assessment,
            },
        )
    resource = P.PhysicalResource(
        resource_id="nvme",
        tier="nvme",
        storage={
            "role": "parameter_offload",
            "path": "C:/offload",
            "assessment": assessment,
            "accepted_suitability": "marginal",
        },
    )
    assert resource.storage is not None


def test_parallel_groups_are_explicit_partitions_with_stable_expert_scopes():
    with pytest.raises(ValidationError, match="explicit communication"):
        P.ParallelGroup(
            group_id="data-0", kind="data", ranks=[0, 1], communication_backend="none"
        )
    with pytest.raises(ValidationError, match="stable parameter scope"):
        P.ParallelGroup(
            group_id="experts-0", kind="expert", ranks=[0, 1], communication_backend="nccl"
        )
    parallel = P.ParallelismSpec(
        world_size=2,
        ranks=[
            P.RankBinding(rank=0, resource_id="gpu-0", local_rank=0),
            P.RankBinding(rank=1, resource_id="gpu-1", local_rank=1),
        ],
        groups=[
            P.ParallelGroup(
                group_id="experts-0",
                kind="expert",
                ranks=[0, 1],
                communication_backend="nccl",
                parameter_scope_ids=["experts.layer0"],
            )
        ],
    )
    assert parallel.world_size == 2


def test_contracts_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        P.FitClassification(classification="NATIVE_SAFE", bogus=1)


# ---- RunEvent ----------------------------------------------------------------


def test_run_event_roundtrip_with_fit_reclassification():
    ev = P.RunEvent(
        event_type="metric",
        run_id="r1",
        seq=31,
        emitted_at="2026-07-10T00:00:00Z",
        stage="backward",
        optimizer_step=2,
        metrics={
            "memory": {"dedicated_gpu_bytes": 11_800_000_000, "shared_gpu_bytes": 400_000_000},
            "step_time_seconds": 44.3,
            "supervised_tokens_per_sec": 618.2,
            "loss": 1.842,
        },
        fit={"classification": "ACCIDENTAL_WDDM_SPILL", "rationale": "shared bytes non-zero"},
    )
    back = P.RunEvent.model_validate_json(ev.model_dump_json())
    assert back == ev
    assert back.fit is not None and back.fit.classification == FitClass.ACCIDENTAL_WDDM_SPILL


# ---- FailureRecord -----------------------------------------------------------


def test_failure_record_requires_taxonomy_and_message():
    fr = P.FailureRecord(taxonomy="KERNEL_STALL", message="fused attn deadlock on sm_120")
    assert P.FailureRecord.model_validate_json(fr.model_dump_json()) == fr
    with pytest.raises(ValidationError):
        P.FailureRecord(message="no taxonomy")


# ---- DatasetManifest ---------------------------------------------------------


def test_dataset_manifest_lineage_roundtrip():
    dm = P.DatasetManifest(
        version_id="20260710-tex",
        row_count=450,
        content_fingerprint="a" * 64,
        token_stats={"example_count": 450, "supervised_tokens": 900_000, "no_truncation": True},
        lineage={
            "transformation_pipeline": [
                {"step": "clean", "tool": "corpus_studio.exporters.cleaning"}
            ],
            "generation": {"teacher_model": "glm-5.2:cloud", "random_seed": 42},
        },
    )
    assert P.DatasetManifest.model_validate_json(dm.model_dump_json()) == dm


def test_dataset_manifest_content_fingerprint_must_be_sha256():
    with pytest.raises(ValidationError):
        P.DatasetManifest(version_id="v1", row_count=1, content_fingerprint="short")


def test_project_manifest_id_pattern():
    P.ProjectManifest(id="wbg-7b", name="WBG", schema_id="chat")
    with pytest.raises(ValidationError):
        P.ProjectManifest(id="Bad Id!", name="x", schema_id="chat")


# ---- WorkerProtocol ----------------------------------------------------------


def test_worker_message_body_by_type_covers_all_types():
    # every message type maps to a body model (or a reused contract)
    for t in (
        "hello",
        "capability_probe_request",
        "capability_report",
        "run_dispatch",
        "run_accepted",
        "run_rejected",
        "run_control",
        "event",
        "heartbeat",
        "terminal_result",
        "failure",
    ):
        assert t in WORKER_BODY_BY_TYPE


def test_worker_message_run_dispatch_roundtrip_and_body_parse():
    plan = _valid_run_plan()
    dispatch = RunDispatchBody(run_id="r1", plan=plan)
    msg = P.WorkerMessage(
        protocol_version="2.0.0",
        message_id="m1",
        direction="core_to_worker",
        type="run_dispatch",
        body=dispatch.model_dump(mode="json"),
    )
    back = P.WorkerMessage.model_validate_json(msg.model_dump_json())
    assert back == msg
    # the body parses back into the type the discriminator selects
    parsed = WORKER_BODY_BY_TYPE[back.type].model_validate(back.body)
    assert isinstance(parsed, RunDispatchBody)
    assert parsed.plan == plan


def test_worker_protocol_version_pattern():
    with pytest.raises(ValidationError):
        P.WorkerMessage(
            protocol_version="v1", message_id="m", direction="core_to_worker", type="heartbeat"
        )


def test_worker_schema_exposes_required_v2_typed_body_union():
    schema = P.WorkerMessage.model_json_schema()
    assert schema["properties"]["protocol_version"]["const"] == "2.0.0"
    assert {"protocol_version", "message_id", "direction", "type", "body"} <= set(
        schema["required"]
    )
    refs = {
        item["$ref"].rsplit("/", 1)[-1]
        for item in schema["properties"]["body"]["anyOf"]
    }
    assert {"HelloBody", "RunDispatchBody", "RunEvent", "TerminalResultBody"} <= refs


def test_worker_message_rejects_wrong_direction_and_body_shape():
    with pytest.raises(ValidationError, match="requires direction"):
        P.WorkerMessage(
            protocol_version="2.0.0",
            message_id="m1",
            direction="worker_to_core",
            type="run_dispatch",
            body={"run_id": "r", "plan": _valid_run_plan().model_dump(mode="json")},
        )
    with pytest.raises(ValidationError):
        P.WorkerMessage(
            protocol_version="2.0.0",
            message_id="m2",
            direction="worker_to_core",
            type="heartbeat",
            body={"run_id": "r", "pid_alive": "not-a-bool"},
        )


# ---- schema export -----------------------------------------------------------


def test_export_json_schemas_writes_language_neutral_files(tmp_path):
    written = P.export_json_schemas(tmp_path)
    # 28 contract schemas + index.json
    assert len(written) == 29
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["contract_version"] == "1.0.0"
    assert len(index["contracts"]) == 28
    # every emitted schema is valid JSON with a proper object shape
    for name in P.ROOT_CONTRACTS:
        schema = json.loads((tmp_path / f"{name}.schema.json").read_text(encoding="utf-8"))
        assert schema["type"] == "object"
        assert "properties" in schema
    # the RunPlan schema carries the supervised-token target (via the BatchingSpec $def)
    run_plan_schema = P.contract_schemas()["RunPlan"]
    assert "BatchingSpec" in run_plan_schema.get("$defs", {})
    assert (
        "supervised_token_accumulation_target"
        in run_plan_schema["$defs"]["BatchingSpec"]["properties"]
    )
    assert "PhysicalExecutionSpec" in run_plan_schema["$defs"]
    physical = run_plan_schema["$defs"]["PhysicalExecutionSpec"]["properties"]
    assert physical["evidence_status"]["const"] == "planned_not_measured"
    assert "resources" in physical and "offload_rules" in physical and "parallelism" in physical
    selector = run_plan_schema["$defs"]["PhysicalScopeSelector"]["properties"]
    assert "parameter_scope_ids" in selector and "expert_ids" in selector
    # Descriptor schemas preserve the fail-closed trust boundary and portable evidence shapes.
    model_schema = P.contract_schemas()["ModelDescriptor"]
    trust_remote_code = model_schema["$defs"]["TrustRequirement"]["properties"][
        "trust_remote_code"
    ]
    assert trust_remote_code["const"] is False
    descriptor_path = model_schema["$defs"]["DescriptorFile"]["properties"]["path"]
    assert "pattern" in descriptor_path
    assert "\\\\" in descriptor_path["pattern"]
    resolved_commit = model_schema["$defs"]["DescriptorSource"]["properties"][
        "resolved_commit"
    ]
    assert resolved_commit["anyOf"][0]["pattern"] == "^[0-9a-f]{7,64}$"
    representation = model_schema["$defs"]["ParameterRepresentation"]["properties"]
    assert "counts" in representation
    assert "parameter_count" not in representation
    topology = model_schema["$defs"]["ModelTopology"]["properties"]
    assert "inspection" in topology and "expert_counts" in topology
    inspection = model_schema["$defs"]["TopologyInspection"]["properties"]
    assert inspection["runtime_capability"]["const"] == "unverified"
    assert inspection["evidence_level"]["default"] == "not_checked"
    expert_group = model_schema["$defs"]["ExpertGroup"]
    assert "routed_expert_count" in expert_group["properties"]
    assert "routed_expert_count" not in expert_group.get("required", [])
    assert "shared_expert_count" not in expert_group.get("required", [])
    assert "component_path" in expert_group["properties"]
    expert_counts = model_schema["$defs"]["ExpertTopologyCounts"]["properties"]
    assert expert_counts["unit"]["const"] == "expert_instances"
    assert "active_expert_instances_per_token" in expert_counts
    # Objective schemas preserve independent loss components/masks and the no-fit-claim boundary.
    objective_schema = P.contract_schemas()["TrainingObjective"]
    objective_properties = objective_schema["properties"]
    assert "loss_components" in objective_properties
    assert "loss_masks" in objective_properties
    fit_claim = objective_schema["$defs"]["ObjectiveHardwareImplications"]["properties"][
        "fit_claim"
    ]
    assert fit_claim["const"] == "none"


def test_export_json_schemas_is_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    P.export_json_schemas(first)
    P.export_json_schemas(second)

    first_files = sorted(path.name for path in first.iterdir())
    second_files = sorted(path.name for path in second.iterdir())
    assert first_files == second_files
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()
