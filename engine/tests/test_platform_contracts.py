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
    assert len(P.ROOT_CONTRACTS) == 25
    assert "StorageProfile" in P.ROOT_CONTRACTS
    for expected in (
        "ModelDescriptor",
        "TokenizerDescriptor",
        "TrainingObjective",
        "ObjectiveCompatibilityReport",
        "PythonRuntime",
        "EnvironmentRecipe",
        "DependencyResolution",
        "EnvironmentInstallation",
        "EnvironmentLock",
        "EnvironmentDescriptor",
        "EnvironmentHealthReport",
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
    assert len(FitClass) == 11
    assert len(StageMarker) == 16
    # the spill-vs-OOM distinction that is the whole point
    assert FailureTaxonomy.KERNEL_STALL.value == "KERNEL_STALL"
    assert FailureTaxonomy.ACCIDENTAL_SPILL.value == "ACCIDENTAL_SPILL"
    assert FitClass.ACCIDENTAL_WDDM_SPILL.value == "ACCIDENTAL_WDDM_SPILL"
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
        protocol_version="1.0.0",
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


# ---- schema export -----------------------------------------------------------


def test_export_json_schemas_writes_language_neutral_files(tmp_path):
    written = P.export_json_schemas(tmp_path)
    # 25 contract schemas + index.json
    assert len(written) == 26
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["contract_version"] == "1.0.0"
    assert len(index["contracts"]) == 25
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
