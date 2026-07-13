"""Multi-backend registry — the 'pick your framework' substrate. Pure tests (no torch): every backend
declares its surface honestly, and selection resolves a plan's requirements against it. The load-
bearing honesty case: Unsloth (flash/sdpa only) is filtered OUT of a Blackwell math plan."""

import corpus_studio.platform as P
from corpus_studio.platform.contracts import EffectiveCapabilities
from corpus_studio.platform.backends import (
    builtin_backends,
    compatible_backends,
    get_backend,
    unmet_physical_requirements,
    unmet_requirements,
)
from corpus_studio.platform.enums import OffloadStrategy

_BLACKWELL = dict(
    os="linux", device="cuda", task_type="sft", precision="bf16", quantization="nf4",
    adapter_method="qlora", attention="math",
)
_AMPERE = {**_BLACKWELL, "attention": "sdpa"}


def test_builtin_backends_roundtrip():
    ids = [b.backend_id for b in builtin_backends()]
    assert ids == ["corpus_studio", "unsloth"]
    for backend in builtin_backends():
        assert P.BackendManifest.model_validate_json(backend.model_dump_json()) == backend
        assert [item.value for item in backend.placement_modes] == ["single_resource"]
        assert [item.value for item in backend.placement_tiers] == ["gpu"]
        assert backend.offload_strategies == []
        assert backend.parallelism_kinds == []


def test_get_backend():
    assert get_backend("corpus_studio").backend_id == "corpus_studio"
    assert get_backend("unsloth").display_name.startswith("Unsloth")
    assert get_backend("megatron") is None


def test_corpus_studio_runs_a_blackwell_math_plan():
    assert unmet_requirements(get_backend("corpus_studio"), **_BLACKWELL) == []


def test_unsloth_refuses_the_math_attention_a_blackwell_plan_needs():
    reasons = unmet_requirements(get_backend("unsloth"), **_BLACKWELL)
    assert any("attention 'math'" in r for r in reasons)


def test_unsloth_has_no_cpu_path():
    reasons = unmet_requirements(get_backend("unsloth"), **{**_AMPERE, "device": "cpu"})
    assert any("device 'cpu'" in r for r in reasons)


def test_unmet_reports_every_unsupported_field():
    reasons = unmet_requirements(
        get_backend("unsloth"),
        os="macos", device="cpu", task_type="reward", precision="fp8",
        quantization="gptq", adapter_method="dora", attention="xformers",
    )
    # OS, device, task, precision, quant, adapter, attention — all seven are unsupported.
    assert len(reasons) == 7


def test_compatible_backends_for_a_math_plan_is_corpus_only():
    ids = [b.backend_id for b in compatible_backends(**_BLACKWELL)]
    assert ids == ["corpus_studio"]  # Unsloth can't do math → routed away from Blackwell


def test_compatible_backends_for_an_sdpa_plan_includes_unsloth():
    ids = [b.backend_id for b in compatible_backends(**_AMPERE)]
    assert set(ids) == {"corpus_studio", "unsloth"}


def test_no_backend_fits_an_impossible_configuration():
    impossible = {**_AMPERE, "adapter_method": "full_finetune"}
    assert compatible_backends(**impossible) == []


def test_physical_capabilities_require_static_declaration_and_functional_proof():
    spec = P.PhysicalExecutionSpec(
        resources=[
            P.PhysicalResource(
                resource_id="compute-0", tier="gpu", device_kind="cuda", device_id="cuda:0"
            )
        ],
        placements=[
            P.StatePlacement(
                placement_id="parameters-authoritative",
                state="parameters",
                selector={"whole_model": True},
                resource_id="compute-0",
                role="authoritative",
            )
        ],
        parallelism=P.ParallelismSpec(
            world_size=1,
            ranks=[P.RankBinding(rank=0, resource_id="compute-0")],
        ),
    )
    backend = get_backend("corpus_studio")
    unverified = unmet_physical_requirements(
        backend,
        EffectiveCapabilities(),
        spec,
        offload_strategy=OffloadStrategy.none,
    )
    assert "placement tier 'gpu' is not functionally verified" in unverified
    assert "placement mode 'single_resource' is not functionally verified" in unverified

    proven = EffectiveCapabilities(
        placement_tiers=["gpu"], placement_modes=["single_resource"]
    )
    assert unmet_physical_requirements(
        backend, proven, spec, offload_strategy=OffloadStrategy.none
    ) == []


def test_physical_capability_gate_detects_replication_sharding_and_expert_scope():
    spec = P.PhysicalExecutionSpec(
        resources=[
            P.PhysicalResource(
                resource_id=f"gpu-{index}",
                tier="gpu",
                device_kind="cuda",
                device_id=f"cuda:{index}",
            )
            for index in range(2)
        ],
        placements=[
            P.StatePlacement(
                placement_id="expert-optimizer",
                state="optimizer_state",
                selector={"expert_ids": ["expert.0"]},
                resource_id="gpu-0",
                role="authoritative",
            ),
            *[
                P.StatePlacement(
                    placement_id=f"gradient-shard-{index}",
                    state="gradients",
                    selector={"whole_model": True},
                    resource_id=f"gpu-{index}",
                    role="shard",
                    shard_group_id="gradient-shards",
                    shard_index=index,
                    shard_count=2,
                )
                for index in range(2)
            ],
            P.StatePlacement(
                placement_id="parameters-authoritative",
                state="parameters",
                selector={"whole_model": True},
                resource_id="gpu-0",
                role="authoritative",
            ),
            P.StatePlacement(
                placement_id="parameters-replica",
                state="parameters",
                selector={"whole_model": True},
                resource_id="gpu-1",
                role="replica",
                source_placement_id="parameters-authoritative",
            ),
        ],
        parallelism=P.ParallelismSpec(
            world_size=2,
            ranks=[
                P.RankBinding(rank=index, resource_id=f"gpu-{index}", local_rank=index)
                for index in range(2)
            ],
        ),
    )
    reasons = unmet_physical_requirements(
        get_backend("corpus_studio"),
        EffectiveCapabilities(placement_tiers=["gpu"]),
        spec,
        offload_strategy=OffloadStrategy.none,
    )
    assert any("placement mode 'identity_scoped'" in reason for reason in reasons)
    assert any("placement mode 'replicated'" in reason for reason in reasons)
    assert any("placement mode 'sharded'" in reason for reason in reasons)
    assert any("placement mode 'expert_scoped'" in reason for reason in reasons)
