"""Static MoE topology parsing without model loading or runtime claims."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from corpus_studio.platform.contracts import ModelDescriptor, ModelTopology
from corpus_studio.platform.enums import ParameterCountKind
from corpus_studio.platform.moe_inspector import inspect_moe_topology
from corpus_studio.platform.parameter_accounting import build_model_parameter_accounting


CONFIG_SHA256 = "a" * 64


def test_mixtral_static_topology_counts_are_expert_instances_not_parameters():
    topology = inspect_moe_topology(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 4,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "router_jitter_noise": 0.01,
        },
        config_sha256=CONFIG_SHA256,
    )

    assert topology.execution_kind.value == "mixture_of_experts"
    assert topology.inspection.status == "detected"
    assert topology.inspection.evidence_level == "static_metadata_only"
    assert topology.inspection.runtime_capability == "unverified"
    assert topology.inspection.config_sha256 == CONFIG_SHA256
    assert topology.inspection.evidence_paths == [
        "model_type",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_local_experts",
        "router_jitter_noise",
    ]
    group = topology.expert_groups[0]
    assert group.layer_indices == [0, 1, 2, 3]
    assert group.expert_count == 8
    assert group.routed_expert_count == 8
    assert group.shared_expert_count == 0
    assert group.experts_per_token == 2
    assert group.component_path == "model.layers[].mlp"
    assert topology.semantic_routing is not None
    assert topology.semantic_routing.routing_noise == "config.router_jitter_noise=0.01"
    counts = topology.expert_counts
    assert counts is not None
    assert counts.unit == "expert_instances"
    assert counts.moe_layer_count == 4
    assert counts.logical_expert_instances == 32
    assert counts.routed_expert_instances == 32
    assert counts.active_expert_instances_per_token == 8


def test_qwen2_sparse_step_excludes_mlp_only_layers_and_counts_shared_expert():
    topology = inspect_moe_topology(
        {
            "model_type": "qwen2_moe",
            "num_hidden_layers": 6,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "decoder_sparse_step": 2,
            "mlp_only_layers": [3],
            "shared_expert_intermediate_size": 5632,
            "norm_topk_prob": False,
        },
        config_sha256=CONFIG_SHA256,
    )

    group = topology.expert_groups[0]
    assert group.layer_indices == [1, 5]
    assert group.expert_count == 61
    assert group.routed_expert_count == 60
    assert group.shared_expert_count == 1
    counts = topology.expert_counts
    assert counts is not None
    assert counts.logical_expert_instances == 122
    assert counts.routed_expert_instances == 120
    assert counts.shared_expert_instances == 2
    assert counts.active_routed_expert_instances_per_token == 8
    assert counts.active_shared_expert_instances_per_token == 2
    assert counts.active_expert_instances_per_token == 10


def test_canonical_qwen2_shape_uses_explicit_legacy_empty_layer_default():
    topology = inspect_moe_topology(
        {
            "model_type": "qwen2_moe",
            "num_hidden_layers": 24,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 5632,
        },
        config_sha256=CONFIG_SHA256,
    )

    assert topology.inspection.status == "detected"
    assert topology.expert_groups[0].layer_indices == list(range(24))
    assert "mlp_only_layers" not in topology.inspection.evidence_paths
    assert any("empty-list default" in warning for warning in topology.inspection.warnings)


@pytest.mark.parametrize("family", ["deepseek_v2", "deepseek_v3"])
def test_deepseek_dense_prefix_and_router_group_metadata(family: str):
    topology = inspect_moe_topology(
        {
            "model_type": family,
            "num_hidden_layers": 5,
            "first_k_dense_replace": 2,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "moe_layer_freq": 1,
            "n_group": 8,
            "topk_group": 4,
            "topk_method": "group_limited_greedy",
        },
        config_sha256=CONFIG_SHA256,
    )

    assert topology.expert_groups[0].layer_indices == [2, 3, 4]
    assert topology.semantic_routing is not None
    assert topology.semantic_routing.selection_policy == "group_limited_greedy"
    assert topology.semantic_routing.details["n_group"] == 8
    assert topology.semantic_routing.details["topk_group"] == 4
    assert "topk_method" in topology.inspection.evidence_paths
    counts = topology.expert_counts
    assert counts is not None
    assert counts.logical_expert_instances == 198
    assert counts.active_expert_instances_per_token == 24


@pytest.mark.parametrize("family", ["deepseek_v2", "deepseek_v3"])
def test_deepseek_layer_frequency_is_evidenced_and_applied(family: str):
    topology = inspect_moe_topology(
        {
            "model_type": family,
            "num_hidden_layers": 8,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 2,
            "n_routed_experts": 64,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
        },
        config_sha256=CONFIG_SHA256,
    )

    assert topology.inspection.status == "detected"
    assert topology.expert_groups[0].layer_indices == [2, 4, 6]
    assert "moe_layer_freq" in topology.inspection.evidence_paths
    counts = topology.expert_counts
    assert counts is not None
    assert counts.moe_layer_count == 3
    assert counts.logical_expert_instances == 195
    assert counts.active_expert_instances_per_token == 21


def test_deepseek_missing_layer_frequency_uses_visible_compatibility_default():
    topology = inspect_moe_topology(
        {
            "model_type": "deepseek_v3",
            "num_hidden_layers": 5,
            "first_k_dense_replace": 2,
            "n_routed_experts": 64,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
        },
        config_sha256=CONFIG_SHA256,
    )

    assert topology.inspection.status == "detected"
    assert topology.expert_groups[0].layer_indices == [2, 3, 4]
    assert "moe_layer_freq" not in topology.inspection.evidence_paths
    assert any("compatibility frequency 1" in warning for warning in topology.inspection.warnings)


@pytest.mark.parametrize(
    "override, warning",
    [
        ({"num_hidden_layers": True}, "num_hidden_layers"),
        ({"num_hidden_layers": 100_001}, "num_hidden_layers"),
        ({"num_experts_per_tok": 9}, "cannot exceed"),
        ({"num_local_experts": 0}, "num_local_experts"),
    ],
)
def test_malformed_allowlisted_metadata_stays_unknown(override: dict[str, object], warning: str):
    config: dict[str, object] = {
        "model_type": "mixtral",
        "num_hidden_layers": 4,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    config.update(override)
    topology = inspect_moe_topology(config, config_sha256=CONFIG_SHA256)
    assert topology.execution_kind.value == "unknown"
    assert topology.inspection.status == "incomplete"
    assert not topology.expert_groups
    assert warning in " ".join(topology.inspection.warnings)


@pytest.mark.parametrize("family", ["future_sparse_router", "Mixtral", " mixtral"])
def test_unknown_family_with_moe_like_keys_is_not_guessed(family: str):
    topology = inspect_moe_topology(
        {
            "model_type": family,
            "num_hidden_layers": 4,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
        config_sha256=CONFIG_SHA256,
    )
    assert topology.execution_kind.value == "unknown"
    assert topology.inspection.status == "unsupported_family"
    assert topology.inspection.family == family
    assert not topology.expert_groups


def test_dense_metadata_absence_is_not_promoted_to_dense_proof():
    topology = inspect_moe_topology(
        {"model_type": "llama", "num_hidden_layers": 4},
        config_sha256=CONFIG_SHA256,
    )
    assert topology.execution_kind.value == "unknown"
    assert topology.inspection.status == "no_recognized_moe_evidence"
    assert topology.inspection.runtime_capability == "unverified"


def test_missing_verified_config_digest_leaves_inspection_not_checked():
    topology = inspect_moe_topology(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 4,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
        config_sha256=None,
    )
    assert topology.execution_kind.value == "unknown"
    assert topology.inspection.status == "not_checked"
    assert topology.inspection.method == "not_checked"
    assert not topology.expert_groups


def test_contract_rejects_tampered_expert_totals_and_representation_conflicts():
    topology = inspect_moe_topology(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
        config_sha256=CONFIG_SHA256,
    )
    payload = topology.model_dump(mode="json")
    payload["expert_counts"]["logical_expert_instances"] = 15
    with pytest.raises(ValidationError, match="logical expert instances"):
        ModelTopology.model_validate(payload)

    with pytest.raises(ValidationError, match="representation kind"):
        ModelDescriptor(
            model_id="m",
            source={"kind": "local", "local_path": "C:/models/m"},
            parameters={"kind": "dense"},
            topology=topology,
        )


def test_legacy_expert_group_payload_is_accepted_and_normalized():
    model = ModelDescriptor.model_validate(
        {
            "contract_version": "1.0.0",
            "model_id": "legacy-moe",
            "source": {"kind": "local", "local_path": "C:/models/legacy-moe"},
            "parameters": {"kind": "mixture_of_experts"},
            "topology": {
                "execution_kind": "mixture_of_experts",
                "semantic_routing": {
                    "router_type": "top-k",
                    "selection_policy": "learned_logits",
                    "top_k": 2,
                    "metadata_source": "legacy-config",
                },
                "expert_groups": [
                    {
                        "group_id": "decoder",
                        "layer_indices": [0, 1],
                        "expert_count": 8,
                        "experts_per_token": 2,
                        "shared_expert_count": None,
                    }
                ],
            },
        }
    )

    group = model.topology.expert_groups[0]
    assert group.routed_expert_count == 8
    assert group.shared_expert_count == 0
    payload = model.model_dump(mode="json")
    assert payload["topology"]["expert_groups"][0]["routed_expert_count"] == 8
    assert payload["topology"]["expert_groups"][0]["shared_expert_count"] == 0


def test_detected_hybrid_expert_topology_is_contract_valid():
    topology = inspect_moe_topology(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
        config_sha256=CONFIG_SHA256,
    )
    payload = topology.model_dump(mode="json")
    payload["execution_kind"] = "hybrid"

    validated = ModelTopology.model_validate(payload)
    assert validated.execution_kind.value == "hybrid"


def test_structural_top_k_never_becomes_active_or_resident_parameter_evidence():
    topology = inspect_moe_topology(
        {
            "model_type": "mixtral",
            "num_hidden_layers": 2,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "num_parameters": 1_000,
        },
        config_sha256=CONFIG_SHA256,
    )
    model = ModelDescriptor(
        model_id="m",
        source={"kind": "local", "local_path": "C:/models/m"},
        parameters={
            "kind": "mixture_of_experts",
            "counts": [
                {
                    "kind": "logical",
                    "value": 1_000,
                    "scope": "model",
                    "measurement_window": "static_model",
                    "source": "config.num_parameters",
                    "evidence": "declared",
                }
            ],
        },
        topology=topology,
    )
    report = build_model_parameter_accounting(model)
    kinds = {item.kind for item in report.observations}
    assert ParameterCountKind.active_token not in kinds
    assert ParameterCountKind.resident not in kinds
    assert any("does not infer active" in note for note in report.notes)
