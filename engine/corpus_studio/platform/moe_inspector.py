"""Pure, bounded static parsing for allowlisted MoE configuration families.

The parser receives an already verified config mapping and its content digest. It performs no I/O,
imports no model framework, and emits structural expert-instance evidence only. It never claims
parameter-coordinate activity, residency, loadability, backend support, or runtime capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .contracts import (
    ExpertGroup,
    ExpertTopologyCounts,
    ModelTopology,
    SemanticRouting,
    TopologyInspection,
)
from .enums import ModelExecutionKind

InspectionStatus = Literal[
    "not_checked",
    "no_recognized_moe_evidence",
    "detected",
    "incomplete",
    "unsupported_family",
]

_MAX_LAYERS = 100_000
_MAX_EXPERTS_PER_LAYER = 1_000_000
_MISSING = object()
_SUPPORTED_FAMILIES = frozenset({"deepseek_v2", "deepseek_v3", "mixtral", "qwen2_moe"})
_MOE_SIGNAL_KEYS = frozenset(
    {
        "decoder_sparse_step",
        "first_k_dense_replace",
        "mlp_only_layers",
        "moe_layer_freq",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts",
        "num_experts_per_tok",
        "num_local_experts",
    }
)
_STATIC_ONLY_WARNING = (
    "Static MoE metadata does not prove model loading, inference, training, backend support, "
    "hardware fit, residency, or offload capability."
)


def _lookup(config: Mapping[str, Any], path: str) -> object:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


@dataclass
class _ConfigReader:
    config: Mapping[str, Any]
    evidence_paths: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def _read(self, path: str) -> object:
        value = _lookup(self.config, path)
        if value is _MISSING:
            self.errors.append(f"missing config.{path}")
        else:
            self.evidence_paths.add(path)
        return value

    def positive_int(self, path: str, *, maximum: int) -> int | None:
        value = self._read(path)
        if value is _MISSING:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > maximum:
            self.errors.append(f"config.{path} must be an integer from 1 to {maximum}")
            return None
        return value

    def optional_positive_int(
        self,
        path: str,
        *,
        maximum: int,
        default: int,
        missing_warning: str,
    ) -> int | None:
        value = _lookup(self.config, path)
        if value is _MISSING:
            self.warnings.append(missing_warning)
            return default
        self.evidence_paths.add(path)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > maximum:
            self.errors.append(f"config.{path} must be an integer from 1 to {maximum}")
            return None
        return value

    def nonnegative_int(self, path: str, *, maximum: int) -> int | None:
        value = self._read(path)
        if value is _MISSING:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
            self.errors.append(f"config.{path} must be an integer from 0 to {maximum}")
            return None
        return value

    def layer_list(self, path: str, *, layer_count: int | None) -> list[int] | None:
        value = self._read(path)
        if value is _MISSING:
            return None
        if not isinstance(value, list):
            self.errors.append(f"config.{path} must be a list of layer indices")
            return None
        if layer_count is None:
            return None
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0 or item >= layer_count
            for item in value
        ):
            self.errors.append(f"config.{path} must contain integers from 0 to {layer_count - 1}")
            return None
        if len(value) != len(set(value)):
            self.errors.append(f"config.{path} must not contain duplicate layer indices")
            return None
        return sorted(value)

    def optional_layer_list(
        self,
        path: str,
        *,
        layer_count: int | None,
        missing_warning: str,
    ) -> list[int] | None:
        value = _lookup(self.config, path)
        if value is _MISSING:
            self.warnings.append(missing_warning)
            return []
        self.evidence_paths.add(path)
        if value is None:
            self.warnings.append(
                f"config.{path} is null; parser applies the documented empty-list default"
            )
            return []
        return self.layer_list(path, layer_count=layer_count)


@dataclass(frozen=True)
class _TopologySpec:
    layers: tuple[int, ...]
    routed_experts: int
    shared_experts: int
    experts_per_token: int
    component_path: str
    router_type: str
    selection_policy: str
    routing_noise: str | None
    details: dict[str, Any]


def _declared_details(reader: _ConfigReader, paths: tuple[str, ...]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for path in paths:
        value = _lookup(reader.config, path)
        if value is not _MISSING:
            reader.evidence_paths.add(path)
            details[path] = value
    return details


def _routing_noise(config: Mapping[str, Any], *paths: str) -> str | None:
    for path in paths:
        value = _lookup(config, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
            return f"config.{path}={value}"
    return None


def _parse_mixtral(reader: _ConfigReader) -> _TopologySpec | None:
    layer_count = reader.positive_int("num_hidden_layers", maximum=_MAX_LAYERS)
    routed = reader.positive_int("num_local_experts", maximum=_MAX_EXPERTS_PER_LAYER)
    top_k = reader.positive_int("num_experts_per_tok", maximum=_MAX_EXPERTS_PER_LAYER)
    if routed is not None and top_k is not None and top_k > routed:
        reader.errors.append("config.num_experts_per_tok cannot exceed num_local_experts")
    if reader.errors or layer_count is None or routed is None or top_k is None:
        return None
    return _TopologySpec(
        layers=tuple(range(layer_count)),
        routed_experts=routed,
        shared_experts=0,
        experts_per_token=top_k,
        component_path="model.layers[].mlp",
        router_type="learned_top_k",
        selection_policy="learned_logits",
        routing_noise=_routing_noise(reader.config, "router_jitter_noise"),
        details=_declared_details(
            reader,
            ("norm_topk_prob", "router_aux_loss_coef", "router_jitter_noise"),
        ),
    )


def _parse_qwen2_moe(reader: _ConfigReader) -> _TopologySpec | None:
    layer_count = reader.positive_int("num_hidden_layers", maximum=_MAX_LAYERS)
    routed = reader.positive_int("num_experts", maximum=_MAX_EXPERTS_PER_LAYER)
    top_k = reader.positive_int("num_experts_per_tok", maximum=_MAX_EXPERTS_PER_LAYER)
    sparse_step = reader.positive_int("decoder_sparse_step", maximum=_MAX_LAYERS)
    dense_layers = reader.optional_layer_list(
        "mlp_only_layers",
        layer_count=layer_count,
        missing_warning=(
            "config.mlp_only_layers is absent; parser applies the documented qwen2_moe "
            "empty-list default"
        ),
    )
    shared_size = reader.positive_int("shared_expert_intermediate_size", maximum=10_000_000_000)
    if routed is not None and top_k is not None and top_k > routed:
        reader.errors.append("config.num_experts_per_tok cannot exceed num_experts")
    if (
        reader.errors
        or None
        in {
            layer_count,
            routed,
            top_k,
            sparse_step,
            shared_size,
        }
        or dense_layers is None
    ):
        return None
    assert layer_count is not None
    assert routed is not None
    assert top_k is not None
    assert sparse_step is not None
    dense_layer_set = set(dense_layers)
    layers = tuple(
        layer
        for layer in range(layer_count)
        if layer not in dense_layer_set and (layer + 1) % sparse_step == 0
    )
    if not layers:
        reader.errors.append("Qwen2MoE config resolves to zero sparse expert layers")
        return None
    return _TopologySpec(
        layers=layers,
        routed_experts=routed,
        shared_experts=1,
        experts_per_token=top_k,
        component_path="model.layers[].mlp",
        router_type="learned_top_k_with_shared_expert",
        selection_policy="learned_logits",
        routing_noise=None,
        details=_declared_details(
            reader,
            (
                "decoder_sparse_step",
                "mlp_only_layers",
                "norm_topk_prob",
                "router_aux_loss_coef",
                "shared_expert_intermediate_size",
            ),
        ),
    )


def _parse_deepseek(reader: _ConfigReader) -> _TopologySpec | None:
    layer_count = reader.positive_int("num_hidden_layers", maximum=_MAX_LAYERS)
    dense_prefix = reader.nonnegative_int("first_k_dense_replace", maximum=_MAX_LAYERS)
    routed = reader.positive_int("n_routed_experts", maximum=_MAX_EXPERTS_PER_LAYER)
    shared = reader.nonnegative_int("n_shared_experts", maximum=_MAX_EXPERTS_PER_LAYER)
    top_k = reader.positive_int("num_experts_per_tok", maximum=_MAX_EXPERTS_PER_LAYER)
    layer_frequency = reader.optional_positive_int(
        "moe_layer_freq",
        maximum=_MAX_LAYERS,
        default=1,
        missing_warning=(
            "config.moe_layer_freq is absent; parser applies compatibility frequency 1"
        ),
    )
    if layer_count is not None and dense_prefix is not None and dense_prefix >= layer_count:
        reader.errors.append("config.first_k_dense_replace must be below num_hidden_layers")
    if routed is not None and top_k is not None and top_k > routed:
        reader.errors.append("config.num_experts_per_tok cannot exceed n_routed_experts")
    if reader.errors or None in {
        layer_count,
        dense_prefix,
        routed,
        shared,
        top_k,
        layer_frequency,
    }:
        return None
    assert layer_count is not None
    assert dense_prefix is not None
    assert routed is not None
    assert shared is not None
    assert top_k is not None
    assert layer_frequency is not None
    selection = _lookup(reader.config, "topk_method")
    layers = tuple(
        layer
        for layer in range(dense_prefix, layer_count)
        if layer % layer_frequency == 0
    )
    if not layers:
        reader.errors.append("DeepSeek config resolves to zero sparse expert layers")
        return None
    return _TopologySpec(
        layers=layers,
        routed_experts=routed,
        shared_experts=shared,
        experts_per_token=top_k,
        component_path="model.layers[].mlp",
        router_type="learned_top_k_with_shared_experts",
        selection_policy=(
            str(selection).strip()
            if isinstance(selection, str) and selection.strip()
            else "learned_logits"
        ),
        routing_noise=None,
        details=_declared_details(
            reader,
            (
                "first_k_dense_replace",
                "moe_layer_freq",
                "n_group",
                "norm_topk_prob",
                "routed_scaling_factor",
                "topk_group",
                "topk_method",
            ),
        ),
    )


def _static_inspection(
    *,
    status: InspectionStatus,
    family: str | None,
    config_sha256: str,
    evidence_paths: set[str] | list[str],
    warnings: list[str],
) -> TopologyInspection:
    return TopologyInspection(
        status=status,
        method="static_config_v1",
        family=family,
        config_file="config.json",
        config_sha256=config_sha256,
        evidence_paths=sorted(set(evidence_paths)),
        warnings=sorted(set(warnings)),
        evidence_level="static_metadata_only",
    )


def _signal_paths(config: Mapping[str, Any]) -> set[str]:
    paths = {key for key in _MOE_SIGNAL_KEYS if _lookup(config, key) is not _MISSING}
    if _lookup(config, "model_type") is not _MISSING:
        paths.add("model_type")
    return paths


def _build_detected_topology(
    *,
    family: str,
    spec: _TopologySpec,
    reader: _ConfigReader,
    config_sha256: str,
) -> ModelTopology:
    layer_count = len(spec.layers)
    routed_instances = layer_count * spec.routed_experts
    shared_instances = layer_count * spec.shared_experts
    active_routed = layer_count * spec.experts_per_token
    metadata_sources = sorted(f"config.{path}" for path in reader.evidence_paths)
    return ModelTopology(
        execution_kind=ModelExecutionKind.mixture_of_experts,
        semantic_routing=SemanticRouting(
            router_type=spec.router_type,
            selection_policy=spec.selection_policy,
            top_k=spec.experts_per_token,
            routing_noise=spec.routing_noise,
            metadata_source="config.json",
            details=spec.details,
        ),
        expert_groups=[
            ExpertGroup(
                group_id="decoder-moe",
                layer_namespace="decoder",
                component_path=spec.component_path,
                layer_indices=list(spec.layers),
                expert_count=spec.routed_experts + spec.shared_experts,
                routed_expert_count=spec.routed_experts,
                experts_per_token=spec.experts_per_token,
                shared_expert_count=spec.shared_experts,
                expert_identity_scheme=(
                    "decoder/layer/{layer_index}/expert/{expert_kind}/{expert_index}"
                ),
                metadata_sources=metadata_sources,
            )
        ],
        expert_counts=ExpertTopologyCounts(
            moe_layer_count=layer_count,
            routed_expert_instances=routed_instances,
            shared_expert_instances=shared_instances,
            logical_expert_instances=routed_instances + shared_instances,
            active_routed_expert_instances_per_token=active_routed,
            active_shared_expert_instances_per_token=shared_instances,
            active_expert_instances_per_token=active_routed + shared_instances,
        ),
        inspection=_static_inspection(
            status="detected",
            family=family,
            config_sha256=config_sha256,
            evidence_paths=reader.evidence_paths,
            warnings=[_STATIC_ONLY_WARNING, *reader.warnings],
        ),
    )


def inspect_moe_topology(
    config: Mapping[str, Any],
    *,
    config_sha256: str | None,
) -> ModelTopology:
    """Return conservative static topology evidence for one verified ``config.json`` mapping."""

    if config_sha256 is None:
        return ModelTopology()
    raw_family = config.get("model_type")
    family = raw_family if isinstance(raw_family, str) and raw_family else None
    if family not in _SUPPORTED_FAMILIES:
        signal_paths = _signal_paths(config)
        normalized_family = family.lower() if family is not None else ""
        moe_named = "moe" in normalized_family or "expert" in normalized_family
        status: InspectionStatus = (
            "unsupported_family"
            if signal_paths - {"model_type"} or moe_named
            else "no_recognized_moe_evidence"
        )
        warnings = (
            [
                "MoE-like metadata was present for an unsupported family; execution_kind remains unknown."
            ]
            if status == "unsupported_family"
            else []
        )
        return ModelTopology(
            inspection=_static_inspection(
                status=status,
                family=family or ("unknown" if status == "unsupported_family" else None),
                config_sha256=config_sha256,
                evidence_paths=signal_paths,
                warnings=warnings,
            )
        )

    reader = _ConfigReader(config=config, evidence_paths={"model_type"})
    parser = (
        _parse_mixtral
        if family == "mixtral"
        else _parse_qwen2_moe
        if family == "qwen2_moe"
        else _parse_deepseek
    )
    spec = parser(reader)
    if spec is None:
        return ModelTopology(
            inspection=_static_inspection(
                status="incomplete",
                family=family,
                config_sha256=config_sha256,
                evidence_paths=reader.evidence_paths,
                warnings=reader.errors,
            )
        )
    return _build_detected_topology(
        family=family,
        spec=spec,
        reader=reader,
        config_sha256=config_sha256,
    )


__all__ = ["inspect_moe_topology"]
