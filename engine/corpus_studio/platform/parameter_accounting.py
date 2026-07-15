"""Dependency-light parameter-accounting evidence production and reconciliation.

This module never loads model weights, imports a training framework, or converts allocator bytes into
parameter coordinates. Static declarations, bounded safetensors-header evidence, and worker-origin
runtime observations remain separately labeled until an exact, comparable reconciliation is possible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import tempfile
from typing import Any, cast

from corpus_studio import __version__

from .common import HashRef, Ref
from .contracts import (
    DescriptorFile,
    ModelDescriptor,
    ParameterAccountingReport,
    ParameterConflict,
    ParameterCount,
    ParameterCountHandling,
    ParameterEvidenceGap,
    ParameterEvidenceSource,
    ParameterObservation,
    ParameterScope,
    ParameterWindow,
    RunEvent,
    required_parameter_kinds,
)
from .enums import (
    CountHandling,
    DescriptorFileRole,
    EvidenceKind,
    ModelFormat,
    ParameterAccountingProfile,
    ParameterAccountingStatus,
    ParameterCountKind,
    ParameterEvidenceSourceKind,
    ParameterGapReason,
    ParameterIdentityBasis,
    ParameterObservationCoverage,
    ParameterScopeKind,
    ParameterValueRelation,
    ParameterWindowKind,
)

_ZERO_HASH = "0" * 64
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_MAX_SAFETENSORS_TENSORS = 1_000_000
_MAX_TENSOR_DIMENSIONS = 32
_MAX_PARAMETER_COORDINATES = (1 << 63) - 1
_MAX_EVENT_LINE_BYTES = 8 * 1024 * 1024
_MAX_EVENTS = 1_000_000

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class ParameterAccountingError(RuntimeError):
    """Bounded, user-facing parameter-accounting failure."""


class _HeaderEvidenceError(RuntimeError):
    def __init__(self, reason: ParameterGapReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class _SafetensorsFileEvidence:
    path: str
    coordinates: int
    tensors: tuple[dict[str, object], ...]
    integrity_verified: bool
    content_sha256: str | None = None


@dataclass(frozen=True)
class SafetensorsTensorStateEvidence:
    content_sha256: str
    tensor_state_sha256: str
    tensor_names: tuple[str, ...]


def canonical_tensor_state_sha256(records: Sequence[Mapping[str, object]]) -> str:
    """Canonical identity shared by in-memory adapter export and saved Safetensors bytes."""

    normalized: list[dict[str, object]] = []
    for record in records:
        raw_shape = cast(Sequence[int], record["shape"])
        normalized.append(
            {
                "name": str(record["name"]),
                "dtype": str(record["dtype"]),
                "shape": list(raw_shape),
                "content_sha256": str(record["content_sha256"]),
            }
        )
    normalized.sort(key=lambda item: str(item["name"]))
    return hashlib.sha256(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(now: Callable[[], datetime]) -> str:
    return now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_id(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() or character in "._-" else "-"
        for character in value
    ).strip("-.")
    normalized = normalized or "item"
    if normalized != value.lower():
        suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        normalized = f"{normalized}-{suffix}"
    return normalized


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parameter_accounting_hash_for(report: ParameterAccountingReport) -> str:
    """Canonical SHA-256 seal over a report, excluding the seal itself."""

    payload = report.model_dump(mode="json", exclude={"report_hash"})
    return _canonical_sha256(payload)


def verify_parameter_accounting_hash(report: ParameterAccountingReport) -> bool:
    return parameter_accounting_hash_for(report) == report.report_hash


def _revalidate_report(
    report: ParameterAccountingReport,
    *,
    label: str,
) -> ParameterAccountingReport:
    try:
        return ParameterAccountingReport.model_validate(report.model_dump(mode="json"))
    except (ValueError, TypeError, RecursionError) as exc:
        raise ParameterAccountingError(f"{label} is structurally invalid: {exc}") from exc


def _model_ref(model: ModelDescriptor) -> Ref:
    if model.source.snapshot_sha256 is not None:
        return Ref(
            id=model.model_id,
            hash=HashRef(algo="sha256", value=model.source.snapshot_sha256),
        )
    if model.inventory_sha256 is not None:
        return Ref(
            id=model.model_id,
            hash=HashRef(algo="sha256-ordered-exact-v1", value=model.inventory_sha256),
        )
    return Ref(id=model.model_id)


def _descriptor_ref(model: ModelDescriptor) -> Ref:
    return Ref(
        id=_stable_id(f"{model.model_id}-descriptor"),
        hash=HashRef(
            algo="sha256",
            value=_canonical_sha256(model.model_dump(mode="json")),
        ),
    )


def _coordinate_hash(model: ModelDescriptor) -> str | None:
    return model.source.snapshot_sha256


def _scope_for_count(
    model: ModelDescriptor,
    count: ParameterCount,
    *,
    coordinate_hash: str | None = None,
) -> ParameterScope:
    component_ids = {item.component_id for item in model.parameters.components}
    scope_id = _stable_id(count.scope)
    scope_kind = (
        ParameterScopeKind.model
        if count.scope == "model"
        else ParameterScopeKind.component_set
        if count.scope in component_ids
        else ParameterScopeKind.custom
    )
    return ParameterScope(
        scope_id=scope_id,
        kind=scope_kind,
        model_ref=_model_ref(model),
        coordinate_universe_id=_stable_id(f"{model.model_id}-coordinates"),
        coordinate_universe_sha256=coordinate_hash or _coordinate_hash(model),
        component_ids=[count.scope] if count.scope in component_ids else [],
        definition=f"Descriptor parameter scope '{count.scope}'.",
    )


def _model_scope(model_ref: Ref, *, coordinate_hash: str | None = None) -> ParameterScope:
    resolved_coordinate_hash = coordinate_hash
    if (
        resolved_coordinate_hash is None
        and model_ref.hash is not None
        and model_ref.hash.algo == "sha256"
    ):
        resolved_coordinate_hash = model_ref.hash.value
    return ParameterScope(
        scope_id="model",
        kind=ParameterScopeKind.model,
        model_ref=model_ref,
        coordinate_universe_id=_stable_id(f"{model_ref.id}-coordinates"),
        coordinate_universe_sha256=resolved_coordinate_hash,
        definition="All independently addressable coordinates for the referenced model revision.",
    )


def _static_window(window_id: str = "static-model") -> ParameterWindow:
    return ParameterWindow(
        window_id=_stable_id(window_id),
        kind=ParameterWindowKind.static_snapshot,
        definition="One immutable model snapshot.",
    )


def _gap_window(
    profile: ParameterAccountingProfile,
    *,
    plan_ref: Ref | None,
    run_ref: Ref | None,
) -> ParameterWindow:
    if run_ref is not None:
        return ParameterWindow(
            window_id="required-run-window",
            kind=ParameterWindowKind.run,
            definition=f"Required {profile.value} run window whose evidence is unavailable.",
            run_ref=run_ref,
        )
    return ParameterWindow(
        window_id="required-evidence-window",
        kind=ParameterWindowKind.static_snapshot,
        definition=(
            f"Required {profile.value} evidence window is unresolved"
            + (f" for plan {plan_ref.id}." if plan_ref is not None else ".")
        ),
        plan_ref=plan_ref,
    )


def _count_handling_complete(handling: ParameterCountHandling) -> bool:
    return CountHandling.unknown not in {
        handling.tied,
        handling.shared,
        handling.replicated,
        handling.generated,
        handling.quantized,
        handling.optimizer_shadows,
        handling.decompressed_caches,
    }


def _handling_complete(observation: ParameterObservation) -> bool:
    return _count_handling_complete(observation.handling)


def _qualifies(
    profile: ParameterAccountingProfile,
    observation: ParameterObservation,
) -> bool:
    if not _handling_complete(observation):
        return False
    if observation.identity_basis in {
        ParameterIdentityBasis.unknown,
        ParameterIdentityBasis.stored_tensor_elements,
    }:
        return False
    if profile in {
        ParameterAccountingProfile.training_runtime,
        ParameterAccountingProfile.inference_runtime,
        ParameterAccountingProfile.checkpoint,
        ParameterAccountingProfile.evaluation,
    } and observation.kind != ParameterCountKind.logical:
        return (
            observation.evidence == EvidenceKind.measured
            and observation.coverage == ParameterObservationCoverage.complete
            and observation.value_relation == ParameterValueRelation.exact
        )
    if profile == ParameterAccountingProfile.training_plan:
        return (
            observation.coverage == ParameterObservationCoverage.complete
            and observation.value_relation
            in {ParameterValueRelation.exact, ParameterValueRelation.estimate}
        )
    if profile == ParameterAccountingProfile.model_static:
        return (
            observation.evidence == EvidenceKind.measured
            and observation.coverage == ParameterObservationCoverage.complete
            and observation.value_relation == ParameterValueRelation.exact
        )
    return (
        observation.coverage == ParameterObservationCoverage.complete
        and observation.value_relation == ParameterValueRelation.exact
    )


def _scope_key(observation: ParameterObservation) -> tuple[str, str, str, str]:
    scope = observation.scope
    return (
        scope.model_ref.id,
        scope.coordinate_universe_id,
        scope.coordinate_universe_sha256 or "",
        scope.scope_id,
    )


def _derive_conflicts(observations: Sequence[ParameterObservation]) -> list[ParameterConflict]:
    conflicts: dict[str, ParameterConflict] = {}

    def add(code: str, left: ParameterObservation, right: ParameterObservation, message: str) -> None:
        ids = sorted({left.observation_id, right.observation_id})
        if len(ids) != 2:
            return
        conflict_id = _stable_id(f"{code}-{'-'.join(ids)}")
        conflicts[conflict_id] = ParameterConflict(
            conflict_id=conflict_id,
            observation_ids=ids,
            reason_code=code,
            explanation=message,
        )

    exact_groups: dict[tuple[object, ...], list[ParameterObservation]] = {}
    for observation in observations:
        if (
            observation.coverage == ParameterObservationCoverage.complete
            and observation.value_relation == ParameterValueRelation.exact
        ):
            key = (
                observation.kind.value,
                *_scope_key(observation),
                observation.window.window_id,
                observation.unit,
            )
            exact_groups.setdefault(key, []).append(observation)
    for group in exact_groups.values():
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if left.value != right.value:
                    add(
                        "same-key-value-mismatch",
                        left,
                        right,
                        "Complete exact observations for the same axis/scope/window disagree.",
                    )

    authoritative = [
        item
        for item in observations
        if item.coverage == ParameterObservationCoverage.complete
        and item.value_relation == ParameterValueRelation.exact
        and item.unit == "coordinates"
    ]
    logical_by_scope: dict[tuple[str, str, str, str], list[ParameterObservation]] = {}
    for observation in authoritative:
        if observation.kind == ParameterCountKind.logical:
            logical_by_scope.setdefault(_scope_key(observation), []).append(observation)
    bounded_kinds = {
        ParameterCountKind.active_token,
        ParameterCountKind.active_sequence,
        ParameterCountKind.touched_window,
        ParameterCountKind.resident,
        ParameterCountKind.updated_window,
        ParameterCountKind.exposed_window,
    }
    for observation in authoritative:
        if observation.kind not in bounded_kinds:
            continue
        for logical in logical_by_scope.get(_scope_key(observation), []):
            if observation.value > logical.value:
                add(
                    "count-exceeds-logical",
                    observation,
                    logical,
                    f"{observation.kind.value} cannot exceed exact N_logical in the same universe.",
                )
    for left in authoritative:
        for right in authoritative:
            if _scope_key(left) != _scope_key(right):
                continue
            if left.window.window_id != right.window.window_id:
                continue
            if (
                left.kind == ParameterCountKind.updated_window
                and right.kind == ParameterCountKind.touched_window
                and left.value > right.value
            ):
                add(
                    "updated-exceeds-touched",
                    left,
                    right,
                    "Updated coordinates cannot exceed touched coordinates in the same window.",
                )
    tokens = [item for item in authoritative if item.kind == ParameterCountKind.active_token]
    sequences = [
        item for item in authoritative if item.kind == ParameterCountKind.active_sequence
    ]
    for token in tokens:
        for sequence in sequences:
            if (
                _scope_key(token) == _scope_key(sequence)
                and token.window.sequence_id == sequence.window.sequence_id
                and token.value > sequence.value
            ):
                add(
                    "token-active-exceeds-sequence-active",
                    token,
                    sequence,
                    "N_active_token cannot exceed N_active_sequence for the same sequence.",
                )
    return [conflicts[key] for key in sorted(conflicts)]


def _missing_gap(
    *,
    kind: ParameterCountKind,
    profile: ParameterAccountingProfile,
    model_ref: Ref,
    candidates: Sequence[ParameterObservation],
    plan_ref: Ref | None,
    run_ref: Ref | None,
) -> ParameterEvidenceGap:
    scope = candidates[0].scope if candidates else _model_scope(model_ref)
    window = candidates[0].window if candidates else _gap_window(
        profile,
        plan_ref=plan_ref,
        run_ref=run_ref,
    )
    if not candidates:
        reason = ParameterGapReason.missing_observation
        explanation = f"No {kind.value} observation was supplied."
    elif any(not _handling_complete(item) for item in candidates):
        reason = ParameterGapReason.unknown_handling
        explanation = (
            f"{kind.value} evidence does not resolve tied/shared/replica/generated/quantized/state handling."
        )
    elif any(
        item.identity_basis == ParameterIdentityBasis.stored_tensor_elements
        for item in candidates
    ):
        reason = ParameterGapReason.stored_elements_not_logical
        explanation = "Stored tensor elements were observed but not proven to be logical coordinates."
    elif profile in {
        ParameterAccountingProfile.training_runtime,
        ParameterAccountingProfile.inference_runtime,
        ParameterAccountingProfile.checkpoint,
        ParameterAccountingProfile.evaluation,
    } and kind != ParameterCountKind.logical:
        reason = ParameterGapReason.measured_evidence_required
        explanation = f"{kind.value} lacks complete exact measured worker evidence."
    elif profile == ParameterAccountingProfile.model_static:
        reason = ParameterGapReason.measured_evidence_required
        explanation = (
            f"{kind.value} lacks complete exact measured evidence corroborating the declaration."
        )
    else:
        reason = ParameterGapReason.estimated_only
        explanation = f"{kind.value} evidence is sampled, partial, estimated, or otherwise non-exact."
    return ParameterEvidenceGap(
        gap_id=_stable_id(f"required-{kind.value}"),
        kind=kind,
        scope=scope,
        window=window,
        reason=reason,
        explanation=explanation,
        resolution=(
            "Provide a complete, explicitly scoped observation with resolved identity and handling."
        ),
    )


def _seal_report(
    *,
    report_id: str,
    generated_at: str,
    profile: ParameterAccountingProfile,
    model_ref: Ref,
    observations: Iterable[ParameterObservation],
    extra_gaps: Iterable[ParameterEvidenceGap] = (),
    plan_ref: Ref | None = None,
    run_ref: Ref | None = None,
    artifact_refs: Sequence[Ref] = (),
    evaluation_refs: Sequence[Ref] = (),
    parent_report_refs: Sequence[Ref] = (),
    notes: Sequence[str] = (),
) -> ParameterAccountingReport:
    observation_by_id: dict[str, ParameterObservation] = {}
    for observation in observations:
        previous = observation_by_id.get(observation.observation_id)
        if previous is not None and previous != observation:
            raise ParameterAccountingError(
                f"observation id maps to different evidence: {observation.observation_id}"
            )
        observation_by_id[observation.observation_id] = observation
    ordered_observations = [observation_by_id[key] for key in sorted(observation_by_id)]

    gap_by_id: dict[str, ParameterEvidenceGap] = {}
    for gap in extra_gaps:
        previous_gap = gap_by_id.get(gap.gap_id)
        if previous_gap is not None and previous_gap != gap:
            raise ParameterAccountingError(f"gap id maps to different evidence: {gap.gap_id}")
        gap_by_id[gap.gap_id] = gap
    for kind in sorted(required_parameter_kinds(profile), key=lambda item: item.value):
        candidates = [item for item in ordered_observations if item.kind == kind]
        if not any(_qualifies(profile, item) for item in candidates):
            gap = _missing_gap(
                kind=kind,
                profile=profile,
                model_ref=model_ref,
                candidates=candidates,
                plan_ref=plan_ref,
                run_ref=run_ref,
            )
            previous_gap = gap_by_id.get(gap.gap_id)
            if previous_gap is not None and previous_gap != gap:
                raise ParameterAccountingError(
                    f"gap id maps to different evidence: {gap.gap_id}"
                )
            gap_by_id[gap.gap_id] = gap
    ordered_gaps = [gap_by_id[key] for key in sorted(gap_by_id)]
    conflicts = _derive_conflicts(ordered_observations)
    status = (
        ParameterAccountingStatus.conflicting
        if conflicts
        else ParameterAccountingStatus.incomplete
        if ordered_gaps
        else ParameterAccountingStatus.complete
    )
    draft = ParameterAccountingReport(
        report_id=_stable_id(report_id),
        report_hash=_ZERO_HASH,
        generated_at=generated_at,
        profile=profile,
        status=status,
        model_ref=model_ref,
        plan_ref=plan_ref,
        run_ref=run_ref,
        artifact_refs=sorted(artifact_refs, key=ParameterAccountingReport._ref_key),
        evaluation_refs=sorted(evaluation_refs, key=ParameterAccountingReport._ref_key),
        parent_report_refs=sorted(parent_report_refs, key=ParameterAccountingReport._ref_key),
        observations=ordered_observations,
        gaps=ordered_gaps,
        conflicts=conflicts,
        notes=sorted(set(notes)),
    )
    sealed = draft.model_copy(update={"report_hash": parameter_accounting_hash_for(draft)})
    if not verify_parameter_accounting_hash(sealed):  # pragma: no cover - construction invariant
        raise ParameterAccountingError("failed to seal parameter accounting report")
    return sealed


def _descriptor_observations(
    model: ModelDescriptor,
    *,
    generated_at: str,
) -> tuple[list[ParameterObservation], list[ParameterEvidenceGap]]:
    observations: list[ParameterObservation] = []
    gaps: list[ParameterEvidenceGap] = []
    model_ref = _model_ref(model)
    descriptor_ref = _descriptor_ref(model)
    for index, count in enumerate(model.parameters.counts):
        scope = _scope_for_count(model, count)
        window = _static_window(count.measurement_window)
        if count.kind != ParameterCountKind.logical:
            gaps.append(
                ParameterEvidenceGap(
                    gap_id=_stable_id(f"descriptor-{count.kind.value}-{index}-unstructured"),
                    kind=count.kind,
                    scope=scope,
                    window=window,
                    reason=ParameterGapReason.unstructured_claim,
                    explanation=(
                        "Legacy descriptor count lacks the structured execution window required for authoritative reconciliation."
                    ),
                    resolution="Re-emit this axis as a structured worker or planner observation.",
                )
            )
            continue
        if count.evidence == EvidenceKind.unknown:
            gaps.append(
                ParameterEvidenceGap(
                    gap_id=_stable_id(f"descriptor-logical-{index}-unknown"),
                    kind=count.kind,
                    scope=scope,
                    window=window,
                    reason=ParameterGapReason.missing_observation,
                    explanation="The descriptor supplied a number with unknown evidence.",
                    resolution="Supply declared, estimated, or measured evidence with provenance.",
                )
            )
            continue
        if count.evidence == EvidenceKind.measured and model.source.snapshot_sha256 is None:
            gaps.append(
                ParameterEvidenceGap(
                    gap_id=_stable_id(f"descriptor-logical-{index}-unhashed"),
                    kind=count.kind,
                    scope=scope,
                    window=window,
                    reason=ParameterGapReason.unpinned_model,
                    explanation="Measured descriptor evidence is not pinned to a model hash.",
                    resolution="Hash-pin the inspected model snapshot before claiming measurement.",
                )
            )
            continue
        source_kind = (
            ParameterEvidenceSourceKind.model_config
            if count.source.startswith("config.")
            else ParameterEvidenceSourceKind.model_descriptor
        )
        observations.append(
            ParameterObservation(
                observation_id=_stable_id(f"descriptor-logical-{index}"),
                kind=count.kind,
                value=count.value,
                scope=scope,
                window=window,
                evidence=count.evidence,
                source=ParameterEvidenceSource(
                    kind=source_kind,
                    producer="corpus_studio.parameter_accounting",
                    producer_version=__version__,
                    method=count.source,
                    captured_at=model.captured_at or generated_at,
                    source_ref=descriptor_ref,
                ),
                coverage=ParameterObservationCoverage.complete,
                value_relation=(
                    ParameterValueRelation.estimate
                    if count.evidence == EvidenceKind.estimated
                    else ParameterValueRelation.exact
                ),
                identity_basis=ParameterIdentityBasis.declared_definition,
                handling=count.handling,
                definition=(
                    "Declared logical-coordinate count imported from the static ModelDescriptor."
                ),
                notes=count.notes,
            )
        )
    if model_ref.hash is None or model.source.snapshot_sha256 is None:
        gaps.append(
            ParameterEvidenceGap(
                gap_id="model-revision-not-content-pinned",
                kind=ParameterCountKind.logical,
                scope=_model_scope(model_ref),
                window=_static_window(),
                reason=ParameterGapReason.unpinned_model,
                explanation=(
                    "The report is not pinned to a complete model-snapshot content hash; an inventory hash is not equivalent."
                ),
                resolution="Re-run model inspection with a complete, fully hashed weight inventory.",
            )
        )
    return observations, gaps


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & 0x400)


def _safe_snapshot_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.exists() or not root.is_dir() or _is_link_like(root):
        raise ParameterAccountingError(f"snapshot path must be a regular directory: {root}")
    return root.resolve(strict=True)


def _safe_snapshot_file(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    if not candidate.exists() or not candidate.is_file() or _is_link_like(candidate):
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"weight path is missing, non-regular, or link-like: {relative}",
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"weight path escapes the inspected snapshot: {relative}",
        ) from exc
    return resolved


def _tensor_coordinates(shape: object, *, tensor_name: str) -> int:
    if not isinstance(shape, list) or len(shape) > _MAX_TENSOR_DIMENSIONS:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"invalid shape for safetensors tensor '{tensor_name}'",
        )
    coordinates = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"invalid dimension for safetensors tensor '{tensor_name}'",
            )
        coordinates *= dimension
        if coordinates > _MAX_PARAMETER_COORDINATES:
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"safetensors tensor '{tensor_name}' exceeds the supported coordinate bound",
            )
    return coordinates


def _decode_safetensors_header(
    header_bytes: bytes,
    *,
    data_bytes: int,
    relative: str,
) -> tuple[int, list[dict[str, object]], list[tuple[int, int, str]]]:
    """Decode and structurally validate one Safetensors header and complete data layout."""

    if not header_bytes.startswith(b"{") or not header_bytes.rstrip(b" ").endswith(b"}"):
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors header has invalid framing or padding: {relative}",
        )

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _HeaderEvidenceError(
                    ParameterGapReason.malformed_evidence,
                    f"duplicate safetensors header key '{key}': {relative}",
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            header_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"invalid safetensors header JSON: {relative}",
        ) from exc
    if not isinstance(payload, dict):
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors header must be an object: {relative}",
        )
    metadata = payload.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors metadata must contain only string pairs: {relative}",
        )
    tensor_items = [(name, value) for name, value in payload.items() if name != "__metadata__"]
    if len(tensor_items) > _MAX_SAFETENSORS_TENSORS:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors header has too many tensors: {relative}",
        )
    if data_bytes < 0:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors header exceeds file size: {relative}",
        )
    coordinates = 0
    records: list[dict[str, object]] = []
    intervals: list[tuple[int, int, str]] = []
    for name, value in sorted(tensor_items):
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"invalid safetensors tensor record: {relative}",
            )
        dtype = value.get("dtype")
        offsets = value.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES or (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in offsets
            )
        ):
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"invalid dtype or offsets for safetensors tensor '{name}'",
            )
        start, end = offsets
        if end < start or end > data_bytes:
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"out-of-range offsets for safetensors tensor '{name}'",
            )
        tensor_coordinates = _tensor_coordinates(value.get("shape"), tensor_name=name)
        if end - start != tensor_coordinates * _DTYPE_BYTES[dtype]:
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"byte size does not match shape/dtype for safetensors tensor '{name}'",
            )
        coordinates += tensor_coordinates
        if coordinates > _MAX_PARAMETER_COORDINATES:
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                "safetensors inventory exceeds the supported coordinate bound",
            )
        intervals.append((start, end, name))
        records.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": value.get("shape"),
                "data_offsets": offsets,
            }
        )
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            reason = "overlapping" if start < cursor else "non-contiguous"
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"{reason} safetensors data range at tensor '{name}'",
            )
        cursor = end
    if cursor != data_bytes:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors tensor ranges do not cover the complete data buffer: {relative}",
        )
    return coordinates, records, intervals


def _read_safetensors_header(
    path: Path,
    relative: str,
    *,
    expected_size: int,
    expected_sha256: str | None,
    hash_tensor_bytes: bool = False,
) -> _SafetensorsFileEvidence:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size != expected_size:
                raise _HeaderEvidenceError(
                    ParameterGapReason.changed_during_read,
                    f"safetensors size differs from the descriptor inventory: {relative}",
                )
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise _HeaderEvidenceError(
                    ParameterGapReason.malformed_evidence,
                    f"truncated safetensors header length: {relative}",
                )
            header_length = struct.unpack("<Q", raw_length)[0]
            if (
                header_length <= 0
                or header_length > _MAX_SAFETENSORS_HEADER_BYTES
                or header_length % 8 != 0
            ):
                raise _HeaderEvidenceError(
                    ParameterGapReason.malformed_evidence,
                    f"safetensors header length is invalid or unaligned: {relative}",
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise _HeaderEvidenceError(
                    ParameterGapReason.malformed_evidence,
                    f"truncated safetensors header: {relative}",
                )
            data_start = 8 + header_length
            coordinates, records, intervals = _decode_safetensors_header(
                header_bytes,
                data_bytes=before.st_size - data_start,
                relative=relative,
            )
            actual_sha256 = None
            if expected_sha256 is not None or hash_tensor_bytes:
                handle.seek(0)
                digest = hashlib.sha256()
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                actual_sha256 = digest.hexdigest()
            if hash_tensor_bytes:
                records_by_name = {str(record["name"]): record for record in records}
                for start, end, name in intervals:
                    handle.seek(data_start + start)
                    remaining = end - start
                    tensor_digest = hashlib.sha256()
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise _HeaderEvidenceError(
                                ParameterGapReason.changed_during_read,
                                f"truncated safetensors tensor bytes for '{name}'",
                            )
                        tensor_digest.update(chunk)
                        remaining -= len(chunk)
                    records_by_name[name]["content_sha256"] = tensor_digest.hexdigest()
            after = os.fstat(handle.fileno())
            path_after = path.stat()
    except _HeaderEvidenceError:
        raise
    except OSError as exc:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"cannot read safetensors header '{relative}': {exc}",
        ) from exc
    opened_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or opened_identity != path_identity
    ):
        raise _HeaderEvidenceError(
            ParameterGapReason.changed_during_read,
            f"safetensors file changed while its header was read: {relative}",
        )
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise _HeaderEvidenceError(
            ParameterGapReason.changed_during_read,
            f"safetensors content differs from the descriptor inventory: {relative}",
        )

    return _SafetensorsFileEvidence(
        path=relative,
        coordinates=coordinates,
        tensors=tuple(records),
        integrity_verified=expected_sha256 is not None,
        content_sha256=actual_sha256,
    )


def validate_safetensors_tensor_file(path: str | Path) -> SafetensorsTensorStateEvidence:
    """Parse/hash one stable, regular Safetensors file and return its exact tensor identity.

    This is the dependency-light artifact-admission surface. It deliberately reuses the same
    strict header, shape, dtype, offset, overlap, containment, and change-during-read checks as
    static parameter accounting, without loading tensor payloads or importing Safetensors/torch.
    """

    candidate = Path(path)
    try:
        root = _safe_snapshot_root(candidate.parent)
        safe_file = _safe_snapshot_file(root, candidate.name)
        stat = safe_file.stat()
        evidence = _read_safetensors_header(
            safe_file,
            candidate.name,
            expected_size=stat.st_size,
            expected_sha256=None,
            hash_tensor_bytes=True,
        )
    except (_HeaderEvidenceError, OSError) as exc:
        raise ParameterAccountingError(str(exc)) from exc
    names = tuple(str(record["name"]) for record in evidence.tensors)
    if not names:
        raise ParameterAccountingError("adapter Safetensors contains no tensors")
    if evidence.content_sha256 is None or any(
        record.get("content_sha256") is None for record in evidence.tensors
    ):
        raise ParameterAccountingError("adapter Safetensors tensor hashing was incomplete")
    return SafetensorsTensorStateEvidence(
        content_sha256=evidence.content_sha256,
        tensor_state_sha256=canonical_tensor_state_sha256(evidence.tensors),
        tensor_names=names,
    )


def _read_safetensors_index(
    root: Path,
    descriptor: DescriptorFile,
) -> dict[str, str]:
    path = _safe_snapshot_file(root, descriptor.path)
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size != descriptor.size_bytes:
                raise _HeaderEvidenceError(
                    ParameterGapReason.changed_during_read,
                    f"safetensors index size differs from the descriptor: {descriptor.path}",
                )
            if before.st_size <= 0 or before.st_size > _MAX_SAFETENSORS_HEADER_BYTES:
                raise _HeaderEvidenceError(
                    ParameterGapReason.malformed_evidence,
                    f"safetensors index exceeds the bounded reader limit: {descriptor.path}",
                )
            payload_bytes = handle.read(_MAX_SAFETENSORS_HEADER_BYTES + 1)
            after = os.fstat(handle.fileno())
    except _HeaderEvidenceError:
        raise
    except OSError as exc:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"cannot read safetensors index '{descriptor.path}': {exc}",
        ) from exc
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise _HeaderEvidenceError(
            ParameterGapReason.changed_during_read,
            f"safetensors index changed while it was read: {descriptor.path}",
        )
    if len(payload_bytes) != before.st_size:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"truncated safetensors index: {descriptor.path}",
        )
    if descriptor.hash_status != "verified" or descriptor.sha256 is None:
        raise _HeaderEvidenceError(
            ParameterGapReason.unpinned_model,
            f"safetensors index is not content-pinned: {descriptor.path}",
        )
    if hashlib.sha256(payload_bytes).hexdigest() != descriptor.sha256:
        raise _HeaderEvidenceError(
            ParameterGapReason.changed_during_read,
            f"safetensors index content differs from the descriptor: {descriptor.path}",
        )

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _HeaderEvidenceError(
                    ParameterGapReason.malformed_evidence,
                    f"duplicate safetensors index key '{key}': {descriptor.path}",
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"invalid safetensors index JSON: {descriptor.path}",
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors index requires a weight_map object: {descriptor.path}",
        )
    weight_map = payload["weight_map"]
    if not weight_map:
        raise _HeaderEvidenceError(
            ParameterGapReason.malformed_evidence,
            f"safetensors index weight_map cannot be empty: {descriptor.path}",
        )
    index_parent = PurePosixPath(descriptor.path).parent
    normalized: dict[str, str] = {}
    for tensor_name, target in weight_map.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or not isinstance(target, str)
            or not target
            or "\\" in target
        ):
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"invalid safetensors weight_map entry: {descriptor.path}",
            )
        target_path = PurePosixPath(target)
        if target_path.is_absolute() or ".." in target_path.parts:
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"unsafe safetensors shard path in index: {target}",
            )
        normalized_target = (index_parent / target_path).as_posix()
        if not normalized_target.lower().endswith(".safetensors"):
            raise _HeaderEvidenceError(
                ParameterGapReason.malformed_evidence,
                f"non-safetensors shard in safetensors index: {target}",
            )
        normalized[tensor_name] = normalized_target
    return normalized


def _safetensors_evidence(
    model: ModelDescriptor,
    snapshot_root: str | Path,
    *,
    generated_at: str,
) -> tuple[list[ParameterObservation], list[ParameterEvidenceGap]]:
    root = _safe_snapshot_root(snapshot_root)
    model_ref = _model_ref(model)
    logical_counts = [
        item
        for item in model.parameters.counts
        if item.kind == ParameterCountKind.logical and item.scope == "model"
    ]
    base_scope = _model_scope(model_ref, coordinate_hash=_coordinate_hash(model))
    base_window = _static_window("safetensors-headers")
    gaps: list[ParameterEvidenceGap] = []
    weight_files = [item for item in model.files if item.role == DescriptorFileRole.weights]
    safe_files = [item for item in weight_files if item.format == ModelFormat.safetensors]
    if not safe_files:
        return [], [
            ParameterEvidenceGap(
                gap_id="safetensors-header-evidence-unavailable",
                kind=ParameterCountKind.logical,
                scope=base_scope,
                window=base_window,
                reason=ParameterGapReason.unsupported_format,
                explanation="No safetensors weight files are available for bounded header enumeration.",
                resolution="Provide safetensors weights or another explicit logical-coordinate source.",
            )
        ]
    evidence_files: list[_SafetensorsFileEvidence] = []
    seen_tensor_names: set[str] = set()
    duplicate_names: set[str] = set()
    for index, item in enumerate(safe_files):
        try:
            evidence = _read_safetensors_header(
                _safe_snapshot_file(root, item.path),
                item.path,
                expected_size=item.size_bytes,
                expected_sha256=item.sha256 if item.hash_status == "verified" else None,
            )
        except _HeaderEvidenceError as exc:
            gaps.append(
                ParameterEvidenceGap(
                    gap_id=_stable_id(f"safetensors-{index}-{exc.reason.value}"),
                    kind=ParameterCountKind.logical,
                    scope=base_scope,
                    window=base_window,
                    reason=exc.reason,
                    explanation=str(exc),
                    resolution="Repair or replace the malformed/unstable weight file and inspect again.",
                )
            )
            continue
        for record in evidence.tensors:
            name = str(record["name"])
            if name in seen_tensor_names:
                duplicate_names.add(name)
            seen_tensor_names.add(name)
        evidence_files.append(evidence)
    if duplicate_names:
        gaps.append(
            ParameterEvidenceGap(
                gap_id="duplicate-safetensors-tensor-names",
                kind=ParameterCountKind.logical,
                scope=base_scope,
                window=base_window,
                reason=ParameterGapReason.incomparable_evidence,
                explanation=(
                    "Tensor names repeat across safetensors files: "
                    + ", ".join(sorted(duplicate_names)[:20])
                ),
                resolution="Provide a non-overlapping shard inventory with stable tensor identity.",
            )
        )
    if not evidence_files:
        return [], gaps
    index_consistent = len(evidence_files) == len(safe_files)
    safe_indexes = [
        item
        for item in model.files
        if item.role == DescriptorFileRole.weight_index
        and item.path.lower().endswith(".safetensors.index.json")
    ]
    if index_consistent and len(safe_indexes) > 1:
        index_consistent = False
        gaps.append(
            ParameterEvidenceGap(
                gap_id="ambiguous-safetensors-shard-index",
                kind=ParameterCountKind.logical,
                scope=base_scope,
                window=base_window,
                reason=ParameterGapReason.incomplete_inventory,
                explanation="More than one safetensors shard index was recorded.",
                resolution="Provide one authoritative, content-pinned safetensors shard index.",
            )
        )
    elif index_consistent and safe_indexes:
        try:
            weight_map = _read_safetensors_index(root, safe_indexes[0])
            evidence_by_path = {item.path: item for item in evidence_files}
            header_tensor_names = {
                str(record["name"])
                for evidence in evidence_files
                for record in evidence.tensors
            }
            mapped_paths = set(weight_map.values())
            if mapped_paths != set(evidence_by_path) or set(weight_map) != header_tensor_names:
                raise _HeaderEvidenceError(
                    ParameterGapReason.incomplete_inventory,
                    "safetensors shard index does not exactly cover the recorded shards/tensors",
                )
            for tensor_name, shard_path in weight_map.items():
                shard_names = {
                    str(record["name"])
                    for record in evidence_by_path[shard_path].tensors
                }
                if tensor_name not in shard_names:
                    raise _HeaderEvidenceError(
                        ParameterGapReason.incomplete_inventory,
                        f"safetensors index maps '{tensor_name}' to the wrong shard",
                    )
        except _HeaderEvidenceError as exc:
            index_consistent = False
            gaps.append(
                ParameterEvidenceGap(
                    gap_id=_stable_id(f"safetensors-index-{exc.reason.value}"),
                    kind=ParameterCountKind.logical,
                    scope=base_scope,
                    window=base_window,
                    reason=exc.reason,
                    explanation=str(exc),
                    resolution="Repair and content-pin one exact safetensors shard index.",
                )
            )
    elif index_consistent and len(safe_files) > 1:
        index_consistent = False
        gaps.append(
            ParameterEvidenceGap(
                gap_id="missing-safetensors-shard-index",
                kind=ParameterCountKind.logical,
                scope=base_scope,
                window=base_window,
                reason=ParameterGapReason.incomplete_inventory,
                explanation="Multiple safetensors shards lack one authoritative shard index.",
                resolution="Provide and content-pin model.safetensors.index.json.",
            )
        )
    canonical_files = [
        {"path": item.path, "coordinates": item.coordinates, "tensors": list(item.tensors)}
        for item in sorted(evidence_files, key=lambda item: item.path)
    ]
    header_hash = _canonical_sha256(canonical_files)
    coordinates = sum(item.coordinates for item in evidence_files)
    complete_inventory = (
        model.inventory_complete
        and len(safe_files) == len(weight_files)
        and len(evidence_files) == len(safe_files)
        and not duplicate_names
        and index_consistent
    )
    verified_inventory = complete_inventory and all(
        item.integrity_verified for item in evidence_files
    )
    declared_matches = [
        item
        for item in logical_counts
        if item.value == coordinates
        and item.evidence in {EvidenceKind.declared, EvidenceKind.measured}
    ]
    resolved_matches = [
        item for item in declared_matches if _count_handling_complete(item.handling)
    ]
    handling = (
        resolved_matches[0].handling
        if resolved_matches
        else declared_matches[0].handling
        if declared_matches
        else logical_counts[0].handling
        if logical_counts
        else ParameterCountHandling()
    )
    exact_logical = verified_inventory and bool(resolved_matches)
    if _coordinate_hash(model) is not None:
        scope = _model_scope(model_ref, coordinate_hash=_coordinate_hash(model))
    else:
        scope = ParameterScope(
            scope_id="safetensors-inventory",
            kind=ParameterScopeKind.model,
            model_ref=model_ref,
            coordinate_universe_id=_stable_id(f"{model.model_id}-safetensors-coordinates"),
            coordinate_universe_sha256=header_hash,
            definition="Coordinate universe enumerated from the supplied safetensors inventory.",
        )
    observation = ParameterObservation(
        observation_id="safetensors-header-logical",
        kind=ParameterCountKind.logical,
        value=coordinates,
        scope=scope,
        window=base_window,
        evidence=EvidenceKind.measured,
        source=ParameterEvidenceSource(
            kind=ParameterEvidenceSourceKind.safetensors_header,
            producer="corpus_studio.parameter_accounting",
            producer_version=__version__,
            method="bounded-safetensors-header-enumeration-v1",
            captured_at=generated_at,
            source_ref=Ref(
                id=_stable_id(f"{model.model_id}-safetensors-headers"),
                hash=HashRef(algo="sha256", value=header_hash),
            ),
        ),
        coverage=(
            ParameterObservationCoverage.complete
            if complete_inventory
            else ParameterObservationCoverage.partial
        ),
        value_relation=(
            ParameterValueRelation.exact
            if exact_logical
            else ParameterValueRelation.estimate
        ),
        identity_basis=(
            ParameterIdentityBasis.independent_coordinates
            if exact_logical
            else ParameterIdentityBasis.stored_tensor_elements
        ),
        handling=handling,
        definition=(
            "Logical coordinates corroborated by complete safetensors headers and resolved handling."
            if exact_logical
            else "Stored safetensors tensor elements used only as an estimate of logical coordinates."
        ),
        assumptions=sorted(
            [
                "Safetensors tensor shapes were enumerated without loading tensor data.",
                *(
                    ["A resolved descriptor declaration agrees with the complete tensor inventory."]
                    if exact_logical
                    else []
                ),
            ]
        ),
    )
    if not exact_logical:
        gaps.append(
            ParameterEvidenceGap(
                gap_id="safetensors-elements-not-logical",
                kind=ParameterCountKind.logical,
                scope=scope,
                window=base_window,
                reason=ParameterGapReason.stored_elements_not_logical,
                explanation=(
                    "Header element totals do not by themselves resolve aliases, ties, generated state, buffers, replicas, or quantized representation."
                ),
                resolution=(
                    "Resolve handling and corroborate the complete header total with an exact logical declaration."
                ),
            )
        )
    if not model.inventory_complete or len(safe_files) != len(weight_files):
        gaps.append(
            ParameterEvidenceGap(
                gap_id="incomplete-weight-inventory",
                kind=ParameterCountKind.logical,
                scope=scope,
                window=base_window,
                reason=ParameterGapReason.incomplete_inventory,
                explanation="Safetensors headers do not cover every recorded model weight file.",
                resolution="Provide a complete safetensors-only weight inventory.",
            )
        )
    if complete_inventory and not verified_inventory:
        gaps.append(
            ParameterEvidenceGap(
                gap_id="unverified-safetensors-file-integrity",
                kind=ParameterCountKind.logical,
                scope=scope,
                window=base_window,
                reason=ParameterGapReason.unpinned_model,
                explanation=(
                    "Safetensors headers were read, but every source file was not verified against a descriptor content hash."
                ),
                resolution="Re-inspect with full weight hashing before claiming exact header evidence.",
            )
        )
    if logical_counts and all(item.value != coordinates for item in logical_counts):
        gaps.append(
            ParameterEvidenceGap(
                gap_id="declared-header-count-disagreement",
                kind=ParameterCountKind.logical,
                scope=scope,
                window=base_window,
                reason=ParameterGapReason.incomparable_evidence,
                explanation=(
                    "Declared logical count disagrees with the stored safetensors element total; the quantities may use different identity semantics."
                ),
                resolution="Resolve coordinate identity/handling before selecting an authoritative value.",
            )
        )
    return [observation], gaps


def build_model_parameter_accounting(
    model: ModelDescriptor,
    *,
    snapshot_root: str | Path | None = None,
    report_id: str | None = None,
    now: Callable[[], datetime] = _utcnow,
) -> ParameterAccountingReport:
    """Build an offline static report from descriptor claims and optional bounded header evidence."""

    generated_at = _timestamp(now)
    observations, gaps = _descriptor_observations(model, generated_at=generated_at)
    if snapshot_root is not None:
        header_observations, header_gaps = _safetensors_evidence(
            model,
            snapshot_root,
            generated_at=generated_at,
        )
        observations.extend(header_observations)
        gaps.extend(header_gaps)
    return _seal_report(
        report_id=report_id or f"{model.model_id}-parameter-accounting",
        generated_at=generated_at,
        profile=ParameterAccountingProfile.model_static,
        model_ref=_model_ref(model),
        observations=observations,
        extra_gaps=gaps,
        notes=[
            "Static accounting does not infer active, resident, touched, updated, or exposed coordinates.",
            "Memory bytes are not converted into parameter-coordinate residency.",
        ],
    )


def reconcile_parameter_accounting_events(
    base_report: ParameterAccountingReport,
    events: Sequence[RunEvent],
    *,
    profile: ParameterAccountingProfile,
    report_id: str | None = None,
    artifact_refs: Sequence[Ref] = (),
    evaluation_refs: Sequence[Ref] = (),
    now: Callable[[], datetime] = _utcnow,
) -> ParameterAccountingReport:
    """Reconcile typed worker observations with a hash-verified static parent report."""

    base_report = _revalidate_report(base_report, label="base parameter-accounting report")
    if not verify_parameter_accounting_hash(base_report):
        raise ParameterAccountingError("base parameter-accounting report hash mismatch")
    if profile not in {
        ParameterAccountingProfile.training_runtime,
        ParameterAccountingProfile.inference_runtime,
        ParameterAccountingProfile.checkpoint,
        ParameterAccountingProfile.evaluation,
    }:
        raise ParameterAccountingError("event reconciliation requires a runtime/checkpoint/evaluation profile")
    if not events:
        raise ParameterAccountingError("event reconciliation requires at least one RunEvent")
    run_ids = {item.run_id for item in events}
    if len(run_ids) != 1:
        raise ParameterAccountingError("RunEvent stream contains more than one run_id")
    sequences = [item.seq for item in events]
    if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
        raise ParameterAccountingError("RunEvent stream must be sorted by unique sequence number")
    event_observations = [
        observation
        for event in events
        if event.metrics is not None
        for observation in event.metrics.parameter_observations
    ]
    run_ref = Ref(id=next(iter(run_ids)))
    observations = [*base_report.observations, *event_observations]
    return _seal_report(
        report_id=report_id or f"{run_ref.id}-parameter-accounting",
        generated_at=_timestamp(now),
        profile=profile,
        model_ref=base_report.model_ref,
        observations=observations,
        run_ref=run_ref,
        artifact_refs=artifact_refs,
        evaluation_refs=evaluation_refs,
        parent_report_refs=[
            Ref(
                id=base_report.report_id,
                hash=HashRef(algo="sha256", value=base_report.report_hash),
            )
        ],
        notes=[
            "Only typed worker observations are reconciled; allocator bytes never become N_resident.",
            "Agreement with declared or estimated evidence does not promote it to measured evidence.",
        ],
    )


def load_parameter_events(path: str | Path) -> list[RunEvent]:
    """Load a bounded JSONL RunEvent stream without executing worker or model code."""

    source = Path(path)
    events: list[RunEvent] = []
    try:
        with source.open("rb") as handle:
            line_number = 0
            while line := handle.readline(_MAX_EVENT_LINE_BYTES + 1):
                line_number += 1
                if line_number > _MAX_EVENTS:
                    raise ParameterAccountingError("RunEvent stream exceeds the event limit")
                if len(line) > _MAX_EVENT_LINE_BYTES:
                    raise ParameterAccountingError(
                        f"RunEvent line {line_number} exceeds the bounded line limit"
                    )
                if not line.strip():
                    continue
                try:
                    events.append(RunEvent.model_validate_json(line))
                except (ValueError, RecursionError) as exc:
                    raise ParameterAccountingError(
                        f"invalid RunEvent JSONL at line {line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise ParameterAccountingError(f"cannot read RunEvent stream: {exc}") from exc
    return events


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_like(path.parent) or (path.exists() and _is_link_like(path)):
        raise ParameterAccountingError(f"output path cannot be link-like: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_parameter_accounting_report(
    report: ParameterAccountingReport,
    path: str | Path,
) -> Path:
    report = _revalidate_report(report, label="parameter-accounting report")
    if not verify_parameter_accounting_hash(report):
        raise ParameterAccountingError("refusing to write a report with an invalid hash")
    output = Path(path)
    _atomic_write_json(output, report.model_dump(mode="json"))
    return output
