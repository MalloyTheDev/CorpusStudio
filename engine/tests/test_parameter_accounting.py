"""MoE-safe parameter-accounting contracts, evidence producers, and CLI surfaces."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

from pydantic import ValidationError
import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.common import HashRef, MemoryMetrics, Ref
from corpus_studio.platform.contracts import (
    EventMetrics,
    ModelDescriptor,
    ParameterCountHandling,
    ParameterEvidenceSource,
    ParameterObservation,
    ParameterScope,
    ParameterWindow,
    RunEvent,
)
from corpus_studio.platform.enums import (
    CountHandling,
    EvidenceKind,
    MemoryTier,
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
from corpus_studio.platform.model_inspector import (
    ModelInspectionBundle,
    ModelInspectionError,
    inspect_model,
    inspect_model_bundle,
    write_inspection_bundle,
)
import corpus_studio.platform.parameter_accounting as accounting
from corpus_studio.platform.parameter_accounting import (
    ParameterAccountingError,
    build_model_parameter_accounting,
    load_parameter_events,
    parameter_accounting_hash_for,
    reconcile_parameter_accounting_events,
    validate_safetensors_tensor_file,
    verify_parameter_accounting_hash,
    write_parameter_accounting_report,
)


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
NOW_TEXT = "2026-07-13T12:00:00Z"
runner = CliRunner()


def _fixed_now() -> datetime:
    return NOW


def _resolved_handling() -> ParameterCountHandling:
    return ParameterCountHandling(
        tied=CountHandling.deduplicated,
        shared=CountHandling.deduplicated,
        replicated=CountHandling.excluded,
        generated=CountHandling.excluded,
        quantized=CountHandling.included,
        optimizer_shadows=CountHandling.not_applicable,
        decompressed_caches=CountHandling.not_applicable,
    )


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header: dict[str, object] = {}
    offset = 0
    dtype_bytes = {"F16": 2, "F32": 4, "BF16": 2, "I8": 1}
    for name, (dtype, shape) in sorted(tensors.items()):
        coordinates = 1
        for dimension in shape:
            coordinates *= dimension
        end = offset + coordinates * dtype_bytes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, end],
        }
        offset = end
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + (b"\0" * offset))


def _model_snapshot(
    tmp_path: Path,
    *,
    declared_count: int = 6,
    valid_safetensors: bool = True,
    model_id: str = "tiny-model",
) -> tuple[ModelDescriptor, Path]:
    root = tmp_path / model_id
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "num_parameters": declared_count,
                "torch_dtype": "float32",
                "vocab_size": 8,
            }
        ),
        encoding="utf-8",
    )
    if valid_safetensors:
        _write_safetensors(root / "model.safetensors", {"weight": ("F32", [2, 3])})
    else:
        (root / "model.safetensors").write_bytes(b"not-a-safetensors-file")
    inspected = inspect_model(
        root,
        model_id=model_id,
        hash_weights=True,
        now=_fixed_now,
    )
    count = inspected.parameters.counts[0].model_copy(
        update={"handling": _resolved_handling()}
    )
    parameters = inspected.parameters.model_copy(update={"counts": [count]})
    model = ModelDescriptor.model_validate(
        inspected.model_copy(update={"parameters": parameters}).model_dump(mode="json")
    )
    return model, root


def _scope(model_ref: Ref) -> ParameterScope:
    coordinate_hash = model_ref.hash.value if model_ref.hash is not None else None
    return ParameterScope(
        scope_id="model",
        kind=ParameterScopeKind.model,
        model_ref=model_ref,
        coordinate_universe_id=f"{model_ref.id}-coordinates",
        coordinate_universe_sha256=coordinate_hash,
        definition="All independently addressable model coordinates.",
    )


def _window(kind: ParameterCountKind, run_ref: Ref) -> ParameterWindow:
    if kind == ParameterCountKind.logical:
        return ParameterWindow(
            window_id="static-model",
            kind=ParameterWindowKind.static_snapshot,
            definition="One immutable model snapshot.",
        )
    if kind == ParameterCountKind.active_token:
        return ParameterWindow(
            window_id="token-0",
            kind=ParameterWindowKind.token,
            definition="Token zero of sequence s0.",
            run_ref=run_ref,
            sequence_id="s0",
            token_index=0,
        )
    if kind == ParameterCountKind.active_sequence:
        return ParameterWindow(
            window_id="sequence-s0",
            kind=ParameterWindowKind.sequence,
            definition="Sequence s0.",
            run_ref=run_ref,
            sequence_id="s0",
        )
    if kind == ParameterCountKind.resident:
        return ParameterWindow(
            window_id="residency-instant",
            kind=ParameterWindowKind.instant,
            definition="One worker residency sample.",
            run_ref=run_ref,
            captured_at=NOW_TEXT,
        )
    return ParameterWindow(
        window_id="run-window",
        kind=ParameterWindowKind.run,
        definition="The complete worker run.",
        run_ref=run_ref,
    )


def _observation(
    kind: ParameterCountKind,
    value: int,
    model_ref: Ref,
    *,
    observation_id: str | None = None,
    coverage: ParameterObservationCoverage = ParameterObservationCoverage.complete,
    relation: ParameterValueRelation = ParameterValueRelation.exact,
) -> ParameterObservation:
    run_ref = Ref(id="run-1")
    identifier = observation_id or f"runtime-{kind.value}"
    scope = _scope(model_ref)
    if kind == ParameterCountKind.resident:
        scope = scope.model_copy(
            update={
                "scope_id": "model-cuda-0-residency",
                "kind": ParameterScopeKind.device_residency,
                "device_id": "cuda:0",
                "memory_tier": MemoryTier.gpu,
                "definition": "Model coordinates resident on CUDA device zero.",
            }
        )
    return ParameterObservation(
        observation_id=identifier,
        kind=kind,
        value=value,
        scope=scope,
        window=_window(kind, run_ref),
        evidence=EvidenceKind.measured,
        source=ParameterEvidenceSource(
            kind=ParameterEvidenceSourceKind.backend_worker,
            producer="test-worker",
            producer_version="1.0.0",
            method="coordinate-identity-set-v1",
            captured_at=NOW_TEXT,
            source_ref=Ref(
                id=f"{identifier}-source",
                hash=HashRef(
                    algo="sha256",
                    value=hashlib.sha256(identifier.encode("ascii")).hexdigest(),
                ),
            ),
        ),
        coverage=coverage,
        value_relation=relation,
        identity_basis=(
            ParameterIdentityBasis.independent_coordinates
            if kind == ParameterCountKind.logical
            else ParameterIdentityBasis.runtime_identity_set
        ),
        handling=_resolved_handling(),
        definition=f"Measured {kind.value} coordinate identity set.",
    )


def _runtime_event(observations: list[ParameterObservation]) -> RunEvent:
    return RunEvent(
        event_type="metric",
        run_id="run-1",
        seq=0,
        emitted_at=NOW_TEXT,
        metrics=EventMetrics(
            parameter_observations=sorted(
                observations,
                key=lambda item: item.observation_id,
            )
        ),
    )


def _runtime_observations(model_ref: Ref) -> list[ParameterObservation]:
    values = {
        ParameterCountKind.active_token: 4,
        ParameterCountKind.touched_window: 6,
        ParameterCountKind.resident: 3,
        ParameterCountKind.updated_window: 2,
        ParameterCountKind.exposed_window: 6,
    }
    return [_observation(kind, value, model_ref) for kind, value in values.items()]


def test_static_report_is_hash_sealed_and_complete_when_evidence_is_resolved(tmp_path: Path):
    model, root = _model_snapshot(tmp_path)

    report = build_model_parameter_accounting(model, snapshot_root=root, now=_fixed_now)

    assert report.status == ParameterAccountingStatus.complete
    assert verify_parameter_accounting_hash(report)
    assert {item.value for item in report.observations} == {6}
    assert {item.source.kind for item in report.observations} == {
        ParameterEvidenceSourceKind.model_config,
        ParameterEvidenceSourceKind.safetensors_header,
    }
    assert report.gaps == []


def test_report_seal_detects_tampering(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)

    tampered = report.model_copy(update={"notes": [*report.notes, "changed"]})

    assert verify_parameter_accounting_hash(report)
    assert not verify_parameter_accounting_hash(tampered)
    assert report.status == ParameterAccountingStatus.incomplete
    assert any(
        gap.reason == ParameterGapReason.measured_evidence_required for gap in report.gaps
    )


def test_descriptor_observation_source_hash_binds_extracted_semantics(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    original = build_model_parameter_accounting(model, now=_fixed_now)
    changed_count = model.parameters.counts[0].model_copy(update={"notes": "changed semantics"})
    changed_model = model.model_copy(
        update={
            "parameters": model.parameters.model_copy(update={"counts": [changed_count]})
        }
    )
    changed = build_model_parameter_accounting(changed_model, now=_fixed_now)

    original_ref = original.observations[0].source.source_ref
    changed_ref = changed.observations[0].source.source_ref
    assert original_ref.id == "tiny-model-descriptor"
    assert original_ref.hash is not None and changed_ref.hash is not None
    assert original_ref.hash.value != changed_ref.hash.value


def test_unresolved_handling_is_a_gap_not_a_zero_or_silent_exact_value(tmp_path: Path):
    model, root = _model_snapshot(tmp_path)
    unresolved = inspect_model(root, model_id="unresolved", hash_weights=True, now=_fixed_now)

    report = build_model_parameter_accounting(unresolved, snapshot_root=root, now=_fixed_now)

    assert report.status == ParameterAccountingStatus.incomplete
    assert any(gap.reason == ParameterGapReason.unknown_handling for gap in report.gaps)
    assert all(item.value != 0 for item in report.observations)


def test_legacy_dynamic_descriptor_count_is_not_treated_as_structured_runtime_evidence(
    tmp_path: Path,
):
    model, _ = _model_snapshot(tmp_path)
    count = model.parameters.counts[0].model_copy(
        update={"kind": ParameterCountKind.effective}
    )
    modified = model.model_copy(
        update={"parameters": model.parameters.model_copy(update={"counts": [count]})}
    )

    report = build_model_parameter_accounting(modified, now=_fixed_now)

    assert any(gap.reason == ParameterGapReason.unstructured_claim for gap in report.gaps)
    assert all(item.kind != ParameterCountKind.effective for item in report.observations)


def test_unknown_descriptor_number_remains_a_missing_evidence_gap(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    count = model.parameters.counts[0].model_copy(update={"evidence": EvidenceKind.unknown})
    modified = model.model_copy(
        update={"parameters": model.parameters.model_copy(update={"counts": [count]})}
    )

    report = build_model_parameter_accounting(modified, now=_fixed_now)

    assert report.observations == []
    assert any(gap.reason == ParameterGapReason.missing_observation for gap in report.gaps)


def test_unpinned_measured_descriptor_count_is_refused(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    count = model.parameters.counts[0].model_copy(update={"evidence": EvidenceKind.measured})
    modified = model.model_copy(
        update={
            "source": model.source.model_copy(update={"snapshot_sha256": None}),
            "parameters": model.parameters.model_copy(update={"counts": [count]}),
        }
    )

    report = build_model_parameter_accounting(modified, now=_fixed_now)

    assert report.observations == []
    assert any(gap.reason == ParameterGapReason.unpinned_model for gap in report.gaps)


def test_unknown_observation_evidence_is_rejected(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)
    observation = _observation(ParameterCountKind.resident, 0, report.model_ref)
    payload = observation.model_dump(mode="json")
    payload["evidence"] = "unknown"

    with pytest.raises(ValidationError, match="represented as a gap"):
        ParameterObservation.model_validate(payload)


def test_estimated_observation_cannot_claim_an_exact_value(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)
    observation = _observation(ParameterCountKind.resident, 1, report.model_ref)
    payload = observation.model_dump(mode="json")
    payload["evidence"] = "estimated"

    with pytest.raises(ValidationError, match="cannot claim an exact value"):
        ParameterObservation.model_validate(payload)


def test_measured_zero_is_valid_when_the_source_is_pinned(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)

    observation = _observation(ParameterCountKind.updated_window, 0, report.model_ref)

    assert observation.value == 0
    assert observation.evidence == EvidenceKind.measured


def test_measured_observation_requires_capture_time_and_pinned_source(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)
    observation = _observation(ParameterCountKind.resident, 1, report.model_ref)
    payload = observation.model_dump(mode="json")
    payload["source"]["source_ref"]["hash"] = None

    with pytest.raises(ValidationError, match="hash-pinned source"):
        ParameterObservation.model_validate(payload)


def test_resident_observation_requires_a_named_device_and_memory_tier(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)
    observation = _observation(ParameterCountKind.resident, 1, report.model_ref)
    payload = observation.model_dump(mode="json")
    payload["scope"]["kind"] = "model"
    payload["scope"]["device_id"] = None
    payload["scope"]["memory_tier"] = None

    with pytest.raises(ValidationError, match="device-residency scope"):
        ParameterObservation.model_validate(payload)
    with pytest.raises(ValidationError, match="device_id and memory_tier"):
        ParameterScope(
            scope_id="residency",
            kind=ParameterScopeKind.device_residency,
            model_ref=report.model_ref,
            coordinate_universe_id="model-coordinates",
            device_id="cuda:0",
            definition="Incomplete device scope.",
        )


def test_sparse_expert_scope_requires_stable_ids_and_universe_hash(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)

    with pytest.raises(ValidationError, match="stable expert IDs"):
        ParameterScope(
            scope_id="experts",
            kind=ParameterScopeKind.expert_set,
            model_ref=report.model_ref,
            coordinate_universe_id="expert-coordinates",
            definition="A routed expert subset.",
        )

    scope = ParameterScope(
        scope_id="experts",
        kind=ParameterScopeKind.expert_set,
        model_ref=report.model_ref,
        coordinate_universe_id="expert-coordinates",
        coordinate_universe_sha256="a" * 64,
        expert_ids=["expert-0", "expert-1"],
        definition="A stable routed expert subset.",
    )
    assert scope.expert_ids == ["expert-0", "expert-1"]


@pytest.mark.parametrize(
    ("kind", "kwargs", "message"),
    [
        (ParameterWindowKind.token, {"run_ref": Ref(id="run")}, "sequence_id"),
        (ParameterWindowKind.sequence, {"run_ref": Ref(id="run")}, "sequence_id"),
        (ParameterWindowKind.run, {"plan_ref": Ref(id="plan")}, "run_ref"),
    ],
)
def test_dynamic_windows_require_their_structured_anchors(
    kind: ParameterWindowKind,
    kwargs: dict[str, object],
    message: str,
):
    with pytest.raises(ValidationError, match=message):
        ParameterWindow(
            window_id="window",
            kind=kind,
            definition="Invalid incomplete window.",
            **kwargs,
        )


def test_runtime_reconciliation_can_complete_all_required_training_axes(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)

    report = reconcile_parameter_accounting_events(
        base,
        [_runtime_event(_runtime_observations(base.model_ref))],
        profile=ParameterAccountingProfile.training_runtime,
        now=_fixed_now,
    )

    assert report.status == ParameterAccountingStatus.complete
    assert report.gaps == []
    assert report.conflicts == []
    assert verify_parameter_accounting_hash(report)
    assert report.parent_report_refs[0].hash.value == base.report_hash


def test_runtime_conflicts_are_explicit_and_hash_sealed(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    observations = _runtime_observations(base.model_ref)
    observations = [
        _observation(ParameterCountKind.active_token, 7, base.model_ref)
        if item.kind == ParameterCountKind.active_token
        else item
        for item in observations
    ]

    report = reconcile_parameter_accounting_events(
        base,
        [_runtime_event(observations)],
        profile=ParameterAccountingProfile.training_runtime,
        now=_fixed_now,
    )

    assert report.status == ParameterAccountingStatus.conflicting
    assert any(item.reason_code == "count-exceeds-logical" for item in report.conflicts)
    assert verify_parameter_accounting_hash(report)


def test_same_key_exact_disagreement_becomes_an_explicit_conflict(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    observations = _runtime_observations(base.model_ref)
    observations.extend(
        [
            _observation(
                ParameterCountKind.active_token,
                3,
                base.model_ref,
                observation_id="runtime-active-token-second",
            )
        ]
    )

    report = reconcile_parameter_accounting_events(
        base,
        [_runtime_event(observations)],
        profile=ParameterAccountingProfile.training_runtime,
        now=_fixed_now,
    )

    assert any(item.reason_code == "same-key-value-mismatch" for item in report.conflicts)


def test_token_active_cannot_exceed_sequence_active_in_the_same_sequence(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    observations = _runtime_observations(base.model_ref)
    observations.append(_observation(ParameterCountKind.active_sequence, 3, base.model_ref))

    report = reconcile_parameter_accounting_events(
        base,
        [_runtime_event(observations)],
        profile=ParameterAccountingProfile.training_runtime,
        now=_fixed_now,
    )

    assert any(
        item.reason_code == "token-active-exceeds-sequence-active"
        for item in report.conflicts
    )


def test_runtime_observations_must_be_anchored_to_the_stream_run(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    observation = _observation(ParameterCountKind.active_token, 3, base.model_ref)
    payload = observation.model_dump(mode="json")
    payload["window"]["run_ref"] = {"id": "another-run"}
    foreign = ParameterObservation.model_validate(payload)

    with pytest.raises(ValidationError, match="report run_ref"):
        reconcile_parameter_accounting_events(
            base,
            [_runtime_event([foreign])],
            profile=ParameterAccountingProfile.training_runtime,
            now=_fixed_now,
        )


def test_one_window_id_cannot_describe_two_different_runtime_windows(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    first = _observation(ParameterCountKind.active_token, 3, base.model_ref)
    payload = first.model_dump(mode="json")
    payload["observation_id"] = "runtime-active-token-second"
    payload["window"]["token_index"] = 1
    second = ParameterObservation.model_validate(payload)

    with pytest.raises(ValidationError, match="window_id must have one definition"):
        reconcile_parameter_accounting_events(
            base,
            [_runtime_event([first, second])],
            profile=ParameterAccountingProfile.training_runtime,
            now=_fixed_now,
        )


def test_updated_does_not_have_to_be_less_than_exposed(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    observations = _runtime_observations(base.model_ref)
    observations = [
        _observation(ParameterCountKind.updated_window, 5, base.model_ref)
        if item.kind == ParameterCountKind.updated_window
        else _observation(ParameterCountKind.exposed_window, 1, base.model_ref)
        if item.kind == ParameterCountKind.exposed_window
        else item
        for item in observations
    ]

    report = reconcile_parameter_accounting_events(
        base,
        [_runtime_event(observations)],
        profile=ParameterAccountingProfile.training_runtime,
        now=_fixed_now,
    )

    assert report.status == ParameterAccountingStatus.complete
    assert report.conflicts == []


def test_updated_cannot_exceed_touched_in_the_same_exact_window(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    observations = _runtime_observations(base.model_ref)
    observations = [
        _observation(ParameterCountKind.updated_window, 6, base.model_ref)
        if item.kind == ParameterCountKind.updated_window
        else _observation(ParameterCountKind.touched_window, 5, base.model_ref)
        if item.kind == ParameterCountKind.touched_window
        else item
        for item in observations
    ]

    report = reconcile_parameter_accounting_events(
        base,
        [_runtime_event(observations)],
        profile=ParameterAccountingProfile.training_runtime,
        now=_fixed_now,
    )

    assert any(item.reason_code == "updated-exceeds-touched" for item in report.conflicts)


def test_estimated_or_partial_counts_do_not_trigger_exact_algebra(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    observations = _runtime_observations(base.model_ref)
    observations = [
        _observation(
            ParameterCountKind.updated_window,
            99,
            base.model_ref,
            coverage=ParameterObservationCoverage.partial,
            relation=ParameterValueRelation.estimate,
        )
        if item.kind == ParameterCountKind.updated_window
        else item
        for item in observations
    ]

    report = reconcile_parameter_accounting_events(
        base,
        [_runtime_event(observations)],
        profile=ParameterAccountingProfile.training_runtime,
        now=_fixed_now,
    )

    assert report.status == ParameterAccountingStatus.incomplete
    assert report.conflicts == []
    assert any(gap.kind == ParameterCountKind.updated_window for gap in report.gaps)


def test_allocator_bytes_do_not_become_resident_parameter_coordinates(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    event = RunEvent(
        event_type="metric",
        run_id="run-1",
        seq=0,
        emitted_at=NOW_TEXT,
        metrics=EventMetrics(
            memory=MemoryMetrics(
                torch_allocated_bytes=10_000,
                dedicated_gpu_bytes=20_000,
            )
        ),
    )

    report = reconcile_parameter_accounting_events(
        base,
        [event],
        profile=ParameterAccountingProfile.inference_runtime,
        now=_fixed_now,
    )

    assert all(item.kind != ParameterCountKind.resident for item in report.observations)
    assert any(
        gap.kind == ParameterCountKind.resident
        and gap.reason == ParameterGapReason.missing_observation
        for gap in report.gaps
    )


def test_header_mismatch_remains_stored_element_evidence(tmp_path: Path):
    model, root = _model_snapshot(tmp_path, declared_count=7)

    report = build_model_parameter_accounting(model, snapshot_root=root, now=_fixed_now)
    header = next(
        item
        for item in report.observations
        if item.source.kind == ParameterEvidenceSourceKind.safetensors_header
    )

    assert header.value == 6
    assert header.value_relation == ParameterValueRelation.estimate
    assert header.identity_basis == ParameterIdentityBasis.stored_tensor_elements
    assert any(gap.gap_id == "declared-header-count-disagreement" for gap in report.gaps)


def test_estimated_descriptor_count_cannot_promote_header_elements_to_exact(tmp_path: Path):
    model, root = _model_snapshot(tmp_path)
    count = model.parameters.counts[0].model_copy(update={"evidence": EvidenceKind.estimated})
    estimated_model = model.model_copy(
        update={"parameters": model.parameters.model_copy(update={"counts": [count]})}
    )

    report = build_model_parameter_accounting(
        estimated_model,
        snapshot_root=root,
        now=_fixed_now,
    )

    header = next(
        item
        for item in report.observations
        if item.source.kind == ParameterEvidenceSourceKind.safetensors_header
    )
    assert header.value_relation == ParameterValueRelation.estimate
    assert report.status == ParameterAccountingStatus.incomplete


def test_malformed_safetensors_is_a_bounded_gap(tmp_path: Path):
    model, root = _model_snapshot(tmp_path, valid_safetensors=False)

    report = build_model_parameter_accounting(model, snapshot_root=root, now=_fixed_now)

    assert any(gap.reason == ParameterGapReason.malformed_evidence for gap in report.gaps)
    assert verify_parameter_accounting_hash(report)


def test_snapshot_files_must_match_the_descriptor_inventory(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _write_safetensors(
        replacement / "model.safetensors",
        {"different": ("F32", [3, 3])},
    )

    report = build_model_parameter_accounting(
        model,
        snapshot_root=replacement,
        now=_fixed_now,
    )

    assert any(gap.reason == ParameterGapReason.changed_during_read for gap in report.gaps)
    assert all(
        item.source.kind != ParameterEvidenceSourceKind.safetensors_header
        for item in report.observations
    )


def test_duplicate_keys_inside_one_safetensors_header_are_rejected(tmp_path: Path):
    root = tmp_path / "duplicate-header"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_parameters": 1}),
        encoding="utf-8",
    )
    header = (
        b'{"weight":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},'
        b'"weight":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}'
    )
    header += b" " * (-len(header) % 8)
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0"
    )
    inspected = inspect_model(root, model_id="duplicate-header", hash_weights=True, now=_fixed_now)
    count = inspected.parameters.counts[0].model_copy(
        update={"handling": _resolved_handling()}
    )
    model = inspected.model_copy(
        update={"parameters": inspected.parameters.model_copy(update={"counts": [count]})}
    )

    report = build_model_parameter_accounting(model, snapshot_root=root, now=_fixed_now)

    assert any(gap.reason == ParameterGapReason.malformed_evidence for gap in report.gaps)


@pytest.mark.parametrize(
    ("offsets", "data_bytes"),
    [
        ([1, 2], b"\0\0"),
        ([0, 1], b"\0\0"),
    ],
)
def test_adapter_safetensors_requires_complete_contiguous_payload(
    tmp_path: Path, offsets: list[int], data_bytes: bytes
):
    path = tmp_path / "adapter_model.safetensors"
    header = json.dumps(
        {"adapter.lora_A.weight": {"dtype": "I8", "shape": [1], "data_offsets": offsets}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data_bytes)

    with pytest.raises(ParameterAccountingError, match="cover|non-contiguous"):
        validate_safetensors_tensor_file(path)


def test_adapter_safetensors_rejects_interior_gap_and_unaligned_header(tmp_path: Path):
    path = tmp_path / "adapter_model.safetensors"
    header = json.dumps(
        {
            "a": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
            "b": {"dtype": "I8", "shape": [1], "data_offsets": [2, 3]},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0")
    with pytest.raises(ParameterAccountingError, match="non-contiguous"):
        validate_safetensors_tensor_file(path)

    unaligned = b'{"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}'
    while len(unaligned) % 8 == 0:
        unaligned += b" "
    path.write_bytes(struct.pack("<Q", len(unaligned)) + unaligned + b"\0")
    with pytest.raises(ParameterAccountingError, match="unaligned"):
        validate_safetensors_tensor_file(path)


def test_adapter_safetensors_detects_atomic_path_replacement_during_read(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "adapter_model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _write_safetensors(path, {"adapter.lora_A.weight": ("F32", [1])})
    _write_safetensors(replacement, {"adapter.lora_A.weight": ("F32", [1])})
    original_decode = accounting._decode_safetensors_header
    replaced = False

    def _replace_after_decode(*args, **kwargs):
        nonlocal replaced
        result = original_decode(*args, **kwargs)
        if not replaced:
            replacement.replace(path)
            replaced = True
        return result

    monkeypatch.setattr(accounting, "_decode_safetensors_header", _replace_after_decode)
    with pytest.raises(ParameterAccountingError, match="changed while"):
        validate_safetensors_tensor_file(path)


def test_duplicate_tensor_names_across_shards_are_not_double_counted_as_exact(tmp_path: Path):
    root = tmp_path / "sharded"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_parameters": 2}),
        encoding="utf-8",
    )
    _write_safetensors(root / "model-1.safetensors", {"weight": ("F16", [1])})
    _write_safetensors(root / "model-2.safetensors", {"weight": ("F16", [1])})
    inspected = inspect_model(root, model_id="sharded", hash_weights=True, now=_fixed_now)
    count = inspected.parameters.counts[0].model_copy(
        update={"handling": _resolved_handling()}
    )
    model = inspected.model_copy(
        update={"parameters": inspected.parameters.model_copy(update={"counts": [count]})}
    )

    report = build_model_parameter_accounting(model, snapshot_root=root, now=_fixed_now)

    assert any(gap.gap_id == "duplicate-safetensors-tensor-names" for gap in report.gaps)
    header = next(
        item
        for item in report.observations
        if item.source.kind == ParameterEvidenceSourceKind.safetensors_header
    )
    assert header.value_relation == ParameterValueRelation.estimate


def test_content_pinned_shard_index_can_prove_a_complete_header_inventory(tmp_path: Path):
    root = tmp_path / "indexed-shards"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_parameters": 2}),
        encoding="utf-8",
    )
    _write_safetensors(root / "model-1.safetensors", {"a": ("I8", [1])})
    _write_safetensors(root / "model-2.safetensors", {"b": ("I8", [1])})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 2},
                "weight_map": {
                    "a": "model-1.safetensors",
                    "b": "model-2.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    inspected = inspect_model(root, model_id="indexed", hash_weights=True, now=_fixed_now)
    count = inspected.parameters.counts[0].model_copy(
        update={"handling": _resolved_handling()}
    )
    model = inspected.model_copy(
        update={"parameters": inspected.parameters.model_copy(update={"counts": [count]})}
    )

    report = build_model_parameter_accounting(model, snapshot_root=root, now=_fixed_now)

    header = next(
        item
        for item in report.observations
        if item.source.kind == ParameterEvidenceSourceKind.safetensors_header
    )
    assert header.value == 2
    assert header.value_relation == ParameterValueRelation.exact
    assert report.status == ParameterAccountingStatus.complete


def test_shard_index_missing_a_recorded_shard_prevents_exact_inventory(tmp_path: Path):
    root = tmp_path / "missing-indexed-shard"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_parameters": 1}),
        encoding="utf-8",
    )
    _write_safetensors(root / "model-1.safetensors", {"a": ("I8", [1])})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-1.safetensors",
                    "b": "missing-model-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    inspected = inspect_model(root, model_id="missing-shard", hash_weights=True, now=_fixed_now)
    count = inspected.parameters.counts[0].model_copy(
        update={"handling": _resolved_handling()}
    )
    model = inspected.model_copy(
        update={"parameters": inspected.parameters.model_copy(update={"counts": [count]})}
    )

    report = build_model_parameter_accounting(model, snapshot_root=root, now=_fixed_now)

    assert report.status == ParameterAccountingStatus.incomplete
    assert any(gap.reason == ParameterGapReason.incomplete_inventory for gap in report.gaps)
    header = next(
        item
        for item in report.observations
        if item.source.kind == ParameterEvidenceSourceKind.safetensors_header
    )
    assert header.value_relation == ParameterValueRelation.estimate


def test_event_loader_is_bounded_and_rejects_mixed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    event_path = tmp_path / "events.jsonl"
    event_path.write_text(
        "\n".join(
            [
                RunEvent(
                    event_type="heartbeat",
                    run_id="run-1",
                    seq=0,
                    emitted_at=NOW_TEXT,
                ).model_dump_json(),
                RunEvent(
                    event_type="heartbeat",
                    run_id="run-2",
                    seq=1,
                    emitted_at=NOW_TEXT,
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(accounting, "_MAX_EVENTS", 1)

    with pytest.raises(ParameterAccountingError, match="event limit"):
        load_parameter_events(event_path)


def test_event_reconciliation_rejects_unsorted_or_mixed_streams(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    mixed = [
        RunEvent(event_type="heartbeat", run_id="a", seq=0, emitted_at=NOW_TEXT),
        RunEvent(event_type="heartbeat", run_id="b", seq=1, emitted_at=NOW_TEXT),
    ]

    with pytest.raises(ParameterAccountingError, match="more than one run_id"):
        reconcile_parameter_accounting_events(
            base,
            mixed,
            profile=ParameterAccountingProfile.training_runtime,
            now=_fixed_now,
        )


def test_event_reconciliation_rejects_invalid_parent_profile_and_order(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    base = build_model_parameter_accounting(model, now=_fixed_now)
    event_0 = RunEvent(
        event_type="heartbeat",
        run_id="run-1",
        seq=0,
        emitted_at=NOW_TEXT,
    )
    event_1 = event_0.model_copy(update={"seq": 1})

    with pytest.raises(ParameterAccountingError, match="hash mismatch"):
        reconcile_parameter_accounting_events(
            base.model_copy(update={"generated_at": "tampered"}),
            [event_0],
            profile=ParameterAccountingProfile.training_runtime,
            now=_fixed_now,
        )
    with pytest.raises(ParameterAccountingError, match="runtime/checkpoint/evaluation"):
        reconcile_parameter_accounting_events(
            base,
            [event_0],
            profile=ParameterAccountingProfile.model_static,
            now=_fixed_now,
        )
    with pytest.raises(ParameterAccountingError, match="at least one"):
        reconcile_parameter_accounting_events(
            base,
            [],
            profile=ParameterAccountingProfile.training_runtime,
            now=_fixed_now,
        )
    with pytest.raises(ParameterAccountingError, match="sorted by unique"):
        reconcile_parameter_accounting_events(
            base,
            [event_1, event_0],
            profile=ParameterAccountingProfile.training_runtime,
            now=_fixed_now,
        )


def test_event_loader_skips_blanks_and_reports_invalid_or_missing_input(tmp_path: Path):
    valid_path = tmp_path / "valid-events.jsonl"
    event = RunEvent(
        event_type="heartbeat",
        run_id="run-1",
        seq=0,
        emitted_at=NOW_TEXT,
    )
    valid_path.write_text(f"\n{event.model_dump_json()}\n", encoding="utf-8")

    assert load_parameter_events(valid_path) == [event]

    invalid_path = tmp_path / "invalid-events.jsonl"
    invalid_path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ParameterAccountingError, match="invalid RunEvent JSONL"):
        load_parameter_events(invalid_path)
    with pytest.raises(ParameterAccountingError, match="cannot read"):
        load_parameter_events(tmp_path / "missing.jsonl")


def test_report_writer_rejects_tampered_report(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    report = build_model_parameter_accounting(model, now=_fixed_now)
    tampered = report.model_copy(update={"generated_at": "changed"})

    with pytest.raises(ParameterAccountingError, match="invalid hash"):
        write_parameter_accounting_report(tampered, tmp_path / "report.json")

    invalid = report.model_copy(update={"status": ParameterAccountingStatus.complete})
    resealed = invalid.model_copy(
        update={"report_hash": parameter_accounting_hash_for(invalid)}
    )
    with pytest.raises(ParameterAccountingError, match="structurally invalid"):
        write_parameter_accounting_report(resealed, tmp_path / "resealed-invalid.json")


def test_model_inspection_bundle_can_emit_parameter_accounting(tmp_path: Path):
    model, root = _model_snapshot(tmp_path)
    del model
    bundle = inspect_model_bundle(
        root,
        model_id="bundle-model",
        hash_weights=True,
        parameter_accounting=True,
        now=_fixed_now,
    )

    assert bundle.parameter_accounting is not None
    outputs = write_inspection_bundle(bundle, tmp_path / "out")
    assert any(path.name == "bundle-model.parameter-accounting.json" for path in outputs)


def test_unhashed_inspection_bundle_reports_gaps_without_scope_collision(tmp_path: Path):
    _, root = _model_snapshot(tmp_path)

    bundle = inspect_model_bundle(
        root,
        model_id="unhashed-bundle",
        parameter_accounting=True,
        now=_fixed_now,
    )

    assert bundle.parameter_accounting is not None
    assert bundle.parameter_accounting.status == ParameterAccountingStatus.incomplete
    assert any(
        gap.reason in {ParameterGapReason.unpinned_model, ParameterGapReason.unknown_handling}
        for gap in bundle.parameter_accounting.gaps
    )


def test_bundle_writer_refuses_a_tampered_parameter_report(tmp_path: Path):
    _, root = _model_snapshot(tmp_path)
    bundle = inspect_model_bundle(
        root,
        model_id="bundle-model",
        hash_weights=True,
        parameter_accounting=True,
        now=_fixed_now,
    )
    assert bundle.parameter_accounting is not None
    tampered = bundle.parameter_accounting.model_copy(update={"generated_at": "tampered"})
    unsafe_bundle = bundle.model_copy(update={"parameter_accounting": tampered})

    with pytest.raises(ModelInspectionError, match="invalid hash"):
        write_inspection_bundle(unsafe_bundle, tmp_path / "tampered-out")


def test_bundle_contract_rejects_parameter_evidence_for_another_model(tmp_path: Path):
    model, _ = _model_snapshot(tmp_path)
    foreign_model, _ = _model_snapshot(tmp_path, model_id="another-model")
    foreign = build_model_parameter_accounting(foreign_model, now=_fixed_now)

    with pytest.raises(ValidationError, match="bundled model revision"):
        ModelInspectionBundle(model=model, parameter_accounting=foreign)


def test_parameter_account_cli_reads_descriptor_and_writes_report(tmp_path: Path):
    model, root = _model_snapshot(tmp_path)
    descriptor_path = tmp_path / "model.json"
    descriptor_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    output_path = tmp_path / "accounting.json"

    result = runner.invoke(
        app,
        [
            "parameter-account",
            str(descriptor_path),
            "--snapshot",
            str(root),
            "--out",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert "written_file" not in payload
    assert output_path.exists()


def test_parameter_account_cli_keeps_static_parent_and_runtime_report_ids_distinct(
    tmp_path: Path,
):
    model, _ = _model_snapshot(tmp_path)
    descriptor_path = tmp_path / "model.json"
    descriptor_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    base = build_model_parameter_accounting(model, now=_fixed_now)
    event_path = tmp_path / "events.jsonl"
    event_path.write_text(
        _runtime_event(_runtime_observations(base.model_ref)).model_dump_json() + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "parameter-account",
            str(descriptor_path),
            "--events",
            str(event_path),
            "--report-id",
            "final-runtime-report",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["report_id"] == "final-runtime-report"
    assert payload["parent_report_refs"][0]["id"] == "tiny-model-parameter-accounting"


def test_model_inspect_cli_parameter_accounting_flag(tmp_path: Path):
    _, root = _model_snapshot(tmp_path)
    out = tmp_path / "cli-out"

    result = runner.invoke(
        app,
        [
            "model-inspect",
            str(root),
            "--model-id",
            "cli-model",
            "--hash-weights",
            "--parameter-accounting",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Parameter accounting:" in result.stdout
    assert (out / "cli-model.parameter-accounting.json").exists()


def test_contract_schemas_include_parameter_evidence_in_lifecycle_records():
    from corpus_studio.platform import contract_schemas

    schemas = contract_schemas()
    assert "ParameterAccountingReport" in schemas
    assert "parameter_accounting_ref" in schemas["RunPlan"]["properties"]
    assert "parameter_accounting_refs" in schemas["RunManifest"]["properties"]
    assert "parameter_accounting_ref" in schemas["ArtifactManifest"]["properties"]
    assert "parameter_accounting_ref" in schemas["EvaluationResult"]["properties"]
    event_metrics = schemas["RunEvent"]["$defs"]["EventMetrics"]
    assert "parameter_observations" in event_metrics["properties"]
