"""Versioned TraceRecord construction, hashing, migration, review, and training gates.

This module is control-plane only: stdlib + pydantic contracts, no model framework imports. Structured
records preserve context, provenance, review, and reasoning/tool boundaries. Legacy ``<think>`` rows
remain readable through explicit adapters, but generated records are never training-ready until a
reviewed, hash-sealed successor is written.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal
from uuid import uuid4

from corpus_studio import __version__
from corpus_studio.exporters.cleaning import exact_row_signature
from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    TraceMessage,
    TracePolicyDecision,
    TraceProducer,
    TraceRecord,
    TraceReview,
    TraceSegment,
    TraceSource,
    TraceValidationEvidence,
    TraceValidationFinding,
)

TRACE_VALIDATOR = "corpus_studio.trace_quality"
TRACE_VALIDATOR_VERSION = "1.0.0"
TRACE_QUALITY_CONFIG: dict[str, int | float | str] = {
    "answer_leak_algorithm": "unicode_nfkc_casefold_whitespace_exact_v1",
    "answer_leak_min_chars": 40,
    "min_reasoning_chars": 24,
    "min_reasoning_answer_ratio": 0.15,
}


class TraceRecordError(ValueError):
    """A TraceRecord is malformed, unsealed, unsafe to write, or not training-ready."""


@dataclass(frozen=True)
class TraceDatasetTrainingCheck:
    record_rows: int
    legacy_rows: int
    blocked: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blocked


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def trace_record_hash_for(record: TraceRecord) -> str:
    """Canonical SHA-256 seal over the fully defaulted record, excluding only its own seal."""

    return canonical_sha256(record.model_dump(mode="json", exclude={"trace_hash"}))


def verify_trace_record_hash(record: TraceRecord) -> bool:
    if record.trace_hash != trace_record_hash_for(record):
        return False
    producer = record.producer
    if producer.policy_decision is not None and producer.policy_snapshot is not None:
        if canonical_sha256(producer.policy_snapshot) != producer.policy_decision.policy_sha256:
            return False
    for segment in record.segments:
        result = segment.tool_result
        if (
            result is not None
            and result.content is not None
            and result.content_ref is None
            and not result.truncated
            and hashlib.sha256(result.content.encode("utf-8")).hexdigest() != result.content_sha256
        ):
            return False
    return True


def seal_trace_record(record: TraceRecord) -> TraceRecord:
    sealed = record.model_copy(update={"trace_hash": trace_record_hash_for(record)})
    validated = TraceRecord.model_validate(sealed.model_dump(mode="json"))
    if not verify_trace_record_hash(validated):
        raise TraceRecordError("TraceRecord contains inconsistent embedded evidence hashes")
    return validated


def is_trace_record_row(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    return (
        "trace_hash" in row
        or ("trace_id" in row and "segments" in row)
        or (row.get("contract_version") == "1.0.0" and "producer" in row and "segments" in row)
    )


def parse_trace_record(row: dict[str, Any]) -> TraceRecord:
    try:
        record = TraceRecord.model_validate(row)
    except (TypeError, ValueError, RecursionError) as exc:
        raise TraceRecordError(f"invalid TraceRecord: {exc}") from exc
    if not verify_trace_record_hash(record):
        raise TraceRecordError(f"TraceRecord hash mismatch: {record.trace_id}")
    return record


def source_row_id(row: dict[str, Any]) -> str:
    return hashlib.sha256(exact_row_signature(row).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_trace_source(
    *,
    artifact_ref: str,
    artifact_sha256: str,
    row: dict[str, Any],
    row_index: int,
) -> TraceSource:
    return TraceSource(
        artifact_ref=artifact_ref,
        artifact_sha256=artifact_sha256,
        source_row_id=source_row_id(row),
        source_row_index=row_index,
    )


def _trace_messages(messages: Sequence[dict[str, Any]]) -> list[TraceMessage]:
    result: list[TraceMessage] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise TraceRecordError(f"unsupported trace context role at message {index}: {role!r}")
        kwargs: dict[str, Any] = {
            "message_id": f"context-{index:04d}",
            "role": role,
            "content": content,
        }
        if message.get("name"):
            kwargs["name"] = str(message["name"])
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                raise TraceRecordError("legacy tool context requires tool_call_id")
            kwargs["tool_call_id"] = str(tool_call_id)
        result.append(TraceMessage(**kwargs))
    if not result:
        raise TraceRecordError("trace context cannot be empty")
    return result


def context_messages_from_trace(trace: Any) -> list[TraceMessage]:
    if trace.messages:
        return _trace_messages(trace.messages)
    prompt = str(trace.prompt or "").strip()
    if not prompt:
        raise TraceRecordError("trace requires prompt or messages")
    return [TraceMessage(message_id="context-0000", role="user", content=prompt)]


def _validation_evidence(trace: Any, checked_at: str) -> TraceValidationEvidence:
    from corpus_studio.training.traces import trace_quality, validate_trace  # noqa: PLC0415

    structural = validate_trace(trace)
    quality = trace_quality(trace)
    findings: list[TraceValidationFinding] = []
    error_codes = {
        "missing answer": "missing_final_answer",
        "missing prompt/messages": "missing_context",
    }
    for error in structural.errors:
        code = error_codes.get(error, "structural_trace_error")
        findings.append(
            TraceValidationFinding(
                code=code,
                severity="block",
                location="record",
                message=error.replace("—", "-"),
            )
        )
    findings.extend(
        TraceValidationFinding(
            code=item.code,
            severity="block" if item.severity == "block" else "warning",
            location=item.location,
            message=item.message.replace("—", "-"),
        )
        for item in quality.findings
    )
    findings = sorted(findings, key=lambda item: (item.code, item.location, item.message))
    severities = {item.severity for item in findings}
    status: Literal["pass", "warn", "block"] = (
        "block" if "block" in severities else "warn" if "warning" in severities else "pass"
    )
    return TraceValidationEvidence(
        validator=TRACE_VALIDATOR,
        validator_version=TRACE_VALIDATOR_VERSION,
        config_sha256=canonical_sha256(TRACE_QUALITY_CONFIG),
        checked_at=checked_at,
        status=status,
        findings=findings,
    )


def imported_trace_producer() -> TraceProducer:
    return TraceProducer(kind="imported", tool="corpus-studio trace-migrate", tool_version=__version__)


def model_trace_producer(
    *,
    backend: str,
    provider_id: str,
    provider_kind: str,
    requested_model_id: str,
    model_id: str,
    route_id: str | None,
    prompt_template_version: str,
    prompt_template: str,
    request: object,
    response_sha256: str,
    response_metadata: dict[str, Any],
    decoding: dict[str, Any],
    seed: int | None,
    policy_snapshot: dict[str, Any],
    policy_source: str,
    captured_at: str,
) -> TraceProducer:
    return TraceProducer(
        kind="model",
        tool="corpus-studio trace-generate",
        tool_version=__version__,
        backend=backend,
        provider_id=provider_id,
        provider_kind=provider_kind,
        requested_model_id=requested_model_id,
        model_id=model_id,
        route_id=route_id,
        prompt_template_version=prompt_template_version,
        prompt_template_sha256=text_sha256(prompt_template),
        request_sha256=canonical_sha256(request),
        response_sha256=response_sha256,
        response_metadata=response_metadata,
        decoding=decoding,
        seed=seed,
        policy_decision=TracePolicyDecision(
            allowed=True,
            policy_source=policy_source,
            policy_sha256=canonical_sha256(policy_snapshot),
            human_review_required=True,
            captured_at=captured_at,
        ),
        policy_snapshot=policy_snapshot,
    )


def build_reasoning_trace_record(
    *,
    trace: Any,
    source: TraceSource,
    producer: TraceProducer,
    created_at: str | None = None,
    trace_id: str | None = None,
    tags: Iterable[str] = (),
    notes: Iterable[str] = (),
) -> TraceRecord:
    """Build and seal one complete reasoning record from the dependency-light legacy view."""

    from corpus_studio.training.traces import validate_trace  # noqa: PLC0415

    structural = validate_trace(trace)
    if not structural.valid:
        raise TraceRecordError("cannot build TraceRecord: " + "; ".join(structural.errors))
    stamp = created_at or utc_now_iso()
    context = context_messages_from_trace(trace)
    segment_origin: Literal["model", "human", "imported"] = (
        "model" if producer.kind == "model" else "human" if producer.kind == "human" else "imported"
    )
    segments: list[TraceSegment] = []
    if str(trace.thinking or "").strip():
        segments.append(
            TraceSegment(
                segment_id=f"segment-{len(segments):04d}",
                sequence=len(segments),
                kind="reasoning",
                actor="assistant",
                origin=segment_origin,
                verification="unverified",
                content=str(trace.thinking).strip(),
            )
        )
    segments.append(
        TraceSegment(
            segment_id=f"segment-{len(segments):04d}",
            sequence=len(segments),
            kind="final_answer",
            actor="assistant",
            origin=segment_origin,
            verification="unverified",
            content=str(trace.answer).strip(),
        )
    )
    record = TraceRecord(
        trace_id=trace_id or f"trace-{uuid4().hex}",
        trace_hash="0" * 64,
        created_at=stamp,
        trace_kind="reasoning",
        source=source,
        context=context,
        segments=segments,
        producer=producer,
        validation=_validation_evidence(trace, stamp),
        review=TraceReview(status="pending"),
        tags=sorted(set(tags)),
        notes=sorted(set(notes)),
    )
    return seal_trace_record(record)


def trace_record_from_legacy_row(
    row: dict[str, Any],
    *,
    source: TraceSource,
    created_at: str | None = None,
    trace_id: str | None = None,
) -> TraceRecord:
    if is_trace_record_row(row):
        raise TraceRecordError("row is already a TraceRecord")
    from corpus_studio.training.traces import trace_from_row  # noqa: PLC0415

    raw_tags = row.get("tags")
    tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
    return build_reasoning_trace_record(
        trace=trace_from_row(row),
        source=source,
        producer=imported_trace_producer(),
        created_at=created_at,
        trace_id=trace_id,
        tags=[str(item) for item in tags],
        notes=["Migrated from an unsealed legacy trace row; missing evidence was not invented."],
    )


def legacy_trace_from_record(record: TraceRecord) -> Any:
    """Project a verified structured record into the existing trainer/evaluator Trace view."""

    if not verify_trace_record_hash(record):
        raise TraceRecordError(f"TraceRecord hash mismatch: {record.trace_id}")
    from corpus_studio.training.traces import Trace  # noqa: PLC0415

    messages: list[dict[str, Any]] = []
    for message in record.context:
        value: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name is not None:
            value["name"] = message.name
        if message.tool_call_id is not None:
            value["tool_call_id"] = message.tool_call_id
        messages.append(value)
    thinking = "\n\n".join(
        segment.content or "" for segment in record.segments if segment.kind == "reasoning"
    ).strip()
    final = next(segment for segment in record.segments if segment.kind == "final_answer")
    return Trace(messages=messages, thinking=thinking, answer=final.content or "")


def review_trace_record(
    record: TraceRecord,
    *,
    decision: Literal["approved", "rejected"],
    reviewer: str,
    reviewed_at: str | None = None,
    notes: Iterable[str] = (),
    provider_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> TraceRecord:
    """Create a hash-sealed review successor that pins the previous record hash."""

    if not verify_trace_record_hash(record):
        raise TraceRecordError(f"refusing to review an invalid TraceRecord hash: {record.trace_id}")
    stamp = reviewed_at or utc_now_iso()
    validation = record.validation
    if decision == "approved":
        validation = _validation_evidence(legacy_trace_from_record(record), stamp)
        if validation.status == "block":
            raise TraceRecordError(
                "blocked trace validation cannot be approved; repair and revalidate it"
            )
        policy_issues = _model_policy_evidence_issues(record, provider_overrides)
        if policy_issues:
            raise TraceRecordError(
                "model trace policy evidence cannot be approved: " + "; ".join(policy_issues)
            )
    previous = Ref(id=record.trace_id, hash=HashRef(algo="sha256", value=record.trace_hash))
    parent_refs = sorted(
        [*record.parent_trace_refs, previous],
        key=lambda item: (item.id, item.hash.value if item.hash and item.hash.value else ""),
    )
    successor = record.model_copy(
        update={
            "trace_hash": "0" * 64,
            "review": TraceReview(
                status=decision,
                reviewer=reviewer.strip(),
                reviewed_at=stamp,
                notes=sorted(set(notes)),
            ),
            "validation": validation,
            "parent_trace_refs": parent_refs,
        }
    )
    return seal_trace_record(TraceRecord.model_validate(successor.model_dump(mode="json")))


def _model_policy_evidence_issues(
    record: TraceRecord,
    provider_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    if record.producer.kind != "model":
        return []

    from corpus_studio.providers.policy import (  # noqa: PLC0415
        ProviderPolicy,
        is_frontier_generation_source,
        resolve_policy,
    )

    producer = record.producer
    issues: list[str] = []
    try:
        policy = ProviderPolicy.model_validate(producer.policy_snapshot)
    except (TypeError, ValueError) as exc:
        return [f"invalid model policy snapshot: {exc}"]
    expected = {
        "provider_id": producer.provider_id,
        "provider_kind": producer.provider_kind,
        "model_id": producer.model_id,
        "route_id": producer.route_id,
    }
    actual = {
        "provider_id": policy.provider_id,
        "provider_kind": policy.provider_kind,
        "model_id": policy.model_id,
        "route_id": policy.route_id,
    }
    if actual != expected:
        issues.append("model policy snapshot identity does not match producer identity")
    if not policy.can_generate_trainable():
        issues.append("model policy snapshot does not authorize trainable generation")

    assert producer.provider_id is not None
    assert producer.requested_model_id is not None
    assert producer.model_id is not None
    requested_route_id = (
        producer.requested_model_id if producer.provider_id == "openrouter" else None
    )
    requested_policy = resolve_policy(
        producer.provider_id,
        model_id=producer.requested_model_id,
        route_id=requested_route_id,
        overrides=provider_overrides,
    )
    authoritative_policy = resolve_policy(
        producer.provider_id,
        model_id=producer.model_id,
        route_id=producer.route_id,
        overrides=provider_overrides,
    )
    if not requested_policy.can_generate_trainable():
        issues.append("external provider authority does not approve the requested model")
    if not authoritative_policy.can_generate_trainable():
        issues.append("external provider authority does not approve the resolved model")
    if canonical_sha256(authoritative_policy.model_dump(mode="json")) != canonical_sha256(
        producer.policy_snapshot
    ):
        issues.append("stored model policy snapshot does not match current external authority")
    if is_frontier_generation_source(producer.provider_id or "", producer.route_id):
        issues.append("frontier providers/routes cannot supply trainable trace data")
    return issues


def _trainer_segment_issues(record: TraceRecord) -> list[str]:
    issues: list[str] = []
    expected_origin = {
        "model": "model",
        "human": "human",
        "imported": "imported",
    }.get(record.producer.kind)
    if expected_origin is None:
        issues.append(f"the current trace trainer does not support {record.producer.kind} producers")

    for segment in record.segments:
        if segment.kind not in {"reasoning", "final_answer"}:
            continue
        label = f"segment {segment.segment_id} ({segment.kind})"
        if segment.actor != "assistant":
            issues.append(f"{label} must be authored by the assistant")
        if segment.origin not in {"model", "human", "imported"}:
            issues.append(f"{label} has unsupported origin {segment.origin}")
        elif expected_origin is not None and segment.origin != expected_origin:
            issues.append(
                f"{label} origin {segment.origin} does not match producer kind {record.producer.kind}"
            )
        if segment.verification == "rejected":
            issues.append(f"{label} is rejected and cannot be a training target")
        signal = segment.training_signal
        if signal is not None:
            if not signal.target:
                issues.append(f"{label} is explicitly marked training_signal.target=false")
            if (
                signal.label is not None
                or signal.reward is not None
                or signal.weight != 1.0
                or signal.verifier_ref is not None
            ):
                issues.append(f"{label} uses training-signal semantics the current SFT trainer ignores")
    return issues


def trace_validation_evidence_issues(record: TraceRecord) -> list[str]:
    """Return a blocker when stored validator evidence is not exactly reproducible now."""

    current = _validation_evidence(legacy_trace_from_record(record), record.validation.checked_at)
    if record.validation.model_dump(mode="json") != current.model_dump(mode="json"):
        return ["trace validation evidence is stale, foreign, or inconsistent"]
    return []


def trace_record_training_issues(
    record: TraceRecord,
    provider_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    issues: list[str] = []
    if not verify_trace_record_hash(record):
        issues.append("trace hash mismatch")
        return issues
    if record.review.status != "approved":
        issues.append(f"review status is {record.review.status}, not approved")
    current_validation = _validation_evidence(legacy_trace_from_record(record), record.validation.checked_at)
    issues.extend(trace_validation_evidence_issues(record))
    if current_validation.status == "block":
        issues.append("trace validation is blocked")
    issues.extend(_model_policy_evidence_issues(record, provider_overrides))
    if record.trace_kind != "reasoning":
        issues.append(f"the current trace trainer does not implement {record.trace_kind} records")
    unsupported = [
        segment.kind
        for segment in record.segments
        if segment.kind not in {"reasoning", "final_answer"}
    ]
    if unsupported:
        issues.append("the current trace trainer cannot render segments: " + ", ".join(unsupported))
    if any(segment.content is None for segment in record.segments):
        issues.append("the current trace trainer requires inline segment content")
    issues.extend(_trainer_segment_issues(record))
    return issues


def check_trace_dataset_for_training(
    rows: Sequence[dict[str, Any]],
    *,
    provider_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> TraceDatasetTrainingCheck:
    from corpus_studio.training.traces import trace_from_row, trace_quality, validate_trace  # noqa: PLC0415

    record_rows = 0
    legacy_rows = 0
    blocked: list[str] = []
    for index, row in enumerate(rows, start=1):
        if is_trace_record_row(row):
            record_rows += 1
            try:
                record = parse_trace_record(row)
            except TraceRecordError as exc:
                blocked.append(f"row {index}: {exc}")
                continue
            blocked.extend(
                f"row {index}: {issue}"
                for issue in trace_record_training_issues(record, provider_overrides)
            )
            continue
        legacy_rows += 1
        metadata = row.get("meta")
        if isinstance(metadata, dict) and metadata.get("trace_record_hash") is not None:
            blocked.append(
                f"row {index}: generated legacy compatibility rows cannot be trained; "
                "use the referenced TraceRecord review workflow"
            )
            continue
        try:
            trace = trace_from_row(row)
            structural = validate_trace(trace)
            quality = trace_quality(trace)
        except ValueError as exc:
            blocked.append(f"row {index}: {exc}")
            continue
        blocked.extend(f"row {index}: {error}" for error in structural.errors)
        if quality.status == "fail":
            blocked.extend(f"row {index}: {issue}" for issue in quality.issues)
    return TraceDatasetTrainingCheck(record_rows, legacy_rows, tuple(blocked))


def _refuse_examples_jsonl(path: Path) -> None:
    if path.name.casefold() == "examples.jsonl":
        raise TraceRecordError(
            "the engine never writes examples.jsonl; write a separate artifact and import it through the desktop"
        )


def write_trace_records(records: Iterable[TraceRecord], path: str | Path) -> Path:
    """Validate every seal, then atomically replace a non-project JSONL artifact."""

    output = Path(path)
    _refuse_examples_jsonl(output)
    materialized = list(records)
    for record in materialized:
        validated = TraceRecord.model_validate(record.model_dump(mode="json"))
        if not verify_trace_record_hash(validated):
            raise TraceRecordError(f"refusing to write invalid TraceRecord hash: {record.trace_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in materialized:
                handle.write(
                    json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def write_jsonl_artifact(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Atomically write a generic compatibility JSONL artifact, never project examples.jsonl."""

    output = Path(path)
    _refuse_examples_jsonl(output)
    materialized = list(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in materialized:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def write_json_atomic(payload: object, path: str | Path) -> Path:
    output = Path(path)
    _refuse_examples_jsonl(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output
