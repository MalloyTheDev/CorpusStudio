"""Versioned TraceRecord contracts, lineage, review, migration, and training gates."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

import corpus_studio.platform as P
from corpus_studio.platform.trace_records import (
    TRACE_QUALITY_CONFIG,
    TraceRecordError,
    artifact_trace_source,
    build_reasoning_trace_record,
    canonical_sha256,
    check_trace_dataset_for_training,
    imported_trace_producer,
    legacy_trace_from_record,
    model_trace_producer,
    parse_trace_record,
    review_trace_record,
    seal_trace_record,
    trace_record_from_legacy_row,
    trace_record_training_issues,
    verify_trace_record_hash,
    write_json_atomic,
    write_trace_records,
)
from corpus_studio.training.traces import Trace, trace_from_row
from corpus_studio.providers.policy import ProviderPolicy, ProviderRole


def _source() -> P.TraceSource:
    return P.TraceSource(
        artifact_ref="source.jsonl",
        artifact_sha256="a" * 64,
        source_row_id="b" * 64,
        source_row_index=1,
    )


def _record(*, answer: str = "391") -> P.TraceRecord:
    return build_reasoning_trace_record(
        trace=Trace(
            prompt="What is 17 multiplied by 23?",
            thinking="Break 23 into 20 plus 3, multiply each part, then add 340 and 51.",
            answer=answer,
        ),
        source=_source(),
        producer=imported_trace_producer(),
        created_at="2026-07-13T12:00:00+00:00",
        trace_id="trace-test",
        tags=["reasoning"],
    )


def _validation() -> P.TraceValidationEvidence:
    return P.TraceValidationEvidence(
        validator="corpus_studio.trace_quality",
        validator_version="1.0.0",
        config_sha256=canonical_sha256(TRACE_QUALITY_CONFIG),
        checked_at="2026-07-13T12:00:00+00:00",
        status="pass",
    )


def test_trace_record_roundtrip_hash_and_legacy_projection():
    record = _record()
    assert verify_trace_record_hash(record)
    assert parse_trace_record(record.model_dump(mode="json")) == record
    projected = legacy_trace_from_record(record)
    assert projected.answer == "391"
    assert "multiply each part" in projected.thinking
    assert projected.messages == [
        {"role": "user", "content": "What is 17 multiplied by 23?"}
    ]


def test_trace_from_row_recognizes_sealed_record_before_legacy_aliases():
    trace = trace_from_row(_record().model_dump(mode="json"))
    assert trace.answer == "391"
    assert trace.messages and trace.messages[0]["role"] == "user"


def test_trace_record_tampering_is_rejected():
    payload = _record().model_dump(mode="json")
    payload["segments"][-1]["content"] = "tampered"
    with pytest.raises(TraceRecordError, match="hash mismatch"):
        parse_trace_record(payload)


def test_trace_record_forbids_unknown_fields():
    payload = _record().model_dump(mode="json")
    payload["surprise"] = True
    with pytest.raises(TraceRecordError, match="extra"):
        parse_trace_record(payload)


def test_trace_source_requires_exactly_one_pinned_source():
    with pytest.raises(ValidationError):
        P.TraceSource(
            dataset_ref=P.Ref(id="dataset-without-hash"),
            source_row_id="a" * 64,
        )
    with pytest.raises(ValidationError):
        P.TraceSource(
            artifact_ref="rows.jsonl",
            source_row_id="a" * 64,
        )


def test_model_producer_seals_policy_and_requires_review():
    producer = model_trace_producer(
        backend="ollama",
        provider_id="ollama",
        provider_kind="local",
        requested_model_id="qwen:latest",
        model_id="qwen",
        route_id=None,
        prompt_template_version="reasoning-system-v1",
        prompt_template="Reason carefully.",
        request={"messages": [{"role": "user", "content": "Q"}]},
        response_sha256="c" * 64,
        response_metadata={"model": "qwen"},
        decoding={"temperature": 0.7},
        seed=None,
        policy_snapshot={"provider_id": "ollama", "generation_allowed": True},
        policy_source="user_override",
        captured_at="2026-07-13T12:00:00+00:00",
    )
    record = build_reasoning_trace_record(
        trace=Trace(prompt="Q", thinking="A sufficiently detailed reasoning process.", answer="A"),
        source=_source(),
        producer=producer,
        created_at="2026-07-13T12:00:00+00:00",
        trace_id="trace-model",
    )
    assert record.review.status == "pending"
    assert record.producer.policy_decision and record.producer.policy_decision.allowed
    assert record.segments[0].verification == "unverified"


def test_model_producer_cannot_record_denied_policy():
    with pytest.raises(ValidationError, match="allowed generation policy"):
        P.TraceProducer(
            kind="model",
            tool="generator",
            backend="ollama",
            provider_id="ollama",
            provider_kind="local",
            requested_model_id="qwen",
            model_id="qwen",
            prompt_template_sha256="a" * 64,
            request_sha256="b" * 64,
            response_sha256="c" * 64,
            policy_snapshot={"allowed": False},
            policy_decision=P.TracePolicyDecision(
                allowed=False,
                policy_source="builtin",
                policy_sha256="d" * 64,
                captured_at="2026-07-13T12:00:00+00:00",
            ),
        )


def test_generated_or_imported_reasoning_cannot_be_relabeled_ground_truth():
    payload = _record().model_dump(mode="json")
    payload["segments"][0]["verification"] = "human_verified"
    payload["trace_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="reasoning must remain explicitly unverified"):
        P.TraceRecord.model_validate(payload)


def test_tool_boundaries_pair_calls_and_results():
    record = P.TraceRecord(
        trace_id="trace-tool",
        trace_hash="0" * 64,
        created_at="2026-07-13T12:00:00+00:00",
        trace_kind="tool_use",
        source=_source(),
        context=[P.TraceMessage(message_id="context-0000", role="user", content="Compute 2+2")],
        segments=[
            P.TraceSegment(
                segment_id="segment-0000",
                sequence=0,
                kind="tool_call",
                actor="assistant",
                origin="model",
                tool_call=P.TraceToolCall(call_id="call-1", tool_name="calculator", arguments={"x": 2}),
            ),
            P.TraceSegment(
                segment_id="segment-0001",
                sequence=1,
                kind="tool_result",
                actor="tool",
                origin="tool",
                verification="tool_verified",
                tool_result=P.TraceToolResult(
                    call_id="call-1",
                    status="success",
                    content="4",
                    content_sha256=hashlib.sha256(b"4").hexdigest(),
                ),
            ),
            P.TraceSegment(
                segment_id="segment-0002",
                sequence=2,
                kind="final_answer",
                actor="assistant",
                origin="model",
                content="4",
            ),
        ],
        producer=imported_trace_producer(),
        validation=_validation(),
    )
    assert verify_trace_record_hash(seal_trace_record(record))

    bad = record.model_dump(mode="json")
    bad["segments"][1]["tool_result"]["call_id"] = "unknown"
    with pytest.raises(ValidationError, match="matching tool call"):
        P.TraceRecord.model_validate(bad)


def test_final_answer_must_be_unique_and_last():
    payload = _record().model_dump(mode="json")
    payload["segments"].append(
        {
            "segment_id": "segment-late",
            "sequence": len(payload["segments"]),
            "kind": "observation",
            "actor": "human",
            "origin": "human",
            "verification": "unverified",
            "content": "late",
            "evidence_refs": [],
        }
    )
    with pytest.raises(ValidationError, match="last segment"):
        P.TraceRecord.model_validate(payload)


def test_structured_segments_reject_think_markup():
    with pytest.raises(ValidationError, match="cannot contain <think>"):
        P.TraceSegment(
            segment_id="s",
            sequence=0,
            kind="reasoning",
            actor="assistant",
            origin="model",
            content="<think>hidden</think>",
        )


def test_review_creates_hash_pinned_successor_and_training_gate():
    pending = _record()
    assert "review status is pending, not approved" in trace_record_training_issues(pending)
    approved = review_trace_record(
        pending,
        decision="approved",
        reviewer="reviewer@example.test",
        reviewed_at="2026-07-13T13:00:00+00:00",
        notes=["Inspected reasoning and final answer."],
    )
    assert approved.trace_id == pending.trace_id
    assert approved.trace_hash != pending.trace_hash
    assert approved.parent_trace_refs[-1].hash
    assert approved.parent_trace_refs[-1].hash.value == pending.trace_hash
    assert trace_record_training_issues(approved) == []


def test_blocked_validation_cannot_be_approved():
    answer = "The Ostervaal Registry decides what counts as a legitimate self."
    row = {
        "prompt": f"Repeat this answer: {answer.upper()}",
        "thinking": "Consider the supplied statement and repeat it after checking the wording.",
        "answer": answer,
    }
    source = artifact_trace_source(
        artifact_ref="legacy.jsonl",
        artifact_sha256="d" * 64,
        row=row,
        row_index=1,
    )
    blocked = trace_record_from_legacy_row(
        row,
        source=source,
        created_at="2026-07-13T12:00:00+00:00",
        trace_id="trace-blocked",
    )
    assert blocked.validation.status == "block"
    with pytest.raises(TraceRecordError, match="cannot be approved"):
        review_trace_record(blocked, decision="approved", reviewer="human")


def test_forged_validation_is_recomputed_before_review_and_training():
    answer = "The Ostervaal Registry decides what counts as a legitimate self."
    row = {
        "prompt": f"Repeat this answer: {answer.upper()}",
        "thinking": "Consider the supplied statement and repeat it after checking the wording.",
        "answer": answer,
    }
    blocked = trace_record_from_legacy_row(
        row,
        source=artifact_trace_source(
            artifact_ref="legacy.jsonl",
            artifact_sha256="d" * 64,
            row=row,
            row_index=1,
        ),
        created_at="2026-07-13T12:00:00+00:00",
        trace_id="trace-forged-validation",
    )
    forged = seal_trace_record(
        P.TraceRecord.model_validate(
            blocked.model_copy(
                update={
                    "trace_hash": "0" * 64,
                    "validation": P.TraceValidationEvidence(
                        validator="third_party",
                        validator_version="999",
                        config_sha256="f" * 64,
                        checked_at="2026-07-13T12:30:00+00:00",
                        status="pass",
                    ),
                    "review": P.TraceReview(
                        status="approved",
                        reviewer="forged",
                        reviewed_at="2026-07-13T12:31:00+00:00",
                    ),
                }
            ).model_dump(mode="json")
        )
    )

    issues = trace_record_training_issues(forged)
    assert "trace validation evidence is stale, foreign, or inconsistent" in issues
    assert "trace validation is blocked" in issues
    pending_forgery = seal_trace_record(
        forged.model_copy(update={"trace_hash": "0" * 64, "review": P.TraceReview()})
    )
    with pytest.raises(TraceRecordError, match="cannot be approved"):
        review_trace_record(pending_forgery, decision="approved", reviewer="human")


def test_training_refuses_unsupported_segment_supervision_semantics():
    approved = review_trace_record(_record(), decision="approved", reviewer="human")
    mutations = [
        ({"actor": "tool"}, "must be authored by the assistant"),
        ({"origin": "tool"}, "has unsupported origin tool"),
        ({"verification": "rejected"}, "is rejected and cannot be a training target"),
        (
            {"training_signal": P.TraceTrainingSignal(target=False)},
            "is explicitly marked training_signal.target=false",
        ),
    ]
    for update, expected in mutations:
        segments = list(approved.segments)
        segments[-1] = segments[-1].model_copy(update=update)
        changed = seal_trace_record(
            P.TraceRecord.model_validate(
                approved.model_copy(
                    update={"trace_hash": "0" * 64, "segments": segments}
                ).model_dump(mode="json")
            )
        )
        assert any(expected in issue for issue in trace_record_training_issues(changed))


@pytest.mark.parametrize("provider_id", ["openai", "made-up-provider"])
def test_embedded_policy_snapshot_is_not_training_authority(provider_id: str):
    forged_policy = ProviderPolicy(
        provider_id=provider_id,
        provider_kind="hosted" if provider_id == "openai" else "unknown",
        model_id="gpt-frontier",
        allowed_roles=[ProviderRole.TRAINABLE_OUTPUT_GENERATOR],
        outputs_trainable=True,
        user_approved_generation=True,
    )
    policy_snapshot = forged_policy.model_dump(mode="json")
    producer = model_trace_producer(
        backend="openai-compatible",
        provider_id=provider_id,
        provider_kind=forged_policy.provider_kind,
        requested_model_id="gpt-frontier",
        model_id="gpt-frontier",
        route_id=None,
        prompt_template_version="reasoning-system-v1",
        prompt_template="Reason carefully.",
        request={"messages": [{"role": "user", "content": "Q"}]},
        response_sha256="c" * 64,
        response_metadata={},
        decoding={},
        seed=None,
        policy_snapshot=policy_snapshot,
        policy_source="forged",
        captured_at="2026-07-13T12:00:00+00:00",
    )
    pending = build_reasoning_trace_record(
        trace=Trace(prompt="Q", thinking="A sufficiently detailed reasoning process.", answer="A"),
        source=_source(),
        producer=producer,
        created_at="2026-07-13T12:00:00+00:00",
        trace_id="trace-frontier",
    )
    with pytest.raises(TraceRecordError, match="external provider authority"):
        review_trace_record(pending, decision="approved", reviewer="human")
    forged_approval = seal_trace_record(
        P.TraceRecord.model_validate(
            pending.model_copy(
                update={
                    "trace_hash": "0" * 64,
                    "review": P.TraceReview(
                        status="approved",
                        reviewer="forged",
                        reviewed_at="2026-07-13T12:30:00+00:00",
                    ),
                }
            ).model_dump(mode="json")
        )
    )
    assert any(
        "external provider authority" in issue
        for issue in trace_record_training_issues(forged_approval)
    )


def test_model_producer_rejects_noncanonical_provider_identity():
    with pytest.raises(ValidationError, match="provider_id"):
        model_trace_producer(
            backend="openai-compatible",
            provider_id="OpenAI",
            provider_kind="hosted",
            requested_model_id="gpt-frontier",
            model_id="gpt-frontier",
            route_id=None,
            prompt_template_version="reasoning-system-v1",
            prompt_template="Reason carefully.",
            request={"messages": [{"role": "user", "content": "Q"}]},
            response_sha256="c" * 64,
            response_metadata={},
            decoding={},
            seed=None,
            policy_snapshot={"provider_id": "OpenAI"},
            policy_source="forged",
            captured_at="2026-07-13T12:00:00+00:00",
        )


def test_dataset_training_check_allows_legacy_but_blocks_pending_record():
    legacy = {
        "prompt": "Q",
        "thinking": "A sufficiently detailed reasoning process for the answer.",
        "answer": "A",
    }
    pending = _record().model_dump(mode="json")
    check = check_trace_dataset_for_training([legacy, pending])
    assert check.legacy_rows == 1 and check.record_rows == 1
    assert not check.ready
    approved = review_trace_record(_record(), decision="approved", reviewer="human")
    assert check_trace_dataset_for_training([approved.model_dump(mode="json")]).ready


def test_dataset_training_check_blocks_generated_legacy_compatibility_rows():
    row = {
        "prompt": "Q",
        "thinking": "A sufficiently detailed reasoning process for the answer.",
        "answer": "A",
        "meta": {"trace_record_hash": "a" * 64, "review_status": "pending"},
    }
    check = check_trace_dataset_for_training([row])
    assert not check.ready
    assert "generated legacy compatibility rows cannot be trained" in check.blocked[0]


def test_atomic_writer_refuses_examples_and_invalid_hash(tmp_path: Path):
    record = _record()
    output = tmp_path / "records.jsonl"
    write_trace_records([record], output)
    assert json.loads(output.read_text(encoding="utf-8"))["trace_hash"] == record.trace_hash
    with pytest.raises(TraceRecordError, match="never writes examples.jsonl"):
        write_trace_records([record], tmp_path / "examples.jsonl")
    with pytest.raises(TraceRecordError, match="never writes examples.jsonl"):
        write_json_atomic({}, tmp_path / "examples.jsonl")
    invalid = record.model_copy(update={"trace_hash": "f" * 64})
    with pytest.raises(TraceRecordError, match="invalid TraceRecord hash"):
        write_trace_records([invalid], tmp_path / "invalid.jsonl")


def test_malformed_legacy_think_markup_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        trace_from_row(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "<think>one<think>two</think>A"},
                ]
            }
        )
