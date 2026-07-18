"""Trace CLI migration, policy, generation evidence, review, and validation workflow."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import corpus_studio.cli as cli
import corpus_studio.platform as P
from corpus_studio.cli import app
from corpus_studio.model_backends.base import (
    BackendGenerateRequest,
    BackendGenerateResponse,
    ModelBackendConfig,
)
from corpus_studio.platform.trace_records import (
    parse_trace_record,
    review_trace_record,
    seal_trace_record,
    trace_record_training_issues,
)
from corpus_studio.providers.overrides import approve_generation, load_overrides


runner = CliRunner()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeTraceBackend:
    config = ModelBackendConfig(
        provider_name="ollama",
        base_url="http://localhost:11434",
        model_name="approved-model",
    )

    def generate(self, request: BackendGenerateRequest) -> BackendGenerateResponse:
        assert request.messages[0]["role"] == "system"
        return BackendGenerateResponse(
            text=(
                "<think>Break the task into small parts, verify each intermediate value, "
                "and then combine them.</think>The final answer."
            ),
            model_name="approved-model@sha256:abc",
            raw={
                "model": "approved-model@sha256:abc",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                "message": {"content": "raw response body must not enter metadata"},
            },
        )


def test_trace_generate_blocks_unapproved_provider_before_backend_call(
    tmp_path: Path, monkeypatch
):
    prompts = tmp_path / "prompts.jsonl"
    _write_rows(prompts, [{"prompt": "Solve this."}])
    called = False

    def forbidden_backend(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("backend must not be constructed")

    monkeypatch.setattr(cli, "_build_backend", forbidden_backend)
    result = runner.invoke(
        app,
        [
            "trace-generate",
            str(prompts),
            "--out",
            str(tmp_path / "records.jsonl"),
            "--model",
            "unapproved",
        ],
    )

    assert result.exit_code == 2
    assert "Provider policy blocked trace generation" in result.output
    assert called is False


def test_trace_generate_writes_pending_records_and_sanitized_report(
    tmp_path: Path, monkeypatch
):
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "records.jsonl"
    _write_rows(
        prompts,
        [
            {
                "messages": [
                    {"role": "system", "content": "Use concise answers."},
                    {"role": "user", "content": "Solve this."},
                    {"role": "assistant", "content": "old target is trimmed"},
                ]
            }
        ],
    )
    approve_generation(tmp_path, "ollama", model_id="approved-model")
    approve_generation(tmp_path, "ollama", model_id="approved-model@sha256:abc")
    monkeypatch.setattr(cli, "_build_backend", lambda *args, **kwargs: FakeTraceBackend())

    result = runner.invoke(
        app,
        [
            "trace-generate",
            str(prompts),
            "--out",
            str(out),
            "--model",
            "approved-model",
            "--project-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    record = parse_trace_record(_read_rows(out)[0])
    assert record.review.status == "pending"
    assert [message.role for message in record.context] == ["system", "user"]
    assert record.producer.model_id == "approved-model@sha256:abc"
    assert record.producer.requested_model_id == "approved-model"
    assert record.producer.policy_decision and record.producer.policy_decision.allowed
    assert record.producer.response_metadata["usage"]["completion_tokens"] == 20
    assert "message" not in record.producer.response_metadata
    overrides = load_overrides(tmp_path)
    approved = review_trace_record(
        record,
        decision="approved",
        reviewer="human",
        provider_overrides=overrides,
    )
    assert trace_record_training_issues(approved, overrides) == []
    report = json.loads((tmp_path / "records.report.json").read_text(encoding="utf-8"))
    assert report["accepted"] == 1 and report["rejected"] == 0
    assert report["attempts"][0]["trace_id"] == record.trace_id
    assert "raw response body" not in json.dumps(report)


def test_trace_validate_review_and_training_approval_loop(tmp_path: Path):
    legacy = tmp_path / "legacy.jsonl"
    pending = tmp_path / "pending.jsonl"
    approved = tmp_path / "approved.jsonl"
    _write_rows(
        legacy,
        [
            {
                "prompt": "What is 17 multiplied by 23?",
                "thinking": "Multiply 17 by 20 and by 3, then add 340 and 51.",
                "answer": "391",
            }
        ],
    )

    migrated = runner.invoke(
        app,
        ["trace-migrate", str(legacy), "--out", str(pending)],
    )
    assert migrated.exit_code == 0, migrated.output
    pending_record = parse_trace_record(_read_rows(pending)[0])
    assert pending_record.review.status == "pending"

    structurally_valid = runner.invoke(app, ["trace-validate", str(pending), "--json"])
    assert structurally_valid.exit_code == 0
    assert json.loads(structurally_valid.output)["record_rows"] == 1

    forged_path = tmp_path / "forged-validation.jsonl"
    forged_record = seal_trace_record(
        P.TraceRecord.model_validate(
            pending_record.model_copy(
                update={
                    "trace_hash": "0" * 64,
                    "validation": P.TraceValidationEvidence(
                        validator="foreign",
                        validator_version="1",
                        config_sha256="f" * 64,
                        checked_at="2026-07-13T12:00:00+00:00",
                        status="pass",
                    ),
                }
            ).model_dump(mode="json")
        )
    )
    _write_rows(forged_path, [forged_record.model_dump(mode="json")])
    forged_validation = runner.invoke(app, ["trace-validate", str(forged_path), "--json"])
    assert forged_validation.exit_code == 3
    assert "stale, foreign, or inconsistent" in forged_validation.output

    not_approved = runner.invoke(
        app,
        ["trace-validate", str(pending), "--require-approved", "--json"],
    )
    assert not_approved.exit_code == 3
    assert json.loads(not_approved.output)["blocked"] == 1

    reviewed = runner.invoke(
        app,
        [
            "trace-review",
            str(pending),
            "--out",
            str(approved),
            "--reviewer",
            "human-reviewer",
            "--decision",
            "approved",
            "--all",
            "--note",
            "Checked derivation.",
        ],
    )
    assert reviewed.exit_code == 0, reviewed.output
    approved_record = parse_trace_record(_read_rows(approved)[0])
    assert approved_record.review.status == "approved"
    assert approved_record.parent_trace_refs[0].hash
    assert approved_record.parent_trace_refs[0].hash.value == pending_record.trace_hash

    training_ready = runner.invoke(
        app,
        ["trace-validate", str(approved), "--require-approved", "--json"],
    )
    assert training_ready.exit_code == 0, training_ready.output


def test_trace_migrate_rejects_malformed_markup_without_partial_output(tmp_path: Path):
    legacy = tmp_path / "legacy.jsonl"
    output = tmp_path / "records.jsonl"
    _write_rows(
        legacy,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "<think>one<think>two</think>A"},
                ]
            }
        ],
    )

    result = runner.invoke(
        app,
        ["trace-migrate", str(legacy), "--out", str(output)],
    )
    assert result.exit_code == 2
    assert "malformed reasoning markup" in result.output
    assert not output.exists()


def test_trace_migrate_refuses_engine_write_to_examples_jsonl(tmp_path: Path):
    legacy = tmp_path / "legacy.jsonl"
    _write_rows(
        legacy,
        [{"prompt": "Q", "thinking": "A detailed enough reasoning process.", "answer": "A"}],
    )
    result = runner.invoke(
        app,
        ["trace-migrate", str(legacy), "--out", str(tmp_path / "examples.jsonl")],
    )
    assert result.exit_code == 2
    assert "never writes examples.jsonl" in result.output  # message no longer names the removed desktop


def test_trace_generate_report_preserves_per_row_rejection(tmp_path: Path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "records.jsonl"
    _write_rows(prompts, [{"prompt": "usable"}, {"not_a_prompt": True}])
    approve_generation(tmp_path, "ollama", model_id="approved-model")
    approve_generation(tmp_path, "ollama", model_id="approved-model@sha256:abc")
    monkeypatch.setattr(cli, "_build_backend", lambda *args, **kwargs: FakeTraceBackend())

    result = runner.invoke(
        app,
        [
            "trace-generate",
            str(prompts),
            "--out",
            str(out),
            "--model",
            "approved-model",
            "--project-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    report = json.loads((tmp_path / "records.report.json").read_text(encoding="utf-8"))
    assert report["accepted"] == 1 and report["rejected"] == 1
    assert report["attempts"][1]["reason"] == "no usable prompt/context"


def test_trace_generate_refuses_unapproved_resolved_model(tmp_path: Path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "records.jsonl"
    _write_rows(prompts, [{"prompt": "usable"}])
    approve_generation(tmp_path, "ollama", model_id="approved-model")
    monkeypatch.setattr(cli, "_build_backend", lambda *args, **kwargs: FakeTraceBackend())

    result = runner.invoke(
        app,
        [
            "trace-generate",
            str(prompts),
            "--out",
            str(out),
            "--model",
            "approved-model",
            "--project-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert _read_rows(out) == []
    report = json.loads((tmp_path / "records.report.json").read_text(encoding="utf-8"))
    assert report["accepted"] == 0 and report["rejected"] == 1
    assert "not generation-approved" in report["attempts"][0]["reason"]


def test_trace_generate_rejects_output_report_or_input_path_collisions(
    tmp_path: Path, monkeypatch
):
    prompts = tmp_path / "prompts.jsonl"
    _write_rows(prompts, [{"prompt": "usable"}])
    original = prompts.read_text(encoding="utf-8")
    called = False

    def forbidden_backend(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("path validation must happen before backend construction")

    monkeypatch.setattr(cli, "_build_backend", forbidden_backend)
    same_as_input = runner.invoke(
        app,
        ["trace-generate", str(prompts), "--out", str(prompts), "--model", "anything"],
    )
    assert same_as_input.exit_code == 2
    assert "distinct input, output, report, and sidecar paths" in same_as_input.output
    assert prompts.read_text(encoding="utf-8") == original

    out = tmp_path / "records.jsonl"
    same_report = runner.invoke(
        app,
        [
            "trace-generate",
            str(prompts),
            "--out",
            str(out),
            "--report",
            str(out),
            "--model",
            "anything",
        ],
    )
    assert same_report.exit_code == 2
    assert "distinct input, output, report, and sidecar paths" in same_report.output
    assert not out.exists()
    assert called is False

    unsafe_report = runner.invoke(
        app,
        [
            "trace-generate",
            str(prompts),
            "--out",
            str(out),
            "--report",
            str(tmp_path / "examples.jsonl"),
            "--model",
            "anything",
        ],
    )
    assert unsafe_report.exit_code == 2
    assert "never writes examples.jsonl" in unsafe_report.output
    assert called is False


def test_generated_legacy_output_is_explicitly_non_trainable(tmp_path: Path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "legacy.jsonl"
    _write_rows(prompts, [{"prompt": "usable"}])
    approve_generation(tmp_path, "ollama", model_id="approved-model")
    approve_generation(tmp_path, "ollama", model_id="approved-model@sha256:abc")
    monkeypatch.setattr(cli, "_build_backend", lambda *args, **kwargs: FakeTraceBackend())

    generated = runner.invoke(
        app,
        [
            "trace-generate",
            str(prompts),
            "--out",
            str(out),
            "--model",
            "approved-model",
            "--project-dir",
            str(tmp_path),
            "--legacy-output",
        ],
    )
    assert generated.exit_code == 0, generated.output
    assert "unsealed and non-trainable" in generated.output
    sidecar = tmp_path / "legacy.trace-records.jsonl"
    pending = parse_trace_record(_read_rows(sidecar)[0])
    assert pending.review.status == "pending"
    assert _read_rows(out)[0]["meta"]["trace_record_hash"] == pending.trace_hash
    assert _read_rows(out)[0]["meta"]["trace_record_ref"] == sidecar.name

    validation = runner.invoke(app, ["trace-validate", str(out), "--require-approved", "--json"])
    assert validation.exit_code == 3
    assert "generated legacy compatibility rows cannot be trained" in validation.output
