"""The AI-generated candidate rows are gated (schema/quality/PII) BEFORE they
reach the human review queue.

The gate is a pre-review *safety signal only*: it never auto-accepts a clean
batch and never auto-rejects a blocked one — ``review_required`` stays True in
every case, and the suggested rows are always preserved for the human. Policy is
still enforced first, so a forbidden provider can never reach the gate step.
"""

import json

import pytest

from corpus_studio.ai_assist.assistant import run_ai_assist
from corpus_studio.gates.models import GateStatus
from corpus_studio.model_backends.base import BackendGenerateResponse
from corpus_studio.providers.policy import ProviderPolicyError, resolve_policy


def _backend_returning(suggested_rows):
    """A fake backend whose response proposes ``suggested_rows`` as suggested_jsonl."""

    suggested_jsonl = "\n".join(json.dumps(row) for row in suggested_rows)

    class FakeBackend:
        def generate(self, request):
            return BackendGenerateResponse(
                text=json.dumps(
                    {
                        "summary": "Drafted candidate rows.",
                        "suggested_jsonl": suggested_jsonl,
                        "tags": [],
                        "warnings": [],
                    }
                ),
                model_name="fake-model",
            )

    return FakeBackend()


def _run(suggested_rows, *, action="draft-example", policy=None, backend=None):
    return run_ai_assist(
        schema_id="instruction",
        action=action,
        rows=[
            {
                "instruction": "Explain what a Python for loop does.",
                "input": "",
                "output": "A for loop iterates over each item of a sequence.",
            }
        ],
        backend=backend or _backend_returning(suggested_rows),
        model="fake-model",
        policy=policy,
    )


def test_clean_candidate_gate_passes_but_stays_review_required():
    result = _run(
        [
            {
                "instruction": "Explain what a Python for loop does.",
                "input": "",
                "output": (
                    "A for loop iterates over a sequence such as a list, tuple, "
                    "or range, running its indented body once for each element "
                    "until the sequence is exhausted."
                ),
            }
        ]
    )

    assert result.candidate_gate is not None
    assert result.candidate_gate.overall_status == GateStatus.PASS
    # A clean gate is NOT approval — the human must still review.
    assert result.review_required is True
    assert result.review_state == "review_required"


def test_generated_secret_blocks_gate_but_is_never_auto_rejected():
    secret_row = {
        "instruction": "Show an AWS credentials snippet.",
        "input": "",
        "output": "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
    }

    result = _run([secret_row])

    assert result.candidate_gate is not None
    # High-severity PII/secret in generated content -> the PII gate blocks.
    assert result.candidate_gate.overall_status == GateStatus.BLOCK
    assert result.candidate_gate.block_count >= 1
    # A block INFORMS review; it does not auto-reject. review_required stays True
    # and the candidate is still present so the human can see and reject it.
    assert result.review_required is True
    assert result.suggested_jsonl.strip() != ""
    assert "AKIAIOSFODNN7EXAMPLE" in result.suggested_jsonl


def test_no_suggested_rows_produces_no_candidate_gate():
    class NotesOnlyBackend:
        def generate(self, request):
            return BackendGenerateResponse(
                text=json.dumps(
                    {
                        "summary": "Review notes only; no rewrite proposed.",
                        "suggested_jsonl": "",
                        "tags": [],
                        "warnings": [],
                    }
                ),
                model_name="fake-model",
            )

    result = _run([], action="review", backend=NotesOnlyBackend())

    # No candidate rows -> nothing to gate -> candidate_gate is None (not a fake pass).
    assert result.suggested_jsonl.strip() == ""
    assert result.candidate_gate is None


def test_duplicate_heavy_candidate_batch_surfaces_quality_gate():
    row = {
        "instruction": "Explain a Python loop.",
        "input": "",
        "output": "A loop repeats a block of code until a condition is met.",
    }

    # Two identical generated rows -> the exact-duplicate quality gate fires over
    # the CANDIDATE rows (reusing the existing dataset gate runner).
    result = _run([row, dict(row)])

    assert result.candidate_gate is not None
    assert result.candidate_gate.overall_status in {GateStatus.WARN, GateStatus.BLOCK}
    dup_gate = next(
        gate for gate in result.candidate_gate.results if gate.gate_id == "quality"
    )
    assert dup_gate.status in {GateStatus.WARN, GateStatus.BLOCK}
    # Still never auto-rejected.
    assert result.review_required is True


def test_policy_denied_provider_raises_before_any_gating():
    # OpenAI is evaluator-only by default: it may not run a trainable-generating
    # action. authorize_action must raise BEFORE the provider is called, so the
    # gate step can never run on a generation the policy forbids.
    policy = resolve_policy("openai")

    class ExplodingBackend:
        def generate(self, request):  # pragma: no cover - must never be reached
            raise AssertionError("provider was called despite a denying policy")

    with pytest.raises(ProviderPolicyError):
        _run([], action="draft-example", policy=policy, backend=ExplodingBackend())
