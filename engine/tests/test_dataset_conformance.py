"""Structural dataset-format conformance preflight - the CPU gate that refuses to seal a plan whose
selected dataset_format cannot render a usable row from the immutable dataset.

The chat rows here mirror the real bring-up fixture ``pipeline_smoke_fixture_v2.jsonl`` (a ``messages``
list of system/user/assistant turns) so this is a regression test for the observed
``UNSUPPORTED_CONFIGURATION`` / "The dataset produced no usable training rows." failure, which happened
because a chat dataset was planned as ``instruction``.
"""

import json

import pytest

from corpus_studio.platform.dataset_conformance import (
    DatasetConformanceError,
    assess_dataset_file_conformance,
    assess_dataset_format_conformance,
    load_jsonl_rows,
)

# One structurally valid chat row, shaped exactly like pipeline_smoke_fixture_v2.jsonl.
CHAT_ROW = {
    "messages": [
        {"role": "system", "content": "Follow the user instruction exactly and answer concisely."},
        {"role": "user", "content": "Write the lowercase form of ALPHA."},
        {"role": "assistant", "content": "The lowercase form is alpha."},
    ]
}
INSTRUCTION_ROW = {"instruction": "Say hello.", "output": "Hello."}


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# ---- chat ----------------------------------------------------------------------------------------


def test_chat_fixture_rows_are_all_compatible():
    report = assess_dataset_format_conformance([CHAT_ROW, CHAT_ROW, CHAT_ROW], "chat")
    assert report.is_conformant
    assert (report.total_rows, report.compatible_rows, report.rejected_rows) == (3, 3, 0)
    assert report.representative_rejections == ()


def test_chat_fixture_planned_as_instruction_is_not_conformant():
    # The exact observed failure: chat rows have no instruction/output fields -> zero usable rows.
    report = assess_dataset_format_conformance([CHAT_ROW, CHAT_ROW], "instruction")
    assert not report.is_conformant
    assert report.compatible_rows == 0
    assert report.rejected_rows == 2
    assert "instruction" in report.representative_rejections[0].reason


def test_chat_missing_messages_is_rejected():
    report = assess_dataset_format_conformance([{"text": "no messages here"}], "chat")
    assert not report.is_conformant
    assert "messages" in report.representative_rejections[0].reason


def test_chat_message_not_an_object_is_rejected():
    row = {"messages": [{"role": "user", "content": "hi"}, "not an object"]}
    report = assess_dataset_format_conformance([row], "chat")
    assert report.compatible_rows == 0
    assert "not an object" in report.representative_rejections[0].reason


def test_chat_unrecognized_role_is_rejected():
    row = {"messages": [{"role": "critic", "content": "hmm"}, {"role": "assistant", "content": "a"}]}
    report = assess_dataset_format_conformance([row], "chat")
    assert report.compatible_rows == 0
    assert "unrecognized role" in report.representative_rejections[0].reason


def test_chat_empty_assistant_content_is_rejected():
    row = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "   "},
        ]
    }
    report = assess_dataset_format_conformance([row], "chat")
    assert report.compatible_rows == 0
    assert "empty content" in report.representative_rejections[0].reason


def test_chat_without_assistant_turn_has_no_trainable_target():
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    report = assess_dataset_format_conformance([row], "chat")
    assert report.compatible_rows == 0
    assert "trainable assistant turn" in report.representative_rejections[0].reason


def test_chat_accepts_structured_content_parts():
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
    }
    report = assess_dataset_format_conformance([row], "chat")
    assert report.is_conformant


# ---- instruction ---------------------------------------------------------------------------------


def test_instruction_rows_pass():
    report = assess_dataset_format_conformance([INSTRUCTION_ROW, INSTRUCTION_ROW], "instruction")
    assert report.is_conformant
    assert report.compatible_rows == 2


def test_instruction_empty_is_rejected():
    report = assess_dataset_format_conformance([{"instruction": "  ", "output": ""}], "instruction")
    assert not report.is_conformant
    assert report.rejected_rows == 1


def test_instruction_output_only_still_renders():
    # format_example_text renders when instruction OR output is non-empty; mirror that exactly.
    report = assess_dataset_format_conformance([{"output": "answer"}], "instruction")
    assert report.is_conformant


# ---- counts + representative rejections ----------------------------------------------------------


def test_mixed_rows_report_exact_counts():
    rows = [CHAT_ROW, {"messages": []}, CHAT_ROW, {"nope": 1}]
    report = assess_dataset_format_conformance(rows, "chat")
    assert (report.total_rows, report.compatible_rows, report.rejected_rows) == (4, 2, 2)
    assert report.is_conformant  # >=1 compatible row keeps planning open


def test_representative_rejections_capped_but_total_counted():
    rows = [{"nope": i} for i in range(9)]  # all rejected
    report = assess_dataset_format_conformance(rows, "chat")
    assert report.total_rows == 9
    assert report.compatible_rows == 0
    assert report.rejected_rows == 9  # every rejected row is counted
    assert len(report.representative_rejections) == 5  # but only a few are shown


def test_non_object_row_is_rejected():
    report = assess_dataset_format_conformance([["not", "a", "dict"]], "chat")
    assert report.compatible_rows == 0
    assert "not a JSON object" in report.representative_rejections[0].reason


# ---- trace ---------------------------------------------------------------------------------------


def test_trace_accepts_chat_style_and_prompt_answer():
    ok_chat = assess_dataset_format_conformance([CHAT_ROW], "trace")
    ok_pa = assess_dataset_format_conformance([{"prompt": "Q", "answer": "A"}], "trace")
    assert ok_chat.is_conformant and ok_pa.is_conformant


def test_trace_without_structure_is_rejected():
    report = assess_dataset_format_conformance([{"note": "nothing trainable"}], "trace")
    assert not report.is_conformant


# ---- trace: #750 regression - the classifier must mirror the worker's own renderer ---------------


def _sealed_trace_record_row() -> dict:
    """A valid sealed TraceRecord row (top-level trace identity/segments/producer, NO messages/prompt) -
    exactly what write_trace_records emits and what the worker renders via trace_from_row."""
    import corpus_studio.platform as P  # noqa: PLC0415
    from corpus_studio.platform.trace_records import (  # noqa: PLC0415
        build_reasoning_trace_record,
        imported_trace_producer,
    )
    from corpus_studio.training.traces import Trace  # noqa: PLC0415

    record = build_reasoning_trace_record(
        trace=Trace(prompt="What is 17 * 23?", thinking="20*17 + 3*17 = 340 + 51.", answer="391"),
        source=P.TraceSource(
            artifact_ref="source.jsonl",
            artifact_sha256="a" * 64,
            source_row_id="b" * 64,
            source_row_index=1,
        ),
        producer=imported_trace_producer(),
        created_at="2026-07-13T12:00:00+00:00",
        trace_id="trace-750",
        tags=["reasoning"],
    )
    return record.model_dump(mode="json")


def test_trace_accepts_sealed_trace_record_row():
    # #750 (most severe): a sealed TraceRecord row (no top-level messages/prompt/answer) is rendered by
    # the worker via trace_from_row -> legacy_trace_from_record; the classifier must recognize it.
    from corpus_studio.platform.trace_records import is_trace_record_row  # noqa: PLC0415

    row = _sealed_trace_record_row()
    assert is_trace_record_row(row) and "messages" not in row and "prompt" not in row
    report = assess_dataset_format_conformance([row], "trace")
    assert report.is_conformant and report.compatible_rows == 1


def test_trace_accepts_full_worker_alias_sets():
    # #750: the classifier omitted input/query (prompt) and solution/completion (answer) that the
    # worker's trace_from_row accepts, so renderable reasoning datasets were falsely refused at plan time.
    rows = [
        {"query": "What is 2+2?", "solution": "4"},
        {"input": "Translate 'hi'.", "completion": "hola"},
    ]
    report = assess_dataset_format_conformance(rows, "trace")
    assert report.is_conformant and report.compatible_rows == 2


def test_trace_messages_not_ending_in_assistant_is_rejected():
    # trace_from_row only derives an answer from the LAST assistant message; a conversation whose last
    # turn is the user (with no separate answer field) renders "" - the classifier must reject it, not
    # false-accept it by delegating to the chat path.
    row = {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
    }
    assert not assess_dataset_format_conformance([row], "trace").is_conformant


def test_trace_classifier_agrees_with_the_worker_renderer():
    # Anti-drift invariant: for every fixture, the trace classifier admits a row IFF the worker's own
    # format_example_text renders it non-empty. This is what keeps the preflight from diverging again
    # (import is torch-free: trainer.py's torch imports are all lazy and the trace path is pure).
    from corpus_studio.platform.dataset_conformance import _classify_trace  # noqa: PLC0415
    from corpus_studio.training.trainer import format_example_text  # noqa: PLC0415

    fixtures = [
        _sealed_trace_record_row(),
        {"query": "q", "solution": "s"},
        {"input": "i", "completion": "c"},
        {"prompt": "p", "answer": "a"},
        CHAT_ROW,  # ends in an assistant turn -> renders
        {"messages": [{"role": "user", "content": "u"}]},  # no assistant -> ""
        {"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}]},  # not last
        {"prompt": "p only, no answer"},  # no answer -> ""
        {"note": "nothing trainable"},  # nothing -> ""
    ]
    for row in fixtures:
        admitted = _classify_trace(row) is None
        renders = format_example_text(dict(row), "trace").strip() != ""
        assert admitted == renders, f"classifier vs worker disagree on {row!r}: {admitted} != {renders}"


# ---- format + loader errors ----------------------------------------------------------------------


def test_unknown_format_raises():
    with pytest.raises(DatasetConformanceError, match="unknown dataset_format"):
        assess_dataset_format_conformance([INSTRUCTION_ROW], "parquet_rows")


def test_never_reinterprets_format():
    # A chat dataset assessed as chat passes; the SAME dataset assessed as instruction fails. The
    # module must not silently reinterpret one as the other.
    assert assess_dataset_format_conformance([CHAT_ROW], "chat").is_conformant
    assert not assess_dataset_format_conformance([CHAT_ROW], "instruction").is_conformant


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(CHAT_ROW) + "\n\n" + json.dumps(CHAT_ROW) + "\n", encoding="utf-8")
    assert len(load_jsonl_rows(path)) == 2


# ---- partial-conformance messaging (the plan over-claim guard) -----------------------------------


def test_describe_partial_refusal_carries_ascii_counts():
    rows = [CHAT_ROW, {"messages": []}, CHAT_ROW]  # 2 compatible, 1 rejected
    report = assess_dataset_format_conformance(rows, "chat")
    message = report.describe_partial_refusal("data/train.jsonl")
    assert message.isascii()
    assert "1 of 3 row(s)" in message
    assert "over-claim the trained row count" in message
    assert "--allow-unrenderable-rows" in message
    assert "row 1:" in message  # the rejected row's index is surfaced


def test_describe_partial_warning_reports_the_kept_count():
    rows = [CHAT_ROW, {"messages": []}, CHAT_ROW]
    report = assess_dataset_format_conformance(rows, "chat")
    warning = report.describe_partial_warning("data/train.jsonl")
    assert warning.isascii()
    assert warning.startswith("WARNING:")
    assert "1 of 3 row(s)" in warning
    assert "only the 2 compatible row(s)" in warning


def test_load_jsonl_malformed_line_raises(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"messages": [}\n', encoding="utf-8")
    with pytest.raises(DatasetConformanceError, match="not valid JSON"):
        load_jsonl_rows(path)


def test_missing_file_raises():
    with pytest.raises(DatasetConformanceError, match="cannot read dataset"):
        assess_dataset_file_conformance("/nonexistent/path/dataset.jsonl", "chat")


# ---- report shape --------------------------------------------------------------------------------


def test_as_dict_is_json_serializable_and_ascii_message():
    report = assess_dataset_format_conformance([CHAT_ROW, {"x": 1}], "chat")
    payload = report.as_dict()
    assert json.loads(json.dumps(payload))["total_rows"] == 2
    message = report.describe_refusal("/data/x.jsonl") if not report.is_conformant else "ok"
    assert message == "ok"  # this dataset IS conformant (1 usable row)
    refusal = assess_dataset_format_conformance([{"x": 1}], "chat").describe_refusal("/data/x.jsonl")
    assert refusal.isascii() and "NOT" in refusal
