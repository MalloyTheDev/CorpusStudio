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
