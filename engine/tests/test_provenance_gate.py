"""Per-row provenance gate (data-safety): teacher classification + dataset verdict.

Anchored on the real world-bible-generator scenario that motivated the feature — a
combined set of 450 Claude-Opus-generated rows + 172 MIT-clean rows + 77 untagged —
which must BLOCK because the 450 are non-trainable-provenance (distilling Claude).
"""

from __future__ import annotations

import json

from corpus_studio.gates.models import GateStatus
from corpus_studio.gates.provenance_gate import (
    TeacherStatus,
    classify_teacher,
    load_provenance_allowlist,
    render_provenance_gate_text,
    run_provenance_gate,
    save_provenance_allowlist,
)


def _row(teacher=None, **meta):
    m = dict(meta)
    if teacher is not None:
        m["teacher"] = teacher
    return {"messages": [{"role": "user", "content": "x"}], "meta": m}


# ---- classify_teacher --------------------------------------------------------


def test_frontier_teachers_are_quarantined():
    for teacher in ("claude-opus-4-8", "claude-3-5-sonnet", "anthropic/claude-x", "gpt-4o", "o1-mini"):
        status, provider, note = classify_teacher(teacher)
        assert status is TeacherStatus.QUARANTINED, teacher
        assert provider in {"anthropic", "openai"}
        assert note  # carries the license/terms reason


def test_untagged_is_unknown_not_quarantined_or_pass():
    for teacher in (None, "", "   "):
        status, provider, _ = classify_teacher(teacher)
        assert status is TeacherStatus.UNKNOWN
        assert provider == ""


def test_unrecognized_teacher_is_unknown_until_allowlisted():
    # An MIT-clean open model the gate can't recognize is UNKNOWN by default (never a false PASS)…
    status, provider, _ = classify_teacher("z-ai/glm-5.2")
    assert status is TeacherStatus.UNKNOWN
    assert provider == "z-ai"

    # …and PASS once the user declares it clean (by exact teacher or by provider).
    by_teacher = classify_teacher("z-ai/glm-5.2", {"z-ai/glm-5.2": "MIT"})
    assert by_teacher[0] is TeacherStatus.PASS
    assert by_teacher[2] == "MIT"
    by_provider = classify_teacher("z-ai/glm-5.2", {"z-ai": "MIT open weights"})
    assert by_provider[0] is TeacherStatus.PASS


def test_recognized_local_provider_is_trainable_clean():
    status, provider, _ = classify_teacher("ollama")
    assert status is TeacherStatus.PASS
    assert provider == "ollama"


def test_allowlist_cannot_be_bypassed_but_frontier_stays_quarantined():
    # Even if a user tries to allow-list a frontier teacher, the allow-list only
    # affects the note→PASS path; a bare-frontier resolve stays blocked otherwise.
    # (Allow-listing IS the user's explicit override, so PASS here is by design;
    # the guardrail is that resolve_policy's frontier block can't be flipped by
    # project overrides — verified in the policy tests. This asserts the allow-list
    # is an *explicit* opt-in, not a default.)
    default_status, _, _ = classify_teacher("claude-opus-4-8")
    assert default_status is TeacherStatus.QUARANTINED


# ---- run_provenance_gate: the WBG scenario -----------------------------------


def test_wbg_combined_set_blocks_on_claude_rows():
    rows = (
        [_row("claude-opus-4-8") for _ in range(450)]
        + [_row("z-ai/glm-5.2") for _ in range(172)]
        + [_row(None) for _ in range(77)]
    )
    report = run_provenance_gate(rows, allowlist={"z-ai/glm-5.2": "MIT"})

    assert report.total_rows == 699
    assert report.quarantined_rows == 450
    assert report.trainable_rows == 172
    assert report.unknown_rows == 77
    assert report.overall_status is GateStatus.BLOCK
    assert "450" in report.summary

    # Buckets are most-severe first (quarantined leads the human table).
    assert report.buckets[0].status is TeacherStatus.QUARANTINED
    assert report.buckets[0].teacher == "claude-opus-4-8"
    assert report.buckets[0].row_count == 450


def test_all_clean_passes():
    rows = [_row("z-ai/glm-5.2") for _ in range(10)]
    report = run_provenance_gate(rows, allowlist={"z-ai/glm-5.2": "MIT"})
    assert report.overall_status is GateStatus.PASS
    assert report.quarantined_rows == 0
    assert report.unknown_rows == 0


def test_unknown_only_warns_but_strict_blocks():
    rows = [_row(None) for _ in range(5)]
    lenient = run_provenance_gate(rows)
    assert lenient.overall_status is GateStatus.WARN
    assert lenient.unknown_rows == 5

    strict = run_provenance_gate(rows, strict=True)
    assert strict.overall_status is GateStatus.BLOCK


def test_non_dict_rows_count_as_untagged():
    report = run_provenance_gate([_row("claude-opus-4-8"), "not-a-dict", 42])  # type: ignore[list-item]
    assert report.total_rows == 3
    assert report.quarantined_rows == 1
    assert report.unknown_rows == 2


def test_custom_teacher_field_dotted_path():
    rows = [{"provenance": {"model": "gpt-4o"}}]
    report = run_provenance_gate(rows, teacher_field="provenance.model")
    assert report.quarantined_rows == 1
    assert report.buckets[0].teacher == "gpt-4o"


def test_render_text_shows_verdict_and_quarantine():
    rows = [_row("claude-opus-4-8"), _row(None)]
    text = render_provenance_gate_text(run_provenance_gate(rows))
    assert "QUARANTINED" in text
    assert "claude-opus-4-8" in text
    assert "VERDICT:" in text
    assert "BLOCK" in text


# ---- allowlist persistence (fail-closed) -------------------------------------


def test_allowlist_round_trips_and_fails_closed(tmp_path):
    assert load_provenance_allowlist(tmp_path) == {}  # absent → empty

    save_provenance_allowlist(tmp_path, {"z-ai/glm-5.2": "MIT", "mistral": ""})
    loaded = load_provenance_allowlist(tmp_path)
    assert loaded["z-ai/glm-5.2"] == "MIT"
    assert loaded["mistral"] == ""

    # A non-dict value (a hand edit that looks like an approval object) is dropped,
    # not coerced into a truthy note — fail closed.
    (tmp_path / "provenance_allowlist.json").write_text(
        json.dumps({"good": "MIT", "bad": {"approved": True}}), encoding="utf-8"
    )
    salvaged = load_provenance_allowlist(tmp_path)
    assert salvaged == {"good": "MIT"}

    # A non-dict top-level file → empty (never crashes the gate).
    (tmp_path / "provenance_allowlist.json").write_text("[]", encoding="utf-8")
    assert load_provenance_allowlist(tmp_path) == {}
