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


def test_google_gemini_palm_and_vertex_teachers_are_quarantined():
    teachers = (
        "gemini-2.5-pro",
        "palm-2",
        "google/gemini-2.5-pro",
        "gemini/gemini-2.0-flash",
        "palm/text-bison",
        "vertex-ai/gemini-1.5-pro",
        "vertexai/gemini-1.5-flash",
    )
    for teacher in teachers:
        status, provider, note = classify_teacher(teacher)
        assert status is TeacherStatus.QUARANTINED, teacher
        assert provider == "google"
        assert "evaluator-only" in note

    report = run_provenance_gate([_row(teacher) for teacher in teachers])
    assert report.overall_status is GateStatus.BLOCK
    assert report.quarantined_rows == len(teachers)
    assert report.unknown_rows == 0


def test_open_gpt_family_models_are_not_mislabeled_openai():
    # #567: open EleutherAI / GPT-2 weights must NOT be quarantined as restricted OpenAI provenance;
    # they fall through to UNKNOWN (allow-listable), which is honest rather than a false 'openai' label.
    for teacher in ("gpt-neo-2.7b", "gpt-j-6b", "gpt-neox-20b", "gpt2", "gpt-2"):
        status, provider, _ = classify_teacher(teacher)
        assert provider != "openai", teacher
        assert status is TeacherStatus.UNKNOWN, teacher


def test_proprietary_openai_models_are_still_quarantined():
    for teacher in (
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "chatgpt-4o-latest",
        "o1-mini",
        "o3-mini",
        "davinci-002",
        "text-davinci-003",
    ):
        status, provider, _ = classify_teacher(teacher)
        assert provider == "openai" and status is TeacherStatus.QUARANTINED, teacher


def test_word_boundary_avoids_over_matching():
    # 'palmyra' (Writer) must not match Google's 'palm'; 'olmo' (AllenAI) must not match the o-series.
    assert classify_teacher("palmyra-x-004")[1] != "google"
    assert classify_teacher("olmo-2-13b")[1] != "openai"


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


# ---- provenance-allowlist-set CLI (#580, G5) ---------------------------------

from typer.testing import CliRunner  # noqa: E402

from corpus_studio.cli import app  # noqa: E402

_runner = CliRunner()


def test_cli_allow_adds_entry_with_note(tmp_path):
    result = _runner.invoke(
        app, ["provenance-allowlist-set", str(tmp_path), "--allow", "mistral=Apache-2.0", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["allowlist"] == {"mistral": "Apache-2.0"} and payload["added"] == ["mistral"]
    assert load_provenance_allowlist(tmp_path) == {"mistral": "Apache-2.0"}


def test_cli_allow_without_note_uses_a_default(tmp_path):
    result = _runner.invoke(app, ["provenance-allowlist-set", str(tmp_path), "--allow", "mistral"])
    assert result.exit_code == 0, result.output
    assert load_provenance_allowlist(tmp_path)["mistral"]  # non-empty default note


def test_cli_remove_deletes_and_reports_not_found(tmp_path):
    save_provenance_allowlist(tmp_path, {"mistral": "Apache-2.0", "qwen": "Apache-2.0"})
    result = _runner.invoke(
        app,
        ["provenance-allowlist-set", str(tmp_path), "--remove", "mistral", "--remove", "ghost", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["removed"] == ["mistral"] and payload["not_found"] == ["ghost"]
    assert load_provenance_allowlist(tmp_path) == {"qwen": "Apache-2.0"}


def test_cli_values_json_replaces_entire_allowlist(tmp_path):
    save_provenance_allowlist(tmp_path, {"old": "note"})
    result = _runner.invoke(
        app,
        ["provenance-allowlist-set", str(tmp_path), "--values-json", '{"qwen": "Apache-2.0"}', "--json"],
    )
    assert result.exit_code == 0, result.output
    assert load_provenance_allowlist(tmp_path) == {"qwen": "Apache-2.0"}  # "old" is gone


def test_cli_values_json_refuses_non_string_note(tmp_path):
    # a hand-crafted "approval object" must be refused, never written - fail closed
    result = _runner.invoke(
        app,
        ["provenance-allowlist-set", str(tmp_path), "--values-json", '{"claude": {"approved": true}}'],
    )
    assert result.exit_code == 1
    assert "must be a string" in result.output
    assert load_provenance_allowlist(tmp_path) == {}  # nothing written


def test_cli_refuses_empty_teacher(tmp_path):
    result = _runner.invoke(app, ["provenance-allowlist-set", str(tmp_path), "--allow", "=just a note"])
    assert result.exit_code == 1
    assert "empty teacher" in result.output


def test_cli_values_json_and_incremental_are_mutually_exclusive(tmp_path):
    result = _runner.invoke(
        app,
        ["provenance-allowlist-set", str(tmp_path), "--allow", "mistral", "--values-json", "{}"],
    )
    assert result.exit_code == 1
    assert "cannot be combined" in result.output


def test_cli_nothing_to_do_is_refused(tmp_path):
    result = _runner.invoke(app, ["provenance-allowlist-set", str(tmp_path)])
    assert result.exit_code == 1
    assert "Nothing to do" in result.output


def test_cli_missing_project_dir(tmp_path):
    result = _runner.invoke(
        app, ["provenance-allowlist-set", str(tmp_path / "nope"), "--allow", "mistral"]
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_cli_report_is_the_net_diff_not_raw_ops(tmp_path):
    # A teacher named in BOTH --remove and --allow ends up allow-listed (the user asked for it); the
    # report must reflect the NET state - it appears in `added`, never in `removed`.
    result = _runner.invoke(
        app,
        ["provenance-allowlist-set", str(tmp_path), "--remove", "claude", "--allow", "claude=oops", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["allowlist"] == {"claude": "oops"}
    assert payload["added"] == ["claude"] and payload["removed"] == [] and payload["not_found"] == []


def test_cli_values_json_discloses_dropped_approvals(tmp_path):
    # Wholesale replace must be HONEST about which prior approvals it drops.
    save_provenance_allowlist(tmp_path, {"old-teacher": "note"})
    result = _runner.invoke(
        app,
        ["provenance-allowlist-set", str(tmp_path), "--values-json", '{"qwen": "Apache-2.0"}', "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["added"] == ["qwen"] and payload["removed"] == ["old-teacher"]
