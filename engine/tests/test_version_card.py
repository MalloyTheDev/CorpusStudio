from types import SimpleNamespace

from corpus_studio.versions.version_card import (
    build_version_card,
    render_version_card_markdown,
)
from corpus_studio.versions.version_registry import DatasetVersionRecord


def _record(**kw) -> DatasetVersionRecord:
    base = dict(version_id="20260101T000000-1", created_at="t", updated_at="t", content_fingerprint="abc")
    base.update(kw)
    return DatasetVersionRecord(**base)


def test_integrity_flags_from_fingerprints():
    record = _record()
    assert build_version_card(record, current_fingerprint="abc").current_integrity == "matches"
    drifted = build_version_card(record, current_fingerprint="zzz")
    assert drifted.current_integrity == "drifted"
    assert any("changed since" in w for w in drifted.warnings)
    unreadable = build_version_card(record, current_fingerprint=None)
    assert unreadable.current_integrity == "unreadable"
    assert any("Cannot verify" in w for w in unreadable.warnings)


def test_missing_run_and_artifact_links_warn():
    record = _record(source_run_ids=["r-gone"], artifact_ids=["a-gone"])
    card = build_version_card(record, current_fingerprint="abc", runs_by_id={}, artifacts_by_id={})
    assert card.runs[0].present is False
    assert card.artifacts[0].present is False
    assert any("training run 'r-gone'" in w for w in card.warnings)
    assert any("model artifact 'a-gone'" in w for w in card.warnings)


def test_present_run_and_artifact_resolve_live():
    record = _record(source_run_ids=["r1"], artifact_ids=["a1"])
    run = SimpleNamespace(status="succeeded", base_model="llama-3-8b")
    artifact = SimpleNamespace(status="kept")
    card = build_version_card(
        record,
        current_fingerprint="abc",
        runs_by_id={"r1": run},
        artifacts_by_id={"a1": (artifact, "ok")},
    )
    assert card.runs[0].present and card.runs[0].status == "succeeded"
    assert card.runs[0].base_model == "llama-3-8b"
    assert card.artifacts[0].present and card.artifacts[0].integrity == "ok"


def test_modified_artifact_integrity_warns():
    record = _record(artifact_ids=["a1"])
    artifact = SimpleNamespace(status="kept")
    card = build_version_card(
        record, current_fingerprint="abc", artifacts_by_id={"a1": (artifact, "modified")}
    )
    assert any("integrity is modified" in w for w in card.warnings)


def test_eval_link_present_and_missing():
    record = _record(eval_report_path="/x/eval.json")
    present = build_version_card(
        record,
        current_fingerprint="abc",
        load_eval_report=lambda _p: SimpleNamespace(average_score=82.5),
    )
    assert present.eval_report_present and present.eval_average_score == 82.5

    missing = build_version_card(record, current_fingerprint="abc", load_eval_report=lambda _p: None)
    assert missing.eval_report_linked and not missing.eval_report_present
    assert any("evaluation report is missing" in w for w in missing.warnings)


def test_non_numeric_eval_score_degrades_not_crash():
    record = _record(eval_report_path="/x/eval.json")
    card = build_version_card(
        record,
        current_fingerprint="abc",
        load_eval_report=lambda _p: SimpleNamespace(average_score="n/a"),
    )
    assert card.eval_report_present and card.eval_average_score is None
    assert any("non-numeric" in w for w in card.warnings)


def test_non_finite_eval_score_degrades():
    record = _record(eval_report_path="/x/eval.json")
    card = build_version_card(
        record,
        current_fingerprint="abc",
        load_eval_report=lambda _p: SimpleNamespace(average_score=float("nan")),
    )
    assert card.eval_average_score is None
    assert any("non-finite" in w for w in card.warnings)


def test_fingerprint_render_is_injection_safe():
    # A hand-edited record with a control char in the (<=12 char) fingerprint must
    # not inject an extra Markdown line — the fingerprint is sanitized on render.
    record = _record(content_fingerprint="x\n> PWNED-fp")
    card = build_version_card(record, current_fingerprint="x\n> PWNED-fp")  # matches
    markdown = render_version_card_markdown(card)
    assert "\n> PWNED-fp" not in markdown


def test_gate_link_reports_overall_status():
    record = _record(gate_report_path="/x/gate.json")
    card = build_version_card(
        record,
        current_fingerprint="abc",
        load_gate_report=lambda _p: SimpleNamespace(overall_status=SimpleNamespace(value="warn")),
    )
    assert card.gate_report_present and card.gate_overall_status == "warn"


def test_render_leads_with_warnings_and_is_injection_safe():
    record = _record(label="line1\n> injected blockquote", content_fingerprint="abc")
    card = build_version_card(record, current_fingerprint="zzz")  # drifted
    markdown = render_version_card_markdown(card)
    # Warning appears before the Lineage section.
    assert markdown.index("⚠") < markdown.index("## Lineage")
    # Newline in the untrusted label is neutralized (no raw injected blockquote line).
    assert "\n> injected blockquote" not in markdown
    assert "drifted" in markdown
