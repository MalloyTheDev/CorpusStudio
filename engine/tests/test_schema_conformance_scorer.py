"""Deterministic schema-conformance scorer (eval workflow S2): does the model emit ONE JSON object with
every required key of a declared schema? Score is 0/100 per example so average_score == conformance rate.
A non-JSON / truncated output is a measured 0 with reason (never a raised scorer_error)."""
import json

from corpus_studio.evaluation.scorers import SchemaConformanceScorer
from corpus_studio.schemas.registry import load_builtin_schema

_REQUIRED = [
    "kind", "module", "title", "entryType", "status", "summary", "tags",
    "relationships", "canonNotes", "risks", "storyHooks", "gameHooks", "suggestedNextAction",
]


def _complete() -> dict:
    # Non-empty scalars + legitimately-empty required arrays (exactly what the reference data carries).
    return {
        "kind": "lore", "module": "locations", "title": "The Grove", "entryType": "location",
        "status": "draft", "summary": "A summary.", "tags": ["forest"], "relationships": [],
        "canonNotes": "notes", "risks": [], "storyHooks": ["a hook"], "gameHooks": [],
        "suggestedNextAction": "draft the archive",
    }


def _scorer(**kwargs) -> SchemaConformanceScorer:
    return SchemaConformanceScorer(load_builtin_schema("airesult"), **kwargs)


def test_airesult_builtin_schema_loads_with_the_13_required_keys():
    schema = load_builtin_schema("airesult")
    names = {f.name for f in schema.fields}
    required = {f.name for f in schema.fields if f.required}
    assert set(_REQUIRED) <= names
    assert required == set(_REQUIRED)  # names/legacyTitle are optional


def test_complete_json_with_empty_required_arrays_scores_100_presence_only():
    result = _scorer().score("p", "e", json.dumps(_complete()))
    assert result.score == 100.0  # empty arrays are VALID under presence-only (the headline metric)


def test_missing_a_required_key_scores_0_with_reason():
    obj = _complete()
    del obj["risks"]
    result = _scorer().score("p", "e", json.dumps(obj))
    assert result.score == 0.0
    assert "risks" in (result.rationale or "")


def test_truncated_or_non_json_is_a_measured_zero_with_reason_not_a_raise():
    truncated = _scorer().score("p", "e", '{"kind": "lore", "module": "loc')  # unterminated
    assert truncated.score == 0.0
    assert "json" in (truncated.rationale or "").lower()
    prose = _scorer().score("p", "e", "I could not produce JSON.")
    assert prose.score == 0.0


def test_markdown_fenced_json_is_parsed():
    fenced = "```json\n" + json.dumps(_complete()) + "\n```"
    assert _scorer().score("p", "e", fenced).score == 100.0


def test_require_nonempty_tier_penalizes_legitimately_empty_required_arrays():
    # The stricter opt-in tier: an output with empty required arrays (valid under presence-only, and
    # present in the reference data) fails - which is exactly why presence-only is the headline metric.
    strict = _scorer(require_nonempty=True).score("p", "e", json.dumps(_complete()))
    assert strict.score == 0.0
