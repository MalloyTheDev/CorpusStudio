"""Content-fidelity scorer (#749): does a structured output reproduce the reference's per-key CONTENT
(empty vs non-empty parity by default, or exact value match)? Reported ALONGSIDE completeness, so a
degenerate all-empty-optional output - which scores 100 on schema_conformance presence-only - is
surfaced as low fidelity. Honesty: an unparseable MODEL output is a measured 0 with a reason; an
unparseable / empty REFERENCE raises (an excluded scorer_error), never a fabricated score."""

import json

import pytest

from corpus_studio.evaluation.scorers import ContentFidelityScorer


def _ref() -> dict:
    # Two filled content fields + one legitimately-empty field: the shape the fidelity metric measures.
    return {"title": "The Grove", "canonNotes": "old growth", "relationships": []}


def test_identical_output_scores_100_in_both_modes():
    ref = json.dumps(_ref())
    assert ContentFidelityScorer().score("p", ref, ref).score == 100.0
    assert ContentFidelityScorer(exact_match=True).score("p", ref, ref).score == 100.0


def test_all_empty_optional_output_is_complete_but_low_fidelity():
    # The #749 case: an output that empties every content field is structurally "complete" (presence of
    # keys) yet reproduces none of the reference's content - it must NOT score 100 on fidelity.
    ref = {"title": "The Grove", "canonNotes": "old growth", "relationships": ["allies"]}
    degenerate = {"title": "", "canonNotes": "", "relationships": []}
    result = ContentFidelityScorer().score("p", json.dumps(ref), json.dumps(degenerate))
    assert result.score == 0.0  # all three reference keys carry content; the model empties all three
    assert "canonNotes" in (result.rationale or "")


def test_empty_nonempty_parity_counts_both_underfill_and_overfill():
    # Under-fill: reference fills canonNotes, model empties it -> mismatch. Over-fill: reference leaves
    # relationships empty, model fills it -> mismatch. title agrees (both filled). 1 of 3 -> 33.33.
    ref = {"title": "The Grove", "canonNotes": "notes", "relationships": []}
    model = {"title": "A Grove", "canonNotes": "", "relationships": ["allies"]}
    result = ContentFidelityScorer().score("p", json.dumps(ref), json.dumps(model))
    assert result.score == pytest.approx(33.33, abs=0.01)
    assert "canonNotes" in (result.rationale or "") and "relationships" in (result.rationale or "")


def test_parity_agrees_on_two_filled_values_even_when_they_differ():
    # Parity measures FILL agreement, not equality: two non-empty values agree (that is what exact_match
    # is for). This is why the parity rate is the honest headline and exact_match is the stricter option.
    ref = {"summary": "a long reference summary"}
    model = {"summary": "different text"}
    assert ContentFidelityScorer().score("p", json.dumps(ref), json.dumps(model)).score == 100.0


def test_exact_match_mode_requires_equal_values():
    ref = {"summary": "reference", "tags": ["a"]}
    model = {"summary": "different", "tags": ["a"]}  # tags equal, summary differs
    # exact: 1 of 2 match -> 50; parity: both filled -> 100.
    assert ContentFidelityScorer(exact_match=True).score("p", json.dumps(ref), json.dumps(model)).score == 50.0
    assert ContentFidelityScorer().score("p", json.dumps(ref), json.dumps(model)).score == 100.0


def test_missing_key_in_model_is_a_mismatch_when_reference_has_content():
    ref = {"title": "The Grove", "canonNotes": "notes"}
    model = {"title": "The Grove"}  # canonNotes omitted; reference has content -> mismatch
    result = ContentFidelityScorer().score("p", json.dumps(ref), json.dumps(model))
    assert result.score == 50.0
    assert "canonNotes" in (result.rationale or "")


def test_unparseable_model_output_is_a_measured_zero_not_a_raise():
    result = ContentFidelityScorer().score("p", json.dumps(_ref()), "I could not produce JSON.")
    assert result.score == 0.0
    assert result.rationale  # a typed reason (the model delivered no content - a real measurement)


def test_unparseable_reference_raises_so_the_row_is_an_excluded_scorer_error():
    # A non-JSON reference cannot anchor a fidelity measurement; raise (the evaluator records an excluded
    # scorer_error) rather than fabricate a 0 or a 100.
    with pytest.raises(ValueError):
        ContentFidelityScorer().score("p", "not json at all", '{"title": "x"}')


def test_empty_reference_object_raises():
    with pytest.raises(ValueError):
        ContentFidelityScorer().score("p", "{}", '{"title": "x"}')
