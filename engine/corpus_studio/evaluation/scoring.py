"""Lightweight scoring helpers for Evaluation Lab skeleton tests."""


def score_text_overlap(expected: str, actual: str) -> float:
    """Return a simple percentage based on expected-token overlap.

    This is placeholder scoring for early UI/report plumbing. Real v0.2 scoring
    should support rubrics, manual scoring, and optional judge-model workflows.
    """

    expected_terms = {term.lower() for term in expected.split() if term.strip()}
    actual_terms = {term.lower() for term in actual.split() if term.strip()}
    if not expected_terms:
        return 0.0

    return round((len(expected_terms & actual_terms) / len(expected_terms)) * 100, 2)

