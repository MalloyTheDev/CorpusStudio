"""Keyword-overlap scoring for the Evaluation Studio.

IMPORTANT: this is a lexical *recall* heuristic, NOT a quality judgment. It measures
how many of the expected output's words appear in the model output (case-folded,
whitespace-split) — nothing about correctness, meaning, order, or precision. A model
that echoes the expected keywords plus noise scores 100; a correct paraphrase using
synonyms scores low. Reports label this metric ``keyword_overlap`` so it is never
presented as a quality score. Trustworthy scoring comes from manual review or the
opt-in judge-model scorer (see ``arena/judge.py``).
"""


def score_text_overlap(expected: str, actual: str) -> float:
    """Return the fraction (0-100) of the expected output's words present in the
    model output — keyword-overlap recall, not a quality score. See module docstring."""

    expected_terms = {term.lower() for term in expected.split() if term.strip()}
    actual_terms = {term.lower() for term in actual.split() if term.strip()}
    if not expected_terms:
        return 0.0

    return round((len(expected_terms & actual_terms) / len(expected_terms)) * 100, 2)

