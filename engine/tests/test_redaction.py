"""Tests for opt-in PII/secret redaction on export (issue #194)."""

from __future__ import annotations

from corpus_studio.exporters.redaction import redact_rows, redact_text
from corpus_studio.quality.basic_quality import build_basic_quality_report

# One representative value per detected pattern.
_SAMPLES = {
    "email": "reach me at jane.doe@example.com please",
    "ssn": "ssn 123-45-6789 on file",
    "api_key": "token sk-Abcd0123456789EfGhIjKlMn here",
    "aws_access_key": "key AKIAIOSFODNN7EXAMPLE rotated",
    "jwt": "auth eyJhbGciOiJIUzI1.eyJzdWIiOiJ1MjM.SflKxwRJSMeK done",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIB...",
    "credit_card": "card 4111 1111 1111 1111 charged",
}


def test_redact_text_masks_each_known_pattern() -> None:
    for kind, text in _SAMPLES.items():
        redacted, hits = redact_text(text)
        assert hits.get(kind, 0) >= 1, f"{kind} not counted"
        assert f"[REDACTED:{kind}]" in redacted, f"{kind} not masked in {redacted!r}"


def test_redact_text_leaves_clean_text_untouched() -> None:
    clean = "The quick brown fox summarizes the water cycle in one sentence."
    redacted, hits = redact_text(clean)
    assert redacted == clean
    assert hits == {}


def test_redact_text_does_not_mask_arbitrary_long_numbers() -> None:
    # A 16-digit run that is NOT Luhn-valid must not be masked as a card.
    redacted, hits = redact_text("order 1234567890123456 shipped")
    assert "credit_card" not in hits
    assert "[REDACTED" not in redacted


def test_redact_rows_manifest_counts_rows_and_spans_without_raw_values() -> None:
    rows = [
        {"instruction": "contact", "output": "email jane.doe@example.com or ssn 123-45-6789"},
        {"instruction": "clean", "output": "no sensitive data here"},
        {"instruction": "secret", "output": "use sk-Abcd0123456789EfGhIjKlMn now"},
    ]
    redacted, report = redact_rows(rows)

    assert report.redacted_rows == 2  # rows 1 and 3
    assert report.redacted_spans == 3  # email + ssn + api_key
    assert report.affected_row_numbers == [1, 3]
    kinds = {hit.kind: hit.count for hit in report.by_kind}
    assert kinds == {"api_key": 1, "email": 1, "ssn": 1}

    # The manifest carries no raw secret values — only kinds/counts/row numbers.
    dumped = report.model_dump_json()
    assert "jane.doe@example.com" not in dumped
    assert "123-45-6789" not in dumped
    assert "sk-Abcd0123456789EfGhIjKlMn" not in dumped


def test_redaction_clears_what_detection_flags() -> None:
    # Honesty parity: whatever the quality reporter flags as PII/secrets is exactly what redaction
    # masks, so a redacted dataset reports zero PII findings.
    rows = [
        {"instruction": "a", "output": "email jane.doe@example.com key sk-Abcd0123456789EfGhIjKlMn"},
        {"instruction": "b", "output": "ssn 123-45-6789 card 4111 1111 1111 1111"},
    ]
    assert build_basic_quality_report(rows).pii_finding_count > 0
    redacted, _ = redact_rows(rows)
    assert build_basic_quality_report(redacted).pii_finding_count == 0


def test_redact_rows_handles_nested_list_and_dict_values() -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": "my email is jane.doe@example.com"},
                {"role": "assistant", "content": "noted"},
            ]
        }
    ]
    redacted, report = redact_rows(rows)
    assert report.redacted_spans == 1
    assert "[REDACTED:email]" in redacted[0]["messages"][0]["content"]
    assert redacted[0]["messages"][1]["content"] == "noted"


def test_redact_masks_numeric_pii_so_the_gate_can_clear() -> None:
    # A payment-card number stored as a JSON number is flagged by PII detection but was
    # not redactable (#505). Now it is; a legit number keeps its value and type.
    rows = [{"card": 4111111111111111, "note": "ok", "qty": 42}]  # "note" separates the numbers

    before = build_basic_quality_report(rows)
    assert before.pii_finding_count >= 1  # the numeric card is flagged

    redacted, report = redact_rows(rows)
    assert redacted[0]["card"] == "[REDACTED:credit_card]"
    assert redacted[0]["qty"] == 42  # legit number unchanged (value + type)
    assert any(hit.kind == "credit_card" for hit in report.by_kind)

    after = build_basic_quality_report(redacted)
    assert after.pii_finding_count == 0  # redaction cleared it -> the gate can pass
