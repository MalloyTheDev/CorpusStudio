from corpus_studio.quality.basic_quality import (
    _luhn_valid,
    build_basic_quality_report,
)


def _kinds(report) -> set[str]:
    return {finding.kind for finding in report.pii_findings}


def test_detects_email():
    report = build_basic_quality_report([{"text": "reach me at john.doe@example.com"}])
    assert "email" in _kinds(report)


def test_detects_aws_access_key():
    report = build_basic_quality_report([{"text": "key AKIAIOSFODNN7EXAMPLE here"}])
    assert "aws_access_key" in _kinds(report)


def test_detects_private_key_block():
    report = build_basic_quality_report([{"text": "-----BEGIN RSA PRIVATE KEY-----\nMIIB"}])
    assert "private_key" in _kinds(report)


def test_detects_api_key_and_jwt():
    rows = [
        {"text": "token sk-abcdefghijklmnop1234567890"},
        {
            "text": "auth eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpM"
        },
    ]
    report = build_basic_quality_report(rows)
    kinds = _kinds(report)
    assert "api_key" in kinds
    assert "jwt" in kinds


def test_detects_ssn():
    report = build_basic_quality_report([{"text": "SSN 123-45-6789 on file"}])
    assert "ssn" in _kinds(report)


def test_detects_valid_credit_card_but_not_invalid():
    valid = build_basic_quality_report([{"text": "card 4111 1111 1111 1111"}])
    assert "credit_card" in _kinds(valid)

    invalid = build_basic_quality_report([{"text": "card 4111 1111 1111 1112"}])
    assert "credit_card" not in _kinds(invalid)


def test_luhn_reference_values():
    assert _luhn_valid("4111111111111111") is True
    assert _luhn_valid("4111111111111112") is False


def test_clean_text_has_no_pii_findings():
    report = build_basic_quality_report(
        [{"instruction": "Explain recursion.", "output": "A function that calls itself."}]
    )
    assert report.pii_finding_count == 0
    assert report.pii_findings == []


def test_findings_aggregate_rows_and_mask_sample():
    rows = [
        {"text": "a@b.com"},
        {"text": "c@d.org and e@f.net"},
    ]
    report = build_basic_quality_report(rows)
    email = next(finding for finding in report.pii_findings if finding.kind == "email")
    assert email.row_numbers == [1, 2]
    assert email.match_count == 3
    # The sample is masked, not the raw address.
    assert "@" not in email.sample or "*" in email.sample


def test_findings_sorted_high_severity_first():
    report = build_basic_quality_report(
        [{"text": "email x@y.com and key AKIAIOSFODNN7EXAMPLE"}]
    )
    assert report.pii_findings[0].severity == "high"


# --- item 12: Luhn-valid non-cards (IMEIs, order numbers) must not flag as cards -----

def test_luhn_valid_imei_is_not_flagged_as_card():
    # 490154203237518 is a Luhn-valid 15-digit IMEI, but matches no card brand's IIN +
    # length (only Amex is 15 digits, and it starts 34/37) — it must not block export.
    assert _luhn_valid("490154203237518") is True
    report = build_basic_quality_report([{"text": "device imei 490154203237518 in logs"}])
    assert "credit_card" not in _kinds(report)


def test_luhn_valid_order_number_is_not_flagged_as_card():
    # A Luhn-valid 16-digit run that starts with 12 (no brand IIN) is not a card.
    assert _luhn_valid("1234567890123452") is True
    report = build_basic_quality_report([{"text": "order 1234567890123452 shipped"}])
    assert "credit_card" not in _kinds(report)


def test_real_amex_card_still_flagged():
    # Canonical 15-digit Amex test number (starts 37, Luhn-valid) — still detected.
    report = build_basic_quality_report([{"text": "amex 378282246310005 on file"}])
    assert "credit_card" in _kinds(report)


# --- C14: UnionPay + Maestro re-added, IMEI-safe -----------------------------

def test_detects_unionpay_and_maestro_cards():
    unionpay = build_basic_quality_report([{"text": "pay 6200 0000 0000 0005 now"}])
    assert "credit_card" in _kinds(unionpay)  # UnionPay (IIN 62, 16 digits)
    maestro = build_basic_quality_report([{"text": "card 5000 0000 0000 0009"}])
    assert "credit_card" in _kinds(maestro)   # Maestro (IIN 50, 16 digits)


def test_15_digit_maestro_prefix_stays_imei_safe():
    # A 15-digit Luhn-valid number starting 50 must NOT flag as a card: UnionPay/Maestro are
    # constrained to 16-19 digits precisely so a 15-digit IMEI can never masquerade as one.
    assert _luhn_valid("500000000000005") is True
    report = build_basic_quality_report([{"text": "serial 500000000000005 logged"}])
    assert "credit_card" not in _kinds(report)
