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
