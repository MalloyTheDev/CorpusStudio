"""Private-key detection and redaction cover the whole PEM block, not only its header.

Regression for the export leak where ``--redact-pii`` masked the ``-----BEGIN ... PRIVATE KEY-----``
line but wrote the base64 key body and the END line into the deliverable, after which the PII
export gate passed; and for the same leak when the block spans several fields or list items, where
only the leaves holding a marker were masked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.exporters.redaction import redact_rows, redact_text
from corpus_studio.gates.models import GateStatus
from corpus_studio.gates.runner import run_export_gates
from corpus_studio.quality.basic_quality import _PII_PRIVATE_KEY_RE, build_basic_quality_report

runner = CliRunner()

MASK = "[REDACTED:private_key]"
BODY_1 = "MIIEowIBAAKCAQEA1c2Vj3Zx9Qm8b5Q7Xl0H2kRj8vq3n5T0bYt9w+8JHq1c7Xy"
BODY_2 = "Z3k2Lr0PqJ5m9Yx8c7v6b5n4m3l2k1j0h9g8f7e6d5c4b3a2Z1Y0X9W8V7U6T5S4"
BODY_3 = "QmFzZTY0Qm9keUxpbmVUaHJlZUZvclRoZVJTQUtleU1hdGVyaWFsWFlaMTIzNDU2"
BEGIN_RSA = "-----BEGIN RSA PRIVATE KEY-----"
END_RSA = "-----END RSA PRIVATE KEY-----"
LABELS = [
    "RSA PRIVATE KEY",
    "EC PRIVATE KEY",
    "DSA PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
    "PRIVATE KEY",
    "PGP PRIVATE KEY BLOCK",
]


def _block(label: str = "RSA PRIVATE KEY") -> str:
    return f"-----BEGIN {label}-----\n{BODY_1}\n{BODY_2}\n-----END {label}-----"


def _assert_no_key_material(text: str) -> None:
    assert BODY_1 not in text
    assert BODY_2 not in text
    assert BODY_3 not in text
    assert "PRIVATE KEY" not in text


def _detected_key_matches(rows: list[dict]) -> int:
    findings = [f for f in build_basic_quality_report(rows).pii_findings if f.kind == "private_key"]
    return findings[0].match_count if findings else 0


def _assert_redaction_clears_the_gate(rows: list[dict], schema: str) -> list[dict]:
    """Redact ``rows`` and check the whole contract: the gate blocked before, redaction counted
    every block the detector counted, no key material is left, and the gate no longer blocks."""
    assert run_export_gates(rows, schema).overall_status == GateStatus.BLOCK
    detected = _detected_key_matches(rows)

    redacted, report = redact_rows(rows)

    _assert_no_key_material(json.dumps(redacted))
    assert {hit.kind: hit.count for hit in report.by_kind} == {"private_key": detected}
    assert _detected_key_matches(redacted) == 0
    assert run_export_gates(redacted, schema).overall_status != GateStatus.BLOCK
    return redacted


@pytest.mark.parametrize("label", LABELS)
def test_redact_text_masks_the_whole_block_for_every_label(label: str) -> None:
    redacted, hits = redact_text(f"deploy key:\n{_block(label)}\nthanks")

    assert redacted == f"deploy key:\n{MASK}\nthanks"
    assert hits == {"private_key": 1}


def test_redact_text_masks_an_unterminated_block_to_the_end_of_the_text() -> None:
    redacted, hits = redact_text(f"key:\n-----BEGIN OPENSSH PRIVATE KEY-----\n{BODY_1}\n{BODY_2}\n")

    assert redacted == f"key:\n{MASK}"
    assert hits == {"private_key": 1}


def test_redact_text_masks_each_block_and_keeps_the_prose_between_them() -> None:
    text = f"first\n{_block('EC PRIVATE KEY')}\nbetween\n{_block('PRIVATE KEY')}\nlast"

    redacted, hits = redact_text(text)

    assert redacted == f"first\n{MASK}\nbetween\n{MASK}\nlast"
    assert hits == {"private_key": 2}


def test_redact_text_masks_the_body_before_an_end_marker_without_a_begin() -> None:
    # The second half of a key split across fields: its BEGIN marker lives in another string.
    redacted, hits = redact_text(f"{BODY_1}\n{BODY_2}\n-----END RSA PRIVATE KEY-----\nafter")

    assert redacted == f"{MASK}\nafter"
    assert hits == {"private_key": 1}


def test_redact_text_folds_a_header_only_residue_into_one_mask() -> None:
    # What the old header-only redaction left behind: the body and END line beside a placeholder.
    residue = f"deploy key:\n{MASK}\n{BODY_1}\n{BODY_2}\n-----END RSA PRIVATE KEY-----"

    redacted, hits = redact_text(residue)

    assert redacted == f"deploy key:\n{MASK}"
    assert hits == {"private_key": 1}


def test_a_stray_end_marker_after_a_block_masks_only_back_to_that_block() -> None:
    text = f"keep\n{_block()}\n{BODY_1}\n-----END RSA PRIVATE KEY-----\ntail"

    redacted, hits = redact_text(text)

    assert redacted == f"keep\n{MASK}{MASK}\ntail"
    assert hits == {"private_key": 2}


def test_public_material_and_prose_are_not_masked() -> None:
    text = (
        "-----BEGIN CERTIFICATE-----\nMIIBszCCAVmgAwIBAgIU\n-----END CERTIFICATE-----\n"
        "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYI\n-----END PUBLIC KEY-----\n"
        "A private key is secret; never share it."
    )

    redacted, hits = redact_text(text)

    assert redacted == text
    assert hits == {}


def test_detection_flags_every_residue_shape_and_samples_only_the_marker() -> None:
    rows = [
        {"text": _block()},
        {"text": f"-----BEGIN PRIVATE KEY-----\n{BODY_1}"},
        {"text": f"{MASK}\n{BODY_1}\n-----END EC PRIVATE KEY-----"},
    ]

    report = build_basic_quality_report(rows)

    (finding,) = [f for f in report.pii_findings if f.kind == "private_key"]
    assert finding.row_numbers == [1, 2, 3]
    assert finding.match_count == 3  # one match per block, not one per line
    # The sample is the masked "-----BEGIN RSA PRIVATE KEY-----" marker: no key body at its edges.
    assert finding.sample == "--" + "*" * (len("-----BEGIN RSA PRIVATE KEY-----") - 4) + "--"


def test_redaction_clears_the_gate_for_whole_split_and_residue_keys() -> None:
    rows = [
        {"instruction": "Store this key.", "output": f"deploy key:\n{_block('DSA PRIVATE KEY')}"},
        # One key split across two fields: BEGIN + first line here, second line + END there.
        {
            "instruction": f"-----BEGIN RSA PRIVATE KEY-----\n{BODY_1}",
            "output": f"{BODY_2}\n-----END RSA PRIVATE KEY-----",
        },
        {"instruction": "Old export.", "output": f"{MASK}\n{BODY_1}\n-----END RSA PRIVATE KEY-----"},
    ]
    before = run_export_gates(rows, "instruction")
    assert before.overall_status == GateStatus.BLOCK
    (detected,) = [f for f in build_basic_quality_report(rows).pii_findings if f.kind == "private_key"]

    redacted, report = redact_rows(rows)

    _assert_no_key_material(json.dumps(redacted))
    assert report.affected_row_numbers == [1, 2, 3]
    # The split key is one block in the row, so redaction counts what the detector counts.
    assert {hit.kind: hit.count for hit in report.by_kind} == {"private_key": 3}
    assert detected.match_count == 3
    assert build_basic_quality_report(redacted).pii_finding_count == 0
    assert run_export_gates(redacted, "instruction").overall_status != GateStatus.BLOCK


def test_export_gate_blocks_a_header_only_residue_without_redaction() -> None:
    rows = [{"text": f"deploy key:\n{MASK}\n{BODY_1}\n{BODY_2}\n-----END RSA PRIVATE KEY-----"}]

    report = run_export_gates(rows, "raw_text")

    assert report.overall_status == GateStatus.BLOCK


def test_redact_rows_masks_a_key_stored_as_a_list_of_lines() -> None:
    # The body lines hold no marker, so masking each leaf on its own left all of them in the export.
    rows = [{"text": "config file", "meta": {"lines": [BEGIN_RSA, BODY_1, BODY_2, BODY_3, END_RSA]}}]

    (redacted,) = _assert_redaction_clears_the_gate(rows, "raw_text")

    assert redacted == {"text": "config file", "meta": {"lines": [MASK] * 5}}


def test_redact_rows_masks_an_unterminated_key_continued_in_a_later_field() -> None:
    rows = [
        {
            "instruction": f"Here is my key:\n-----BEGIN EC PRIVATE KEY-----\n{BODY_1}",
            "output": BODY_2,
        }
    ]

    (redacted,) = _assert_redaction_clears_the_gate(rows, "instruction")

    assert redacted == {"instruction": f"Here is my key:\n{MASK}", "output": MASK}


def test_redact_rows_masks_a_header_only_residue_back_to_its_placeholder() -> None:
    # A list-of-lines key after the old header-only redaction: placeholder, body lines, END line.
    rows = [{"text": "keep me", "meta": {"lines": [MASK, BODY_1, BODY_2, END_RSA]}}]

    (redacted,) = _assert_redaction_clears_the_gate(rows, "raw_text")

    assert redacted == {"text": "keep me", "meta": {"lines": [MASK] * 4}}


def test_a_stray_end_with_no_begin_in_the_row_masks_back_to_the_start_of_the_row() -> None:
    # Nothing marks where the body starts, so the mask fails closed over the earlier fields too.
    rows = [{"instruction": "Key tail follows.", "output": f"{BODY_2}\n-----END PRIVATE KEY-----\nafter"}]

    (redacted,) = _assert_redaction_clears_the_gate(rows, "instruction")

    assert redacted == {"instruction": MASK, "output": f"{MASK}\nafter"}


def test_a_key_split_across_chat_messages_keeps_the_roles_valid() -> None:
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": f"key part 1:\n{BEGIN_RSA}\n{BODY_1}"},
                {"role": "assistant", "content": "ok, send more"},
                {"role": "user", "content": f"{BODY_2}\n{END_RSA}"},
                {"role": "assistant", "content": "Thanks."},
            ]
        }
    ]

    (redacted,) = _assert_redaction_clears_the_gate(rows, "chat")

    # Every content inside the block is masked; the role words stay, so the row is still valid chat.
    assert redacted["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": f"key part 1:\n{MASK}"},
        {"role": "assistant", "content": MASK},
        {"role": "user", "content": MASK},
        {"role": "assistant", "content": "Thanks."},
    ]
    assert run_export_gates([redacted], "chat").overall_status == GateStatus.PASS


def test_a_role_field_that_is_not_a_chat_role_word_is_masked_inside_a_block() -> None:
    rows = [{"text": BEGIN_RSA, "meta": {"role": BODY_1, "kind": "user"}, "footer": END_RSA}]

    (redacted,) = _assert_redaction_clears_the_gate(rows, "raw_text")

    # Only a "role" key holding a chat role word is kept; any other leaf in the block is masked.
    assert redacted == {"text": MASK, "meta": {"role": MASK, "kind": MASK}, "footer": MASK}


def test_row_spans_map_onto_every_kind_of_leaf() -> None:
    rows = [
        {
            "text": "head",
            "a": f"x {_block()} y {BEGIN_RSA}\n{BODY_1}",
            "b": None,
            "c": "",
            "d": [BODY_2, 7, 2.5, True, {"n": None}],
            "e": f"{BODY_3}\n{END_RSA} z",
            "f": "tail",
        }
    ]

    (redacted,) = _assert_redaction_clears_the_gate(rows, "raw_text")

    # Two blocks: one inside "a", one from "a" through "e". An empty string stays empty, and JSON
    # numbers and booleans inside a block keep their value and type (they are not PEM text).
    assert redacted == {
        "text": "head",
        "a": f"x {MASK} y {MASK}",
        "b": None,
        "c": "",
        "d": [MASK, 7, 2.5, True, {"n": None}],
        "e": f"{MASK} z",
        "f": "tail",
    }


def test_rows_without_a_private_key_are_unchanged_by_the_row_pass() -> None:
    rows = [{"text": "-----BEGIN PUBLIC KEY-----\nMFkw\n-----END PUBLIC KEY-----", "n": 3}]

    redacted, report = redact_rows(rows)

    assert redacted == rows
    assert report.redacted_spans == 0


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_export_redact_pii_writes_no_key_material(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    _write(source, [{"text": f"deploy key:\n{_block()}"}])
    output = tmp_path / "out.jsonl"

    blocked = runner.invoke(app, ["export", str(source), str(output), "raw_text"])
    assert blocked.exit_code == 2
    assert not output.exists()

    result = runner.invoke(app, ["export", str(source), str(output), "raw_text", "--redact-pii"])

    assert result.exit_code == 0, result.output
    exported = output.read_text(encoding="utf-8")
    _assert_no_key_material(exported)
    assert json.loads(exported) == {"text": f"deploy key:\n{MASK}"}
    manifest = json.loads(
        output.with_name(output.name + ".redaction_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["redacted_spans"] == 1
    _assert_no_key_material(json.dumps(manifest))


def test_cli_export_blocks_a_header_only_residue(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    _write(source, [{"text": f"{MASK}\n{BODY_1}\n-----END RSA PRIVATE KEY-----"}])
    output = tmp_path / "out.jsonl"

    result = runner.invoke(app, ["export", str(source), str(output), "raw_text"])

    assert result.exit_code == 2
    assert "private_key" in result.output
    assert not output.exists()


def test_cli_export_redact_pii_masks_a_key_stored_as_a_list_of_lines(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    _write(source, [{"text": "config file", "meta": {"lines": [BEGIN_RSA, BODY_1, BODY_2, BODY_3, END_RSA]}}])
    output = tmp_path / "out.jsonl"

    blocked = runner.invoke(app, ["export", str(source), str(output), "raw_text"])
    assert blocked.exit_code == 2
    assert not output.exists()

    result = runner.invoke(app, ["export", str(source), str(output), "raw_text", "--redact-pii"])

    assert result.exit_code == 0, result.output
    exported = output.read_text(encoding="utf-8")
    _assert_no_key_material(exported)
    assert json.loads(exported) == {"text": "config file", "meta": {"lines": [MASK] * 5}}
    manifest = json.loads(
        output.with_name(output.name + ".redaction_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["redacted_spans"] == 1


# Inputs shaped to make a backtracking pattern re-scan: near-miss markers, repeated BEGINs with no
# END, many stray ENDs after long base64 runs, and long dash runs. ~200 KB each: a linear pass takes
# milliseconds, while a quadratic one takes tens of seconds, so the bound below has wide headroom.
_ADVERSARIAL_SIZE = 200_000
_ADVERSARIAL_INPUTS = {
    # Each near miss fails on its last character, after the longest possible partial match.
    "near_miss_end": "-----BEGIN RSA PRIVATE KEY-----"
    + "-----END RSA PRIVATE KEY----x" * (_ADVERSARIAL_SIZE // 29),
    "unterminated_begins": "-----BEGIN PRIVATE KEY-----" * (_ADVERSARIAL_SIZE // 27),
    "near_miss_begin": "-----BEGIN RSA PRIVATE KEY----x" * (_ADVERSARIAL_SIZE // 31),
    "stray_ends": ("A" * 64 + "-----END PRIVATE KEY-----") * (_ADVERSARIAL_SIZE // 89),
    "dashes": "-" * _ADVERSARIAL_SIZE,
    "base64_run": "-----END PRIVATE KEY" + "A" * _ADVERSARIAL_SIZE,
}


@pytest.mark.parametrize("name", sorted(_ADVERSARIAL_INPUTS))
def test_private_key_matching_stays_linear_on_adversarial_input(name: str) -> None:
    text = _ADVERSARIAL_INPUTS[name]

    started = time.perf_counter()
    _PII_PRIVATE_KEY_RE.findall(text)
    redacted, _ = redact_text(text)
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0, f"{name}: {elapsed:.2f}s on {len(text)} chars"
    assert "PRIVATE KEY-----" not in redacted


# Rows with many leaves: every leaf a stray END, one unterminated BEGIN followed by many body leaves,
# and many blocks each split across two leaves. Mapping spans onto leaves must stay linear too.
_ADVERSARIAL_LEAF_COUNT = 20_000
_ADVERSARIAL_ROWS = {
    "stray_end_leaves": {"lines": ["-----END PRIVATE KEY-----"] * _ADVERSARIAL_LEAF_COUNT},
    "unterminated_then_body_leaves": {
        "lines": ["-----BEGIN PRIVATE KEY-----"] + ["A" * 64] * _ADVERSARIAL_LEAF_COUNT
    },
    "split_blocks": {
        "lines": [f"{BEGIN_RSA}\n{BODY_1}", f"{BODY_2}\n{END_RSA}"] * (_ADVERSARIAL_LEAF_COUNT // 2)
    },
}


@pytest.mark.parametrize("name", sorted(_ADVERSARIAL_ROWS))
def test_row_level_private_key_redaction_stays_linear_in_the_leaf_count(name: str) -> None:
    row = _ADVERSARIAL_ROWS[name]

    started = time.perf_counter()
    redacted, report = redact_rows([row])
    elapsed = time.perf_counter() - started

    assert elapsed < 3.0, f"{name}: {elapsed:.2f}s on {len(row['lines'])} leaves"
    assert report.redacted_rows == 1
    serialized = json.dumps(redacted)
    assert "PRIVATE KEY-----" not in serialized
    assert "A" * 64 not in serialized
