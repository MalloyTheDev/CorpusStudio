"""Static screening for the mode-3 custom-block path: the analyzer rejects the obvious dangerous surface
fail-closed, requires the declared entry class, and records a content-addressed, self-consistent
ModelCodeVettingReport. A static screen is a pre-screen, NOT a safety proof - these tests pin the screen's
behavior, not a containment guarantee."""

import hashlib
import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.contracts import ModelCodeVettingReport, VettingFinding
from corpus_studio.platform.custom_code_vetting import build_report, vet_source

_runner = CliRunner()

CLEAN = """
from __future__ import annotations

import torch
import torch.nn as nn


class MyDecoderForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids):
        return self.embed(input_ids)
"""


def _codes(findings):
    return {f.code for f in findings}


def _errors(findings):
    return {f.code for f in findings if f.severity == "error"}


# ---- the static screen ------------------------------------------------------------------------------


def test_a_clean_block_is_admitted():
    findings = vet_source(CLEAN, entry_symbol="MyDecoderForCausalLM")
    assert _errors(findings) == set()


@pytest.mark.parametrize(
    "snippet, code",
    [
        ("import os\n", "forbidden-import"),
        ("import subprocess\n", "forbidden-import"),
        ("import pickle\n", "forbidden-import"),
        ("import numpy\n", "import-not-allowlisted"),
        ("from . import helpers\n", "relative-import"),
        ("from torch import *\n", "wildcard-import"),
        ("import torch\nx = eval('1')\n", "forbidden-call"),
        ("import torch\ny = open('/etc/passwd')\n", "forbidden-call"),
        ("import torch\nz = ().__class__.__bases__\n", "reflective-escape"),
        ("import torch\ng = (0).__class__.__subclasses__\n", "reflective-escape"),
        ("import torch\nprint('side effect')\n", "module-side-effect"),
        ("def f(:\n", "syntax-error"),
    ],
)
def test_dangerous_surface_is_rejected(snippet, code):
    findings = vet_source(snippet + "\nclass C:\n    pass\n", entry_symbol="C")
    assert code in _errors(findings), findings


def test_missing_entry_class_is_an_error():
    findings = vet_source("import torch\n", entry_symbol="Nope")
    assert "entry-class-missing" in _errors(findings)


def test_entry_class_without_a_base_is_a_warning_not_an_error():
    findings = vet_source("class Bare:\n    pass\n", entry_symbol="Bare")
    assert "entry-class-no-base" in _codes(findings)
    assert _errors(findings) == set()  # a warning does not reject


# ---- the report (content-addressed, self-consistent) -----------------------------------------------


def test_build_report_pins_the_bundle_hash_and_admits_clean_code():
    raw = CLEAN.encode("utf-8")
    report = build_report(raw, entry_symbol="MyDecoderForCausalLM")
    assert report.verdict == "admitted"
    assert report.bundle_sha256 == hashlib.sha256(raw).hexdigest()
    assert report.analyzer_version and report.interface_version == "custom_decoder_v1"


def test_build_report_rejects_dangerous_code():
    report = build_report(b"import os\nclass C:\n    pass\n", entry_symbol="C")
    assert report.verdict == "rejected"
    assert any(f.severity == "error" for f in report.findings)


def test_build_report_refuses_an_unsupported_interface():
    with pytest.raises(ValueError, match="not supported"):
        build_report(CLEAN.encode("utf-8"), entry_symbol="MyDecoderForCausalLM", interface_version="v99")


def test_report_verdict_must_be_consistent_with_findings():
    err = VettingFinding(severity="error", code="x", message="m")
    with pytest.raises(ValidationError):
        ModelCodeVettingReport(
            analyzer_version="1.0.0", bundle_sha256="a" * 64, entry_symbol="C",
            interface_version="custom_decoder_v1", verdict="admitted", findings=[err],
        )
    with pytest.raises(ValidationError):
        ModelCodeVettingReport(
            analyzer_version="1.0.0", bundle_sha256="a" * 64, entry_symbol="C",
            interface_version="custom_decoder_v1", verdict="rejected", findings=[],
        )


# ---- the vet-model-code CLI (fail-closed) ----------------------------------------------------------


def test_cli_admits_a_clean_bundle_and_writes_the_report(tmp_path):
    bundle = tmp_path / "block.py"
    bundle.write_text(CLEAN, encoding="utf-8")
    out = tmp_path / "report.json"
    result = _runner.invoke(
        app,
        ["vet-model-code", str(bundle), "--entry-symbol", "MyDecoderForCausalLM", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "admitted"
    assert report["bundle_sha256"] == hashlib.sha256(CLEAN.encode("utf-8")).hexdigest()


def test_cli_rejects_a_dangerous_bundle_fail_closed(tmp_path):
    bundle = tmp_path / "evil.py"
    bundle.write_text("import os\nclass C:\n    pass\n", encoding="utf-8")
    result = _runner.invoke(app, ["vet-model-code", str(bundle), "--entry-symbol", "C"])
    assert result.exit_code == 2


def test_cli_reports_an_unreadable_bundle(tmp_path):
    missing = tmp_path / "nope.py"
    result = _runner.invoke(app, ["vet-model-code", str(missing), "--entry-symbol", "C"])
    assert result.exit_code == 2
    assert "cannot read bundle" in result.output


# ---- the sealed CustomModelCodeSpec (slice 1b contract) --------------------------------------------


def _pinned(ref_id):
    from corpus_studio.platform.common import HashRef, Ref

    return Ref(id=ref_id, hash=HashRef(value="a" * 64))


def test_custom_model_code_spec_requires_pinned_refs():
    from corpus_studio.platform.common import Ref
    from corpus_studio.platform.contracts import CustomModelCodeSpec

    ok = CustomModelCodeSpec(
        code_bundle_ref=_pinned("bundle"), entry_symbol="C", interface_version="custom_decoder_v1",
        vetting_ref=_pinned("vet"), vetting_verdict="admitted",
    )
    assert ok.trust_remote_code is False
    with pytest.raises(ValidationError):  # an UNPINNED code bundle cannot be sealed
        CustomModelCodeSpec(
            code_bundle_ref=Ref(id="bundle"), entry_symbol="C", interface_version="custom_decoder_v1",
            vetting_ref=_pinned("vet"), vetting_verdict="admitted",
        )


def test_continued_init_forbids_custom_code():
    from corpus_studio.platform.contracts import CustomModelCodeSpec, ModelInitializationSpec

    cc = CustomModelCodeSpec(
        code_bundle_ref=_pinned("b"), entry_symbol="C", interface_version="custom_decoder_v1",
        vetting_ref=_pinned("v"), vetting_verdict="admitted",
    )
    with pytest.raises(ValidationError):  # continued init derives its architecture from the checkpoint
        ModelInitializationSpec(mode="continued", source_checkpoint_ref=_pinned("ck"), custom_code=cc)
