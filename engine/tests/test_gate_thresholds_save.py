"""Tests for saving project-local gate thresholds (issue #198)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.gates.models import (
    GateThresholds,
    load_gate_thresholds,
    save_gate_thresholds,
)

_runner = CliRunner()


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    saved = GateThresholds(
        max_exact_duplicates=5,
        block_normalized_duplicates=True,
        min_eval_pass_rate=0.8,
        min_chat_turns=3,
        max_chat_turns=12,
    )
    path = save_gate_thresholds(tmp_path, saved)

    assert path.name == "gate_thresholds.json"
    loaded = load_gate_thresholds(tmp_path)
    assert loaded.max_exact_duplicates == 5
    assert loaded.block_normalized_duplicates is True
    assert loaded.min_eval_pass_rate == 0.8
    assert loaded.min_chat_turns == 3
    assert loaded.max_chat_turns == 12


def test_out_of_range_values_are_rejected_by_the_model() -> None:
    with pytest.raises(ValidationError):
        GateThresholds(min_eval_pass_rate=2.0)  # le=1
    with pytest.raises(ValidationError):
        GateThresholds(max_exact_duplicates=-1)  # ge=0


def test_gate_thresholds_set_cli_writes_valid_and_rejects_invalid(tmp_path: Path) -> None:
    ok = _runner.invoke(
        app,
        ["gate-thresholds-set", str(tmp_path), "--values-json", '{"max_exact_duplicates": 7}'],
    )
    assert ok.exit_code == 0, ok.output
    assert load_gate_thresholds(tmp_path).max_exact_duplicates == 7

    bad = _runner.invoke(
        app,
        ["gate-thresholds-set", str(tmp_path), "--values-json", '{"min_eval_pass_rate": 2.0}'],
    )
    assert bad.exit_code == 1
    # The pre-existing valid file is untouched by the rejected write.
    assert load_gate_thresholds(tmp_path).max_exact_duplicates == 7
