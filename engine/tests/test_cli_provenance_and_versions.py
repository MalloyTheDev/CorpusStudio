"""CLI-level coverage for four Typer handlers whose library cores are tested but whose command
surface (arg parsing, allowlist merge, JSON output shape, exit codes, error branches) was not: #506.

provenance-gate / run-provenance / dataset-tokens / dataset-version-gc."""

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app

# stdout must be JSON-only, so keep the human/error text on a separate stderr stream. click < 8.2 needs
# mix_stderr=False; click >= 8.2 removed the kwarg and separates the streams by default.
try:
    runner = CliRunner(mix_stderr=False)
except TypeError:  # pragma: no cover - depends on the installed click version
    runner = CliRunner()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _new_project(tmp_path: Path) -> Path:
    result = runner.invoke(app, ["new-project", "proj", "Proj", "instruction", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    return tmp_path / "proj"


# ---- provenance-gate --------------------------------------------------------


def test_provenance_gate_emits_json_and_stays_exit_zero(tmp_path):
    ds = tmp_path / "d.jsonl"
    _write_rows(ds, [{"prompt": "p", "response": "r", "meta": {"teacher": "z-ai/glm-5.2"}}])
    result = runner.invoke(app, ["provenance-gate", str(ds)])
    assert result.exit_code == 0
    assert isinstance(json.loads(result.stdout), dict)  # the verdict IS the JSON report on stdout


def test_provenance_gate_allowlist_merge_changes_the_verdict(tmp_path):
    # A restricted teacher quarantines; --allow-teacher must merge into the allowlist and move it out
    # (the allowlist-merge branch the CLI owns, not the library core).
    ds = tmp_path / "d.jsonl"
    _write_rows(ds, [{"prompt": "p", "response": "r", "meta": {"teacher": "anthropic/claude"}}])
    blocked = runner.invoke(app, ["provenance-gate", str(ds)])
    allowed = runner.invoke(app, ["provenance-gate", str(ds), "--allow-teacher", "anthropic/claude"])
    assert blocked.exit_code == 0 and allowed.exit_code == 0
    assert json.loads(blocked.stdout) != json.loads(allowed.stdout)


def test_provenance_gate_rejects_malformed_jsonl(tmp_path):
    ds = tmp_path / "bad.jsonl"
    ds.write_text("{not valid json\n", encoding="utf-8")
    result = runner.invoke(app, ["provenance-gate", str(ds)])
    assert result.exit_code == 1  # the read_jsonl error branch, classified not crashed


# ---- run-provenance ---------------------------------------------------------


def test_run_provenance_emits_json_for_a_project(tmp_path):
    proj = _new_project(tmp_path)
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"base_model": "demo", "seq_len": 512}), encoding="utf-8")
    result = runner.invoke(app, ["run-provenance", str(proj), str(config)])
    assert result.exit_code == 0
    assert isinstance(json.loads(result.stdout), dict)


def test_run_provenance_is_best_effort_on_a_missing_config(tmp_path):
    proj = _new_project(tmp_path)
    result = runner.invoke(app, ["run-provenance", str(proj), str(tmp_path / "nope.json")])
    assert result.exit_code == 0  # missing config leaves that field null rather than failing
    assert isinstance(json.loads(result.stdout), dict)


# ---- dataset-tokens ---------------------------------------------------------


def test_dataset_tokens_fails_closed_without_the_train_extra(tmp_path, monkeypatch):
    # A None entry in sys.modules makes `from transformers import ...` raise ImportError regardless of
    # whether the [train] extra is installed, so this pins the graceful exit-2 branch deterministically.
    monkeypatch.setitem(sys.modules, "transformers", None)
    ds = tmp_path / "d.jsonl"
    _write_rows(ds, [{"prompt": "p", "response": "r"}])
    result = runner.invoke(app, ["dataset-tokens", str(ds), "--base-model", "x"])
    assert result.exit_code == 2
    assert "[train]" in result.stderr


# ---- dataset-version-gc -----------------------------------------------------


def test_dataset_version_gc_dry_run_emits_json(tmp_path):
    proj = _new_project(tmp_path)
    result = runner.invoke(app, ["dataset-version-gc", str(proj), "--dry-run", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert {"scanned_rows", "pruned_rows", "kept_rows"} <= set(payload)


def test_dataset_version_gc_aborts_when_a_manifest_is_unreadable(tmp_path, monkeypatch):
    proj = _new_project(tmp_path)
    import corpus_studio.versions.gc as gc_module

    def _boom(*_args, **_kwargs):
        raise OSError("a version manifest could not be read")

    monkeypatch.setattr(gc_module, "gc_row_store", _boom)
    result = runner.invoke(app, ["dataset-version-gc", str(proj)])
    assert result.exit_code == 1  # GC aborts rather than risk deleting referenced rows
