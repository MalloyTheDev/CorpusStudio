"""End-to-end smoke test: drive the *installed* CLI as a real subprocess.

Every other test exercises the engine in-process (typer's CliRunner or direct
calls). This one runs the shipped console entry point (`python -m
corpus_studio.cli`, the same callable as the `corpus-studio` script) in a fresh
process through a full create -> validate -> quality -> export loop over real
files. It proves the packaged artifact actually launches — imports resolve,
typer dispatch works, UTF-8 stdio round-trips — which an in-process test can't.

No network or model backend is required, so it is safe to run in CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "corpus_studio.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd) if cwd else None,
    )


def test_installed_cli_runs_the_full_loop_as_a_subprocess(tmp_path: Path):
    # 1. Create a project scaffold on disk.
    created = _run("new-project", "e2e", "E2E Project", "instruction", "--root", str(tmp_path))
    assert created.returncode == 0, created.stderr
    assert (tmp_path / "e2e").is_dir()

    # A small valid instruction dataset, including non-ASCII so we prove UTF-8
    # survives the desktop-less subprocess path end to end.
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"instruction": "Explain recursion.", "input": "", '
        '"output": "A function calls itself."}\n'
        '{"instruction": "Greet the user.", "input": "", '
        '"output": "안녕하세요 — hello"}\n',
        encoding="utf-8",
    )

    # 2. Validate — exit 0 and a valid report.
    validated = _run("validate", str(dataset), "instruction")
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["valid"] is True

    # 3. Quality — a real report over the saved rows.
    quality = _run("quality", str(dataset))
    assert quality.returncode == 0, quality.stderr
    assert json.loads(quality.stdout)["example_count"] == 2

    # 4. Export (with cleaning) — writes the model-ready file.
    out = tmp_path / "export.jsonl"
    exported = _run("export", str(dataset), str(out), "instruction", "--dedupe")
    assert exported.returncode == 0, exported.stderr
    assert out.is_file()

    written = out.read_text(encoding="utf-8")
    lines = [line for line in written.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "안녕하세요" in written  # UTF-8 round-tripped through the subprocess


def test_installed_cli_reports_a_nonzero_exit_on_invalid_data(tmp_path: Path):
    # The shipped CLI must signal failure with an exit code (what a desktop or a
    # shell pipeline relies on), not just print an error.
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"instruction": "missing output"}\n', encoding="utf-8")

    result = _run("validate", str(bad), "instruction")
    assert result.returncode == 1
    assert json.loads(result.stdout)["valid"] is False
