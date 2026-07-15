from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_VALIDATOR = _REPOSITORY_ROOT / "research/ieee-linux-training/validate_protocol.py"
_EFFECTIVE_HASH = "731101c9ee81bd93f9acabb0433788180b86bcec3d78a7af03892df7af41f4f9"


def _run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned script.
        [sys.executable, str(_VALIDATOR), *args],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _fresh_candidate(**updates: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "schema_version": "1.0.0",
        "stage": "runplan",
        "environment_ids": [
            "backend-corpus-studio-research-flash-v4",
            "backend-corpus-studio-research-math-v4",
        ],
        "environment_lock_hashes": ["1" * 64, "2" * 64],
        "worker_wheel_sha256": ["3" * 64],
        "plan_ids": ["plan-fresh-flash-v4", "plan-fresh-math-v4"],
        "plan_hashes": ["4" * 64, "5" * 64],
        "execution_configuration_ids": [
            "plan-fresh-flash-v4-execution",
            "plan-fresh-math-v4-execution",
        ],
        "execution_configuration_hashes": ["6" * 64, "7" * 64],
        "run_ids": [],
        "output_paths": [
            "/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/"
            "phase3-qwen25-05b-matched-v4"
        ],
        "artifact_ids": [],
        "evidence_roots": [
            "/mnt/training-nvme/corpusstudio/evidence/production-smoke-matched-v4"
        ],
    }
    candidate.update(updates)
    return candidate


def test_effective_research_protocol_is_exactly_reconstructible() -> None:
    result = _run_validator()
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "valid"
    assert payload["effective_matrix_sha256"] == _EFFECTIVE_HASH


def test_research_protocol_rejects_reserved_identity_reuse(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-identities.json"
    candidate.write_text(
        json.dumps(
            _fresh_candidate(
                plan_ids=[
                    "plan-019f644b-a3c2-7373-abc0-39a0f7d753eb",
                    "plan-fresh-math-v4",
                ]
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run_validator("--candidate-identities", str(candidate))
    assert result.returncode == 1
    assert "candidate reuses reserved plan_ids" in result.stderr


def test_research_protocol_accepts_fresh_candidate_identity(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-identities.json"
    candidate.write_text(
        json.dumps(
            _fresh_candidate(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run_validator("--candidate-identities", str(candidate))
    assert result.returncode == 0, result.stderr


def test_research_protocol_rejects_output_descendant_reuse(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-identities.json"
    candidate.write_text(
        json.dumps(
            _fresh_candidate(
                output_paths=[
                    "/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/"
                    "phase3-qwen25-05b-matched-v3/reused-child"
                ]
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run_validator("--candidate-identities", str(candidate))
    assert result.returncode == 1
    assert "candidate reuses reserved output_paths" in result.stderr


def test_research_protocol_rejects_incomplete_candidate_identity(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-identities.json"
    candidate.write_text("{}\n", encoding="utf-8")
    result = _run_validator("--candidate-identities", str(candidate))
    assert result.returncode == 1
    assert "candidate identity fields do not match" in result.stderr


def test_research_protocol_rejects_noncanonical_hash(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-identities.json"
    candidate.write_text(
        json.dumps(
            _fresh_candidate(
                plan_hashes=["A" * 64, "B" * 64],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run_validator("--candidate-identities", str(candidate))
    assert result.returncode == 1
    assert "lowercase SHA-256" in result.stderr


def test_research_protocol_rejects_noncanonical_path(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-identities.json"
    candidate.write_text(
        json.dumps(
            _fresh_candidate(
                output_paths=[
                    "/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/../alias-v4"
                ],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run_validator("--candidate-identities", str(candidate))
    assert result.returncode == 1
    assert "canonical absolute POSIX paths" in result.stderr
