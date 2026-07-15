from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPOSITORY_ROOT / "research" / "ieee-linux-training"
INDEX_PATH = RESEARCH_ROOT / "amendments" / "index.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_index() -> dict[str, Any]:
    with INDEX_PATH.open("rb") as stream:
        value = json.load(stream)
    assert isinstance(value, dict)
    return value


def _research_path(relative: str) -> Path:
    declared = Path(relative)
    assert not declared.is_absolute()
    resolved = (RESEARCH_ROOT / declared).resolve()
    assert resolved.is_relative_to(RESEARCH_ROOT.resolve())
    return resolved


def test_ieee_protocol_base_files_remain_frozen() -> None:
    index = _load_index()
    base_files = index["base_files"]

    assert index["study_id"] == "cs-ieee-linux-training-v1"
    assert index["base_protocol_version"] == "1.0.0"
    assert [item["path"] for item in base_files] == [
        "README.md",
        "PROTOCOL.md",
        "HYPOTHESES.md",
        "EXPERIMENT_MATRIX.yaml",
        "METRICS.md",
        "FAILURE_TAXONOMY.md",
        "REPRODUCIBILITY.md",
    ]
    for item in base_files:
        path = _research_path(item["path"])
        assert path.is_file()
        assert _sha256(path) == item["sha256"]

    matrix = (RESEARCH_ROOT / "EXPERIMENT_MATRIX.yaml").read_text(encoding="utf-8")
    assert 'protocol_version: "1.0.0"' in matrix
    assert "sequence_lengths:\n    - 512\n    - 1024\n    - 2048\n    - 3072\n    - 4096" in matrix
    assert "seed: 41999\n  data_seed: 51999" in matrix


def test_ieee_protocol_amendment_chain_is_ordered_and_hash_bound() -> None:
    index = _load_index()
    previous_version = index["base_protocol_version"]

    for expected_sequence, amendment in enumerate(index["amendments"], start=1):
        assert amendment["sequence"] == expected_sequence
        assert amendment["previous_protocol_version"] == previous_version
        assert amendment["status"] == "prospective"
        assert amendment["paper_matrix_result"] is False
        assert amendment["execution_authorized"] is False
        path = _research_path(amendment["path"])
        assert path.is_file()
        assert _sha256(path) == amendment["sha256"]
        previous_version = amendment["effective_protocol_version"]

    assert previous_version == index["effective_protocol_version"]


def test_corrected_smoke_bindings_remain_distinct_and_matched() -> None:
    amendment = _load_index()["amendments"][0]
    math = amendment["execution_paths"]["first-party-math"]
    flash = amendment["execution_paths"]["first-party-flash"]

    assert amendment["scope"] == "phase3-production-path-smoke-pair"
    assert amendment["repository_commit_policy"] == (
        "final-merged-amendment-commit-per-runplan"
    )
    assert amendment["worker"]["source_commit"] == (
        "16ef6e95722ec3988ee8826b45333c9356ef76f9"
    )
    assert amendment["input_scope"] == {
        "model_repository": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "dataset_id": "pipeline-smoke-fixture-v2",
        "dataset_sha256": (
            "a322b1059709a30c4f927b087e0e655724d6e2a06873175b71d03073a17fa289"
        ),
        "dataset_rows": 8,
        "user_500_output_corpus_available": False,
        "seven_billion_parameter_workloads_available": False,
    }
    assert math["environment_id"] == "backend-corpus-studio-research-math-v3"
    assert flash["environment_id"] == "backend-corpus-studio-research-flash-v3"
    assert math["lock_hash"] != flash["lock_hash"]
    assert math["attention_kernel"] == "torch_sdpa_math"
    assert flash["attention_kernel"] == "torch_sdpa_flash"
    assert (
        math["flash_sdp_enabled"],
        math["memory_efficient_sdp_enabled"],
        math["math_sdp_enabled"],
    ) == (False, False, True)
    assert (
        flash["flash_sdp_enabled"],
        flash["memory_efficient_sdp_enabled"],
        flash["math_sdp_enabled"],
    ) == (True, False, False)
    assert amendment["matched_environment_result"] == (
        "MATCHED_FOR_ATTENTION_KERNEL_STUDY"
    )
