from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_STUDY = _REPOSITORY_ROOT / "research/ieee-linux-training"
_VALIDATOR = _STUDY / "validate_protocol.py"
_EFFECTIVE_MATRIX = _STUDY / "EXPERIMENT_MATRIX.v1.4.0.json"
_CONTRACTS = _REPOSITORY_ROOT / "docs/contracts"
# Amendment 0004 -> effective matrix 1.4.0 (v7 worker lineage). Reconstructed byte-deterministically
# from the base matrix plus the 0004 manifest.
_EFFECTIVE_HASH = "0ce1fbd425e0401824c3f75f430b72bc4cc51b74e592399cd503a7084c4e593e"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("vp_test", _VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned script.
        [sys.executable, str(_VALIDATOR), *args],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _fresh_candidate(**updates: object) -> dict[str, object]:
    # Under effective 1.4.0 the sealed environment ids are the fresh v7 pair, and every v6 identity is
    # reserved, so a valid fresh runplan candidate binds the exact math-v7/flash-v7 environments and
    # otherwise-unallocated identities.
    candidate: dict[str, object] = {
        "schema_version": "1.0.0",
        "stage": "runplan",
        "environment_ids": [
            "backend-corpus-studio-research-flash-v7",
            "backend-corpus-studio-research-math-v7",
        ],
        "environment_lock_hashes": ["1" * 64, "2" * 64],
        "worker_wheel_sha256": ["3" * 64],
        "plan_ids": ["plan-fresh-flash-v7", "plan-fresh-math-v7"],
        "plan_hashes": ["4" * 64, "5" * 64],
        "execution_configuration_ids": [
            "plan-fresh-flash-v7-execution",
            "plan-fresh-math-v7-execution",
        ],
        "execution_configuration_hashes": ["6" * 64, "7" * 64],
        "run_ids": [],
        "output_paths": [
            "/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/"
            "phase3-qwen25-05b-matched-v7"
        ],
        "artifact_ids": [],
        "evidence_roots": [
            "/mnt/training-nvme/corpusstudio/evidence/production-smoke-matched-v7"
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


# ---- amendment/implementation consistency (defects CI+validator did not previously catch) ---------

def test_effective_matrix_manager_version_matches_implementation() -> None:
    # Protocol version and Environment Manager version are INDEPENDENT identities. The sealed matrix
    # must not preregister a manager version the actual implementation cannot satisfy, so the matrix's
    # manager_version_exact must equal the imported MANAGER_VERSION constant (torch-free import).
    from corpus_studio.platform.environment_manager import MANAGER_VERSION

    matrix = json.loads(_EFFECTIVE_MATRIX.read_text(encoding="utf-8"))
    assert matrix["environment_admission"]["manager_version_exact"] == MANAGER_VERSION


def _resolve_schema_node(node: dict, defs: dict) -> dict:
    # Follow $ref and unwrap nullable anyOf to the concrete object schema.
    seen = 0
    while True:
        seen += 1
        assert seen < 50, "schema $ref cycle"
        if "$ref" in node:
            node = defs[node["$ref"].rsplit("/", 1)[-1]]
            continue
        if "anyOf" in node:
            branches = [b for b in node["anyOf"] if b.get("type") != "null"]
            assert branches, "anyOf had only null branch"
            node = branches[0]
            continue
        return node


def _summary_path_resolves(path: str, schema: dict, defs: dict) -> bool:
    node: dict = schema
    for part in path.split("."):
        node = _resolve_schema_node(node, defs)
        props = node.get("properties")
        if not props or part not in props:
            return False
        node = props[part]
    return True


def test_required_paper_telemetry_paths_resolve_against_schemas() -> None:
    # Every declared summary path must resolve against the committed RunTelemetrySummary schema, and
    # every raw event-metrics field against the committed RunEvent -> EventMetrics schema. This is the
    # guard that would have caught the pseudo-paths (`throughput.paper_performance_complete`, etc.).
    summary_schema = json.loads((_CONTRACTS / "RunTelemetrySummary.schema.json").read_text())
    summary_defs = summary_schema.get("$defs", {})
    event_schema = json.loads((_CONTRACTS / "RunEvent.schema.json").read_text())
    event_metrics_props = event_schema["$defs"]["EventMetrics"]["properties"]

    matrix = json.loads(_EFFECTIVE_MATRIX.read_text(encoding="utf-8"))
    wsa = matrix["worker_success_admission"]

    for path in wsa["required_paper_telemetry_fields"]:
        assert _summary_path_resolves(path, summary_schema, summary_defs), (
            f"summary evidence path does not resolve against RunTelemetrySummary: {path}"
        )
    for field in wsa["required_paper_event_metrics_fields"]:
        assert field in event_metrics_props, (
            f"event evidence field does not resolve against EventMetrics: {field}"
        )
    # The corrected token evidence is actually declared where it belongs.
    assert "step.nonpadding_tokens_per_second" in wsa["required_paper_telemetry_fields"]
    assert "step.supervised_tokens_per_second" in wsa["required_paper_telemetry_fields"]
    assert "completeness.scientific_throughput_complete" in wsa["required_paper_telemetry_fields"]
    assert "completeness.paper_performance_complete" in wsa["required_paper_telemetry_fields"]
    assert set(wsa["required_paper_event_metrics_fields"]) == {
        "nonpadding_tokens", "supervised_tokens", "observed_microbatches",
    }
    # The paper-performance promotion rule is prospectively required.
    assert wsa["paper_performance_complete_required_for_paper_promotion"] is True
    assert wsa["token_throughput_validity_required_for_paper_performance"] is True


def test_pseudo_paths_do_not_resolve_against_the_schema() -> None:
    # Negative control: the exact bad paths from the earlier draft, plus a typo and an unknown prefix,
    # must all fail to resolve (proving the resolver is not vacuously true).
    schema = json.loads((_CONTRACTS / "RunTelemetrySummary.schema.json").read_text())
    defs = schema.get("$defs", {})
    for bad in (
        "throughput.paper_performance_complete",   # unknown prefix
        "throughput.scientific_throughput_complete",
        "step.nonpadding_tokens",                  # raw count is NOT on the summary
        "step.supervised_tokens",
        "completeness.paper_performance_complet",   # typo
        "step.does_not_exist",
    ):
        assert not _summary_path_resolves(bad, schema, defs), f"pseudo-path unexpectedly resolved: {bad}"


def test_amendment_authored_at_is_not_in_the_future() -> None:
    vp = _load_validator_module()
    manifest = json.loads(
        (_STUDY / "amendments"
         / "0004-2026-07-16-v7-worker-lineage-token-throughput-observer.manifest.json").read_text()
    )
    # The committed amendment validates against the real clock.
    vp._validate_authored_at(manifest)
    # A future timestamp is rejected regardless of the clock.
    fixed_now = datetime(2026, 7, 16, 3, 53, 36, tzinfo=timezone.utc)
    with pytest.raises(vp.ProtocolValidationError, match="in the future"):
        vp._validate_authored_at({"authored_at": "2030-01-01T00:00:00Z"}, now=fixed_now)
    # A malformed timestamp is rejected.
    with pytest.raises(vp.ProtocolValidationError, match="ISO-8601"):
        vp._validate_authored_at({"authored_at": "2026-07-16 03:53:36"}, now=fixed_now)


def test_effective_matrix_reconstruction_is_byte_deterministic() -> None:
    vp = _load_validator_module()
    base = vp._load_yaml(_STUDY / "EXPERIMENT_MATRIX.yaml")
    manifest = json.loads(
        (_STUDY / "amendments"
         / "0004-2026-07-16-v7-worker-lineage-token-throughput-observer.manifest.json").read_text()
    )
    first = json.dumps(vp._build_expected_effective(base, manifest), indent=2, ensure_ascii=False)
    second = json.dumps(vp._build_expected_effective(base, manifest), indent=2, ensure_ascii=False)
    assert first == second
    # And the committed file matches the reconstruction byte-for-byte (plus one trailing LF).
    committed = _EFFECTIVE_MATRIX.read_bytes()
    assert committed == (first + "\n").encode("utf-8")
