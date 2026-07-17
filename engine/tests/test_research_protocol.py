from __future__ import annotations

import copy
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
_EFFECTIVE_MATRIX = _STUDY / "EXPERIMENT_MATRIX.v1.5.0.json"
_CONTRACTS = _REPOSITORY_ROOT / "docs/contracts"
# Amendment 0005 -> effective matrix 1.5.0 (v8 worker lineage; Environment Manager 1.4.0 + exact
# per-lineage floor binding). Reconstructed byte-deterministically from the base matrix plus the 0005
# manifest.
_EFFECTIVE_HASH = "2939a8cb78aff658561eed7ad2f2b640bb0c4257b882b6bf5fde753cab010f87"


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
    # Under effective 1.5.0 the sealed environment ids are the fresh v8 pair, and every v1-v7 identity
    # is reserved, so a valid fresh runplan candidate binds the exact math-v8/flash-v8 environments and
    # otherwise-unallocated identities.
    candidate: dict[str, object] = {
        "schema_version": "1.0.0",
        "stage": "runplan",
        "environment_ids": [
            "backend-corpus-studio-research-flash-v8",
            "backend-corpus-studio-research-math-v8",
        ],
        "environment_lock_hashes": ["1" * 64, "2" * 64],
        "worker_wheel_sha256": ["3" * 64],
        "plan_ids": ["plan-fresh-flash-v8", "plan-fresh-math-v8"],
        "plan_hashes": ["4" * 64, "5" * 64],
        "execution_configuration_ids": [
            "plan-fresh-flash-v8-execution",
            "plan-fresh-math-v8-execution",
        ],
        "execution_configuration_hashes": ["6" * 64, "7" * 64],
        "run_ids": [],
        "output_paths": [
            "/mnt/training-nvme/corpusstudio/runs/ieee-linux-training/"
            "phase3-qwen25-7b-feasibility-v8"
        ],
        "artifact_ids": [],
        "evidence_roots": [
            "/mnt/training-nvme/corpusstudio/evidence/seven-b-feasibility-v8"
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
                    # A now-reserved instantiated v7 plan id (added to RESERVED_IDENTITIES.v5).
                    "plan-019f6944-782a-7658-bc95-672994f9c08a",
                    "plan-fresh-math-v8",
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
         / "0005-2026-07-16-v8-manager-1.4-floor-binding-lineage.manifest.json").read_text()
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
         / "0005-2026-07-16-v8-manager-1.4-floor-binding-lineage.manifest.json").read_text()
    )
    first = json.dumps(vp._build_expected_effective(base, manifest), indent=2, ensure_ascii=False)
    second = json.dumps(vp._build_expected_effective(base, manifest), indent=2, ensure_ascii=False)
    assert first == second
    # And the committed file matches the reconstruction byte-for-byte (plus one trailing LF).
    committed = _EFFECTIVE_MATRIX.read_bytes()
    assert committed == (first + "\n").encode("utf-8")


# ---- amendment 0005 semantic corrections: the 7B feasibility ladder is unambiguous ----------------
#
# These guard the four preregistration ambiguities the 0005 semantic correction closed: the model
# reference must resolve (not be a repository path), the feasibility fixture carries its own identity
# contract distinct from the private corpus, the per-kernel run count is explicit, and the rung success
# criteria are exact. Each mutates the committed effective matrix and asserts the validator rejects it,
# so none of the gates is vacuous.


def _effective_matrix() -> dict:
    return json.loads(_EFFECTIVE_MATRIX.read_text(encoding="utf-8"))


def _ladder(effective: dict) -> dict:
    return effective["seven_b_native_linux_feasibility_ladder"]


def test_committed_effective_matrix_passes_the_ladder_and_lineage_gates() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    # The committed matrix passes every semantic gate unmodified.
    vp._validate_seven_b_feasibility_ladder(effective)
    vp._validate_lineage_change_classification(effective)
    vp._validate_math_terminal_flash_eligibility(effective)


def test_feasibility_model_reference_resolves_and_rejects_ambiguity() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    ladder = _ladder(effective)
    # It binds a resolvable short id (not a repository path) whose repository matches models[].
    assert ladder["model_id"] == "qwen2.5-7b-instruct"
    assert ladder["model_source_repository"] == "Qwen/Qwen2.5-7B-Instruct"
    resolved = [m for m in effective["models"] if m["id"] == ladder["model_id"]]
    assert len(resolved) == 1
    assert resolved[0]["source_repository"] == ladder["model_source_repository"]

    # A repository-shaped model_id is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["model_id"] = "Qwen/Qwen2.5-7B-Instruct"
    with pytest.raises(vp.ProtocolValidationError, match="not a repository path"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # An unknown model_id is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["model_id"] = "qwen2.5-99b-instruct"
    with pytest.raises(vp.ProtocolValidationError, match="resolve to exactly one"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # A mismatched source repository is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["model_source_repository"] = "Qwen/Qwen2.5-7B"
    with pytest.raises(vp.ProtocolValidationError, match="does not match the resolved model"):
        vp._validate_seven_b_feasibility_ladder(bad)


def test_feasibility_fixture_identity_is_bound_and_distinct_from_private_corpus() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    fixture = _ladder(effective)["feasibility_fixture"]
    # The fixture is explicitly not the primary private corpus and requires its identity before planning.
    assert fixture["is_primary_private_corpus"] is False
    assert fixture["distinct_from_primary_private_corpus"] is True
    assert fixture["content_read_or_frozen_in_this_amendment"] is False
    for sha_field in (
        "content_sha256",
        "rendered_examples_sha256",
        "tokenizer_content_sha256",
        "chat_template_sha256",
    ):
        assert fixture[sha_field] == "required-before-planning"
    assert fixture["license_evidence_required"] is True
    assert fixture["fixed_row_order"] is True
    assert fixture["packing"] is False
    assert fixture["truncation"] is False

    # Marking the fixture as the private corpus is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["feasibility_fixture"]["is_primary_private_corpus"] = True
    with pytest.raises(vp.ProtocolValidationError, match="distinct from the primary private corpus"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # Enabling truncation or packing on the fixture is refused.
    for field, message in (("truncation", "disable truncation"), ("packing", "disable packing")):
        bad = copy.deepcopy(effective)
        _ladder(bad)["feasibility_fixture"][field] = True
        with pytest.raises(vp.ProtocolValidationError, match=message):
            vp._validate_seven_b_feasibility_ladder(bad)

    # A non-sentinel, non-SHA fixture content hash is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["feasibility_fixture"]["content_sha256"] = "tbd"
    with pytest.raises(vp.ProtocolValidationError, match="required-before-planning.*or a SHA-256"):
        vp._validate_seven_b_feasibility_ladder(bad)


def test_feasibility_fixed_configuration_is_bound_exactly() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    config = _ladder(effective)["fixed_ladder_configuration"]
    assert config["sequence_length_rungs"] == [512, 1024, 2048, 3072, 4096]
    assert config["rung_order"] == "ascending"
    assert config["microbatch"] == 1
    assert config["gradient_accumulation"] == 1
    assert config["bounded_optimizer_steps"] == 12
    assert config["offload"] == "none"
    assert config["automatic_workload_retry_count"] == 0
    assert config["math_runs"] == "first-once"
    assert config["flash_runs"] == "at-most-once"

    # Each controlling scalar is bound exactly (not by nonempty prose): a mutation is refused.
    for field, bad_value, message in (
        ("sequence_length_rungs", [512, 1024, 2048, 4096], "rungs must be exactly"),
        ("sequence_length_rungs", [4096, 3072, 2048, 1024, 512], "rungs must be exactly"),
        ("microbatch", 2, "microbatch must be 1"),
        ("gradient_accumulation", 4, "gradient_accumulation must be 1"),
        ("bounded_optimizer_steps", 24, "bounded_optimizer_steps must be 12"),
        ("offload", "cpu", "offload must be 'none'"),
        ("automatic_workload_retry_count", 1, "automatic_workload_retry_count must be 0"),
    ):
        bad = copy.deepcopy(effective)
        _ladder(bad)["fixed_ladder_configuration"][field] = bad_value
        with pytest.raises(vp.ProtocolValidationError, match=message):
            vp._validate_seven_b_feasibility_ladder(bad)


def _mapping(effective: dict) -> dict:
    return effective["math_terminal_flash_eligibility"]


def test_feasibility_flash_eligibility_references_the_grounded_mapping() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    flash = _ladder(effective)["flash_eligibility"]
    # Flash runs after math success OR when the math FailureRecord taxonomy/stage matches the grounded
    # top-level mapping; it is a scheduling decision, not a math-kernel causation claim.
    assert flash["flash_condition"] == (
        "run-after-math-success-or-mapped-clean-math-terminal-taxonomy-and-stage"
    )
    assert flash["run_when_math_succeeds"] is True
    assert flash["run_when_math_failure_matches_mapped_terminal_taxonomy_and_stage"] is True
    assert flash["terminal_taxonomy_mapping"] == "math_terminal_flash_eligibility"
    assert flash["confirmed_forced_math_no_fallback_required"] is True
    assert flash["decision_is_scheduling_eligibility_not_math_kernel_causation"] is True
    assert flash["withheld_flash_status"] == "NOT_RUN"
    assert set(flash["clean_math_specific_failure_preconditions"]) == {
        "shared_preparation_passed",
        "declared_math_kernel_selected_no_fallback",
        "evidence_remained_valid",
        "process_terminated_cleanly",
        "gpu_memory_released",
        "environment_health_and_drift_checks_passed",
    }
    # The invented terminal classes are gone from the whole ladder.
    ladder_blob = json.dumps(_ladder(effective))
    assert "kernel_specific_oom" not in ladder_blob
    assert "kernel_specific_timeout" not in ladder_blob

    # Not referencing the grounded mapping is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["flash_eligibility"]["terminal_taxonomy_mapping"] = "invented_mapping"
    with pytest.raises(vp.ProtocolValidationError, match="reference the math_terminal_flash_eligibility"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # Reintroducing an invented terminal class anywhere in the ladder is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["flash_eligibility"]["note"] = "kernel_specific_oom"
    with pytest.raises(vp.ProtocolValidationError, match="invented terminal class must not reappear"):
        vp._validate_seven_b_feasibility_ladder(bad)


def test_math_terminal_flash_eligibility_is_grounded_in_live_contracts() -> None:
    # The declared taxonomy/stage snapshots must equal the LIVE runtime enums exactly (torch-free
    # import), so the mapping is grounded in the existing FailureRecord contracts, not invented classes.
    from corpus_studio.platform.contracts import FailureRecord
    from corpus_studio.platform.enums import FailureTaxonomy, StageMarker

    effective = _effective_matrix()
    mapping = _mapping(effective)
    assert mapping["known_failure_taxonomy"] == [e.value for e in FailureTaxonomy]
    assert mapping["known_stage_markers"] == [e.value for e in StageMarker]
    # FailureRecord localizes a failure only by taxonomy + stage (no attention-kernel/execution field),
    # which is exactly why a generic TIMEOUT cannot be proven math-attention-specific.
    assert "taxonomy" in FailureRecord.model_fields
    assert "stage" in FailureRecord.model_fields
    assert not ({"attention_backend", "effective_backend", "kernel"} & set(FailureRecord.model_fields))
    assert mapping["grounded_in_contracts"]["evidence_fields"] == ["taxonomy", "stage"]

    # The eligible set is exactly OOM and KERNEL_STALL at the real forward/backward stages.
    eligible = {e["taxonomy"]: set(e["stages"]) for e in mapping["eligible"]}
    assert eligible == {"OOM": {"forward", "backward"}, "KERNEL_STALL": {"forward", "backward"}}
    for taxonomy in ("OOM", "TIMEOUT", "KERNEL_STALL"):
        assert taxonomy in {e.value for e in FailureTaxonomy}
    for stage in ("forward", "backward", "model_load"):
        assert stage in {e.value for e in StageMarker}

    # TIMEOUT is withheld fail-closed, with a recorded evidence basis; defaults are fail-closed.
    assert mapping["timeout_decision"] == "withhold"
    assert isinstance(mapping["timeout_evidence_basis"], str) and mapping["timeout_evidence_basis"]
    assert mapping["default_action"] == "withhold-flash-and-stop"
    assert mapping["unmapped_combination_action"] == "NOT_RUN-stop-fail-closed"
    assert mapping["decision_is_scheduling_eligibility_not_math_kernel_causation"] is True

    # The committed mapping passes the validator gate unmodified.
    vp = _load_validator_module()
    vp._validate_math_terminal_flash_eligibility(effective)


def test_math_terminal_flash_eligibility_rejects_bad_mappings() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()

    def bad(mutate) -> dict:
        clone = copy.deepcopy(effective)
        mutate(_mapping(clone))
        return clone

    # Unknown taxonomy / unknown stage in an eligible entry.
    with pytest.raises(vp.ProtocolValidationError, match="unknown taxonomy"):
        vp._validate_math_terminal_flash_eligibility(
            bad(lambda m: m["eligible"].append({"taxonomy": "BOGUS", "stages": ["forward"]}))
        )
    with pytest.raises(vp.ProtocolValidationError, match="unknown stage"):
        vp._validate_math_terminal_flash_eligibility(
            bad(lambda m: m["eligible"][0]["stages"].append("not_a_stage"))
        )
    # OOM marked eligible at model_load.
    with pytest.raises(vp.ProtocolValidationError, match="OOM at model_load must not be eligible"):
        vp._validate_math_terminal_flash_eligibility(
            bad(lambda m: m["eligible"][0]["stages"].append("model_load"))
        )
    # Generic TIMEOUT marked eligible without machine-readable evidence.
    with pytest.raises(vp.ProtocolValidationError, match="exactly OOM and KERNEL_STALL"):
        vp._validate_math_terminal_flash_eligibility(
            bad(lambda m: m["eligible"].append(
                {"taxonomy": "TIMEOUT", "stages": ["forward", "backward"]}
            ))
        )
    # Missing KERNEL_STALL treatment.
    with pytest.raises(vp.ProtocolValidationError, match="omits KERNEL_STALL"):
        vp._validate_math_terminal_flash_eligibility(bad(lambda m: m["eligible"].pop(1)))
    # Missing / non-fail-closed default action.
    with pytest.raises(vp.ProtocolValidationError, match="default_action must be"):
        vp._validate_math_terminal_flash_eligibility(bad(lambda m: m.pop("default_action")))
    with pytest.raises(vp.ProtocolValidationError, match="default_action must be"):
        vp._validate_math_terminal_flash_eligibility(
            bad(lambda m: m.__setitem__("default_action", "run-flash-anyway"))
        )
    # Incomplete mapping of existing FailureTaxonomy values.
    with pytest.raises(vp.ProtocolValidationError, match="known_failure_taxonomy is incomplete"):
        vp._validate_math_terminal_flash_eligibility(
            bad(lambda m: m["known_failure_taxonomy"].remove("OOM"))
        )
    # Reintroduction of an invented terminal class.
    with pytest.raises(vp.ProtocolValidationError, match="unknown taxonomy"):
        vp._validate_math_terminal_flash_eligibility(
            bad(lambda m: m.__setitem__(
                "eligible", [{"taxonomy": "kernel_specific_oom", "stages": ["forward"]}]
            ))
        )


def test_feasibility_rung_aggregation_and_matched_pair_are_separate() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    results = _ladder(effective)["rung_result_definitions"]
    assert results["kernel_success"] == "one-kernel-satisfies-every-rung_success_requires-condition"
    assert results["rung_success"] == "at-least-one-executed-kernel-succeeds"
    assert results["matched_pair_success"] == "both-math-and-flash-succeed"
    assert results["sequence_4096_feasibility_claim"] == "at-least-one-kernel-succeeds-at-rung-4096"
    # A flash failure must not erase a valid math success; math-fail then flash-success is rung
    # success but not matched-pair success.
    assert results["flash_failure_does_not_erase_math_success"] is True
    assert results["math_fail_then_flash_success_is_rung_success_not_matched_pair"] is True

    # Requiring both kernels for rung success (instead of at-least-one) is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["rung_result_definitions"]["rung_success"] = "both-kernels-succeed"
    with pytest.raises(vp.ProtocolValidationError, match="rung_success definition must be"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # Letting a flash failure erase a math success is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["rung_result_definitions"]["flash_failure_does_not_erase_math_success"] = False
    with pytest.raises(vp.ProtocolValidationError, match="must not erase a valid math success"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # A weaker 4096 claim (both kernels) than "at least one kernel" is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["rung_result_definitions"]["sequence_4096_feasibility_claim"] = "both-kernels-at-4096"
    with pytest.raises(vp.ProtocolValidationError, match="sequence_4096_feasibility_claim definition"):
        vp._validate_seven_b_feasibility_ladder(bad)


def test_feasibility_progression_and_stopping_rule_is_bound() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    progression = _ladder(effective)["progression"]
    assert progression["advance_only_after_rung_success"] is True
    assert progression["stop_after_rung_with_no_kernel_success"] is True
    assert progression["longer_rungs_status_after_stop"] == "NOT_RUN_PRIOR_RUNG_NO_SUCCESS"
    assert progression["impute_longer_rungs"] is False
    assert progression["impute_any_result"] is False
    assert progression["shared_path_failure_stops_ladder_immediately"] is True
    assert progression["preserve_every_terminal_result"] is True

    # Advancing without rung success is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["progression"]["advance_only_after_rung_success"] = False
    with pytest.raises(vp.ProtocolValidationError, match="advance_only_after_rung_success must be True"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # A wrong NOT_RUN status for longer rungs is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["progression"]["longer_rungs_status_after_stop"] = "NOT_RUN"
    with pytest.raises(vp.ProtocolValidationError, match="NOT_RUN_PRIOR_RUNG_NO_SUCCESS"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # Imputing longer rungs is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["progression"]["impute_longer_rungs"] = True
    with pytest.raises(vp.ProtocolValidationError, match="impute_longer_rungs must be False"):
        vp._validate_seven_b_feasibility_ladder(bad)


def test_feasibility_rung_success_criteria_are_exact_and_4096_no_weaker() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    ladder = _ladder(effective)
    expected = list(vp.RUNG_SUCCESS_CRITERIA)
    assert ladder["rung_success_requires"] == expected
    assert ladder["sequence_length_4096_success_requires"] == expected
    # Every directive-named success condition is present.
    for required in (
        "exactly_12_optimizer_steps",
        "finite_loss_at_every_step",
        "forced_declared_kernel_no_fallback",
        "positive_token_evidence",
        "changed_adapter_state",
        "admitted_artifact",
        "complete_telemetry",
        "measured_fit",
        "clean_gpu_release",
    ):
        assert required in expected

    # Weakening the rung criteria is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["rung_success_requires"] = ["measured_fit"]
    with pytest.raises(vp.ProtocolValidationError, match="exact required criteria set"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # Weakening seq-4096 below the rung criteria is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["sequence_length_4096_success_requires"] = expected[:-1]
    with pytest.raises(vp.ProtocolValidationError, match="no weaker than the rung criteria"):
        vp._validate_seven_b_feasibility_ladder(bad)


def test_feasibility_ladder_is_separated_from_the_primary_paper_matrix() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    ladder = _ladder(effective)
    assert ladder["classification"] == "non-paper-feasibility"
    assert ladder["is_primary_paper_cell"] is False
    assert ladder["satisfies_three_trial_characterization_matrix"] is False
    # The private corpus 7B primary cells remain their own, separate arm.
    assert "qwen2.5-7b-instruct" in effective["primary_matrix"]["model_ids"]

    # Claiming the ladder is a primary paper cell is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["is_primary_paper_cell"] = True
    with pytest.raises(vp.ProtocolValidationError, match="not be a primary paper cell"):
        vp._validate_seven_b_feasibility_ladder(bad)

    # Claiming it satisfies the three-trial matrix is refused.
    bad = copy.deepcopy(effective)
    _ladder(bad)["satisfies_three_trial_characterization_matrix"] = True
    with pytest.raises(vp.ProtocolValidationError, match="three-trial characterization matrix"):
        vp._validate_seven_b_feasibility_ladder(bad)


def test_lineage_classification_separates_wheel_identity_from_worker_execution() -> None:
    vp = _load_validator_module()
    effective = _effective_matrix()
    classification = effective["lineage_change_classification"]
    # v8 requires a fresh wheel/environment lineage but is NOT a worker-execution change.
    assert classification["NEW_WHEEL_AND_ENVIRONMENT_LINEAGE_REQUIRED"] is True
    assert classification["WORKER_EXECUTION_CHANGE_REQUIRED"] is False
    assert classification["reason_code"] == (
        "worker-artifact-identity-and-manager-lock-generation-changed-fresh-v8-required"
    )
    # The manifest reason codes carry the corrected, non-overclaiming code and not the old one.
    manifest = json.loads(
        (_STUDY / "amendments"
         / "0005-2026-07-16-v8-manager-1.4-floor-binding-lineage.manifest.json").read_text()
    )
    assert (
        "worker-artifact-identity-and-manager-lock-generation-changed-fresh-v8-required"
        in manifest["reason_codes"]
    )
    assert (
        "worker-execution-and-wheel-identity-changed-fresh-v8-required"
        not in manifest["reason_codes"]
    )

    # A reason code that claims a worker-execution change while the flag denies one is refused.
    bad = copy.deepcopy(effective)
    bad["lineage_change_classification"]["reason_code"] = "worker-execution-changed-fresh-v8"
    with pytest.raises(vp.ProtocolValidationError, match="worker-execution change while"):
        vp._validate_lineage_change_classification(bad)
