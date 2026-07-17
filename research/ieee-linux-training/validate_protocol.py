"""Fail-closed validator for the append-only native-Linux research specification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

# A prospective amendment must record an actual authoring instant, never a future one. A modest skew
# tolerance keeps CI green across machines whose clocks differ by seconds/minutes, without allowing a
# convenient future timestamp.
_AUTHORED_AT_FUTURE_TOLERANCE = timedelta(hours=1)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = Path(__file__).resolve().parent
BASE_PROTOCOL = STUDY_ROOT / "PROTOCOL.md"
BASE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.yaml"
# Current (newest) amendment: 0005 -> effective matrix 1.5.0, reserved-identity registry v5.
EFFECTIVE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.v1.5.0.json"
AMENDMENT = STUDY_ROOT / (
    "amendments/0005-2026-07-16-v8-manager-1.4-floor-binding-lineage.md"
)
AMENDMENT_MANIFEST = STUDY_ROOT / (
    "amendments/0005-2026-07-16-v8-manager-1.4-floor-binding-lineage.manifest.json"
)
RESERVED_IDENTITIES = STUDY_ROOT / "amendments/RESERVED_IDENTITIES.v5.json"
# Frozen prior amendment (0004 -> effective matrix 1.4.0). The current amendment supersedes it; the
# chain is verified below so 0004 stays byte-frozen and the amendment ordering is provable.
PRIOR_AMENDMENT = STUDY_ROOT / (
    "amendments/0004-2026-07-16-v7-worker-lineage-token-throughput-observer.md"
)
PRIOR_AMENDMENT_MANIFEST = STUDY_ROOT / (
    "amendments/0004-2026-07-16-v7-worker-lineage-token-throughput-observer.manifest.json"
)
PRIOR_EFFECTIVE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.v1.4.0.json"
PRIOR_RESERVED_IDENTITIES = STUDY_ROOT / "amendments/RESERVED_IDENTITIES.v4.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HASH_IDENTITY_FIELDS = {
    "environment_lock_hashes",
    "worker_wheel_sha256",
    "plan_hashes",
    "execution_configuration_hashes",
}
PATH_IDENTITY_FIELDS = {"output_paths", "evidence_roots"}
ID_IDENTITY_FIELDS = {
    "environment_ids",
    "plan_ids",
    "execution_configuration_ids",
    "run_ids",
    "artifact_ids",
}
STAGE_REQUIRED_NONEMPTY = {
    "environment_plan": {"environment_ids", "worker_wheel_sha256", "evidence_roots"},
    "runplan": {
        "environment_ids",
        "environment_lock_hashes",
        "worker_wheel_sha256",
        "plan_ids",
        "plan_hashes",
        "execution_configuration_ids",
        "execution_configuration_hashes",
        "output_paths",
        "evidence_roots",
    },
    "trial": {
        "environment_ids",
        "environment_lock_hashes",
        "worker_wheel_sha256",
        "plan_ids",
        "plan_hashes",
        "execution_configuration_ids",
        "execution_configuration_hashes",
        "run_ids",
        "output_paths",
        "evidence_roots",
    },
}
STAGE_REQUIRED_EMPTY = {
    "environment_plan": {
        "environment_lock_hashes",
        "plan_ids",
        "plan_hashes",
        "execution_configuration_ids",
        "execution_configuration_hashes",
        "run_ids",
        "output_paths",
        "artifact_ids",
    },
    "runplan": {"run_ids", "artifact_ids"},
    "trial": set(),
}

# The separate, non-paper 7B feasibility ladder is a preregistered top-level field of the effective
# matrix. These bind its semantics unambiguously so the sealed spec cannot be read two ways.
SEVEN_B_LADDER_KEY = "seven_b_native_linux_feasibility_ladder"
LINEAGE_CLASSIFICATION_KEY = "lineage_change_classification"
FIXTURE_SHA_SENTINEL = "required-before-planning"
# A feasibility rung counts as a success only under this exact, ordered set of conditions - nothing
# weaker, and sequence length 4096 is held to the identical set.
RUNG_SUCCESS_CRITERIA = (
    "exactly_12_optimizer_steps",
    "finite_loss_at_every_step",
    "forced_declared_kernel_no_fallback",
    "positive_token_evidence",
    "changed_adapter_state",
    "admitted_artifact",
    "complete_telemetry",
    "measured_fit",
    "clean_gpu_release",
)
# Fixed feasibility ladder - exact controlling values, bound (not validated by nonempty prose).
LADDER_RUNGS = [512, 1024, 2048, 3072, 4096]
LADDER_RUNG_ORDER = "ascending"
# Flash runs at most once, only after math succeeds OR after a clean, conclusively math-specific OOM or
# timeout; it is withheld for any other failure class. The exact controlling values follow.
FLASH_CONDITION_REQUIRED = "run-after-math-success-or-clean-math-specific-oom-or-timeout"
ELIGIBLE_MATH_TERMINAL_CLASSES = frozenset({"kernel_specific_oom", "kernel_specific_timeout"})
CLEAN_MATH_SPECIFIC_PRECONDITIONS = frozenset(
    {
        "shared_preparation_passed",
        "declared_math_kernel_selected_no_fallback",
        "evidence_remained_valid",
        "process_terminated_cleanly",
        "gpu_memory_released",
        "environment_health_and_drift_checks_passed",
    }
)
WITHHOLD_FLASH_FAILURE_CLASSES = frozenset(
    {
        "shared_path",
        "identity",
        "environment",
        "artifact",
        "telemetry",
        "protocol",
        "corruption",
        "uncontrolled_health",
    }
)
KERNEL_SUCCESS_DEFINITION = "one-kernel-satisfies-every-rung_success_requires-condition"
RUNG_SUCCESS_DEFINITION = "at-least-one-executed-kernel-succeeds"
MATCHED_PAIR_DEFINITION = "both-math-and-flash-succeed"
SEQ_4096_CLAIM_DEFINITION = "at-least-one-kernel-succeeds-at-rung-4096"
NOT_RUN_LONGER_STATUS = "NOT_RUN_PRIOR_RUNG_NO_SUCCESS"


class ProtocolValidationError(ValueError):
    """The committed study specification is ambiguous, stale, or identity-inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ProtocolValidationError(f"{path.name} must end in exactly one LF")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{path.name} must contain one JSON object")
    return value


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ProtocolValidationError("research matrix mapping keys must be strings")
        if key in result:
            raise ProtocolValidationError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProtocolValidationError(f"invalid YAML in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"{path.name} must contain one mapping")
    return value


def _require_hash(actual: str, expected: object, label: str) -> None:
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise ProtocolValidationError(f"{label} does not carry a lowercase SHA-256")
    if actual != expected:
        raise ProtocolValidationError(f"{label} hash mismatch: {actual} != {expected}")


def _apply_change(document: dict[str, Any], change: dict[str, Any]) -> None:
    selector = change.get("selector")
    if not isinstance(selector, dict) or set(selector) != {"collection", "id", "field"}:
        raise ProtocolValidationError("an amendment selector is malformed")
    collection_name = selector["collection"]
    item_id = selector["id"]
    field = selector["field"]
    collection = document.get(collection_name)
    if not isinstance(collection, list):
        raise ProtocolValidationError(f"selector collection is not a list: {collection_name}")
    matches = [item for item in collection if isinstance(item, dict) and item.get("id") == item_id]
    if len(matches) != 1:
        raise ProtocolValidationError(
            f"selector must match exactly one item: {collection_name}/{item_id}"
        )
    target = matches[0]
    if target.get(field) != change.get("old_value"):
        raise ProtocolValidationError(
            f"amendment old-value precondition failed: {collection_name}/{item_id}/{field}"
        )
    target[field] = change.get("new_value")


def _build_expected_effective(
    base: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    expected = copy.deepcopy(base)
    direct_changes = manifest.get("direct_changes")
    changes = manifest.get("changes")
    immutable_additions = manifest.get("immutable_binding_additions")
    additions = manifest.get("added_top_level_fields")
    if not isinstance(direct_changes, dict):
        raise ProtocolValidationError("amendment direct changes must be an object")
    if not isinstance(immutable_additions, dict):
        raise ProtocolValidationError("amendment immutable-binding additions must be an object")
    if not isinstance(additions, dict):
        raise ProtocolValidationError("amendment top-level additions must be an object")
    if not isinstance(changes, list) or not all(isinstance(item, dict) for item in changes):
        raise ProtocolValidationError("amendment changes must be an array of objects")
    for field, values in direct_changes.items():
        if not isinstance(values, dict) or set(values) != {"old_value", "new_value"}:
            raise ProtocolValidationError(f"malformed direct change: {field}")
        if expected.get(field) != values["old_value"]:
            raise ProtocolValidationError(f"direct-change old-value precondition failed: {field}")
        expected[field] = values["new_value"]
    for change in changes:
        _apply_change(expected, change)
    bindings = expected.get("immutable_bindings")
    if not isinstance(bindings, dict):
        raise ProtocolValidationError("base immutable_bindings is not an object")
    for field, value in immutable_additions.items():
        if field in bindings:
            raise ProtocolValidationError(f"immutable binding addition already exists: {field}")
        bindings[field] = value
    for field, value in additions.items():
        if field in expected:
            raise ProtocolValidationError(f"top-level amendment addition already exists: {field}")
        expected[field] = value
    return expected


def _validate_reserved(reserved: dict[str, Any]) -> None:
    identity_classes = reserved.get("disjointness_required_for_new_evidence")
    if not isinstance(identity_classes, list) or not identity_classes:
        raise ProtocolValidationError("reserved identity classes are missing")
    for field in identity_classes:
        values = reserved.get(field)
        if not isinstance(field, str) or not isinstance(values, list):
            raise ProtocolValidationError(f"reserved identity class is malformed: {field}")
        if values != sorted(set(values)):
            raise ProtocolValidationError(f"reserved identities must be sorted and unique: {field}")
        if not all(isinstance(value, str) and value for value in values):
            raise ProtocolValidationError(f"reserved identity values must be nonempty strings: {field}")
    if reserved.get("reuse_authorized") is not False:
        raise ProtocolValidationError("historical identity reuse must remain unauthorized")


def _validate_reserved_superset(reserved: dict[str, Any]) -> None:
    """The reserved registry is append-only: every prior-version reserved identity must remain
    reserved, so a superseding amendment can never silently drop a historical identity from the
    non-reuse set."""

    prior = _load_json(PRIOR_RESERVED_IDENTITIES)
    prior_classes = prior.get("disjointness_required_for_new_evidence")
    if not isinstance(prior_classes, list) or not prior_classes:
        raise ProtocolValidationError("prior reserved identity classes are missing")
    for field in prior_classes:
        if field not in reserved.get("disjointness_required_for_new_evidence", []):
            raise ProtocolValidationError(f"reserved registry dropped identity class: {field}")
        missing = sorted(set(prior.get(field, [])) - set(reserved.get(field, [])))
        if missing:
            raise ProtocolValidationError(
                f"reserved registry dropped prior identities in {field}: {', '.join(missing)}"
            )


def _validate_supersession(manifest: dict[str, Any]) -> None:
    """Bind the frozen prior amendment by exact hash so the amendment chain is ordered and each prior
    amendment stays byte-frozen. A superseding amendment records the prior effective version and the
    raw-byte hashes of the prior manifest, narrative, effective matrix, and reserved-identity set."""

    supersedes = manifest.get("supersedes")
    if not isinstance(supersedes, dict):
        raise ProtocolValidationError("amendment must record the superseded prior amendment")
    if supersedes.get("effective_protocol_version") != "1.4.0":
        raise ProtocolValidationError("amendment must supersede exactly effective version 1.4.0")
    prior_files = {
        "prior_amendment_manifest_sha256": PRIOR_AMENDMENT_MANIFEST,
        "prior_narrative_sha256": PRIOR_AMENDMENT,
        "prior_effective_matrix_sha256": PRIOR_EFFECTIVE_MATRIX,
        "prior_reserved_identities_sha256": PRIOR_RESERVED_IDENTITIES,
    }
    for field, path in prior_files.items():
        _require_hash(_sha256(path), supersedes.get(field), f"superseded {field}")


def _validate_authored_at(manifest: dict[str, Any], now: datetime | None = None) -> None:
    """A prospective amendment must record a real authoring instant, not a future one. Rejects a
    missing/malformed ``authored_at`` and any value later than the current UTC time plus a small
    clock-skew tolerance."""

    raw = manifest.get("authored_at")
    if not isinstance(raw, str):
        raise ProtocolValidationError("amendment authored_at is missing")
    try:
        authored = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ProtocolValidationError(
            f"amendment authored_at is not an ISO-8601 UTC instant (YYYY-MM-DDTHH:MM:SSZ): {raw}"
        ) from exc
    current = now or datetime.now(timezone.utc)
    if authored > current + _AUTHORED_AT_FUTURE_TOLERANCE:
        raise ProtocolValidationError(f"amendment authored_at is in the future: {raw}")


def _validate_affected_counts(effective: dict[str, Any]) -> None:
    snapshot = effective.get("affected_scope_snapshot")
    primary = effective.get("primary_matrix")
    secondary = effective.get("secondary_adapter_matrix")
    if not isinstance(snapshot, dict):
        raise ProtocolValidationError("affected-scope snapshot is malformed")
    if not isinstance(primary, dict):
        raise ProtocolValidationError("primary matrix is malformed")
    if not isinstance(secondary, dict):
        raise ProtocolValidationError("secondary matrix is malformed")
    primary_count = (
        len(primary.get("model_ids", []))
        * len(primary.get("sequence_lengths", []))
        * len(primary.get("adapter_methods", []))
        * len(primary.get("execution_path_ids", []))
    )
    secondary_first_party = [
        item
        for item in secondary.get("execution_path_ids", [])
        if item in {"first-party-math", "first-party-flash"}
    ]
    secondary_count = (
        len(secondary.get("model_ids", []))
        * len(secondary.get("sequence_lengths", []))
        * len(secondary.get("adapter_methods", []))
        * len(secondary_first_party)
    )
    if primary_count != snapshot.get("primary_cells", {}).get("expected_affected_cell_count"):
        raise ProtocolValidationError("primary affected-cell count is stale")
    if secondary_count != snapshot.get("secondary_cells", {}).get("expected_affected_cell_count"):
        raise ProtocolValidationError("secondary affected-cell count is stale")


def _validate_lineage_change_classification(effective: dict[str, Any]) -> None:
    """A superseding lineage amendment must state, honestly and separately, whether it forces a fresh
    wheel/environment lineage and whether it changed the worker execution closure. The two claims are
    independent: a change to the wheel's build-provenance content or the manager lock generation changes
    the wheel/environment identity WITHOUT changing the worker execution bytes. This guard keeps the two
    reason codes distinct and refuses a matrix that omits the classification, while leaving the
    worker-execution flag free to be true for an amendment that genuinely changes the closure."""

    classification = effective.get(LINEAGE_CLASSIFICATION_KEY)
    if not isinstance(classification, dict):
        raise ProtocolValidationError("effective matrix omits the lineage-change classification")
    new_wheel = classification.get("NEW_WHEEL_AND_ENVIRONMENT_LINEAGE_REQUIRED")
    worker_execution = classification.get("WORKER_EXECUTION_CHANGE_REQUIRED")
    if not isinstance(new_wheel, bool) or not isinstance(worker_execution, bool):
        raise ProtocolValidationError(
            "lineage-change classification flags must be booleans "
            "(NEW_WHEEL_AND_ENVIRONMENT_LINEAGE_REQUIRED, WORKER_EXECUTION_CHANGE_REQUIRED)"
        )
    if not new_wheel:
        raise ProtocolValidationError(
            "a superseding lineage amendment must require a new wheel/environment lineage"
        )
    reason_code = classification.get("reason_code")
    if not isinstance(reason_code, str) or not reason_code:
        raise ProtocolValidationError("lineage-change classification must carry a reason code")
    if not worker_execution and "worker-execution" in reason_code:
        raise ProtocolValidationError(
            "reason code claims a worker-execution change while the classification denies one"
        )


def _validate_seven_b_feasibility_ladder(effective: dict[str, Any]) -> None:
    """The 7B feasibility ladder is a separate, non-paper arm. This binds its semantics shut, by exact
    controlling values (not descriptive prose): (0) non-paper separation; (1) the model reference must
    resolve to exactly one ``models[].id`` and match its repository (a repository-shaped id is refused);
    (2) the feasibility fixture carries its own identity contract, distinct from the private corpus;
    (3) the fixed ladder configuration (rung list and order, microbatch 1, gradient accumulation 1, 12
    bounded steps, no offload, zero automatic retries, math first-once, flash at-most-once); (4) the
    flash-eligibility rule (flash runs after math success OR a clean, conclusively math-specific OOM or
    timeout, and is withheld for every other enumerated failure class - so a math OOM that flash would
    survive is not a false negative); (5) the separate rung-result definitions (kernel success, rung
    success = at least one executed kernel succeeds, matched-pair = both succeed, seq-4096 claim = at
    least one kernel succeeds at 4096; a flash failure never erases a valid math success); (6) the
    progression and stopping rule (advance only after rung success, stop with no kernel success, longer
    rungs NOT_RUN_PRIOR_RUNG_NO_SUCCESS, no imputation, shared-path failure stops immediately); and
    (7) the exact per-kernel success criteria, with sequence length 4096 held no weaker."""

    ladder = effective.get(SEVEN_B_LADDER_KEY)
    if not isinstance(ladder, dict):
        raise ProtocolValidationError("seven-B feasibility ladder is missing or malformed")

    # (0) non-paper / primary-matrix separation.
    if ladder.get("classification") != "non-paper-feasibility":
        raise ProtocolValidationError("feasibility ladder must be classified non-paper-feasibility")
    if ladder.get("is_primary_paper_cell") is not False:
        raise ProtocolValidationError("feasibility ladder must not be a primary paper cell")
    if ladder.get("satisfies_three_trial_characterization_matrix") is not False:
        raise ProtocolValidationError(
            "feasibility ladder must not satisfy the three-trial characterization matrix"
        )

    # (1) model reference resolution.
    model_id = ladder.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ProtocolValidationError("feasibility ladder model_id is missing")
    if "/" in model_id:
        raise ProtocolValidationError(
            f"feasibility ladder model_id must be a models[].id, not a repository path: {model_id}"
        )
    models = effective.get("models")
    if not isinstance(models, list):
        raise ProtocolValidationError("effective matrix models list is malformed")
    resolved = [item for item in models if isinstance(item, dict) and item.get("id") == model_id]
    if len(resolved) != 1:
        raise ProtocolValidationError(
            f"feasibility ladder model_id must resolve to exactly one models[].id: {model_id}"
        )
    if ladder.get("model_source_repository") != resolved[0].get("source_repository"):
        raise ProtocolValidationError(
            "feasibility ladder model_source_repository does not match the resolved model"
        )

    # (2) feasibility fixture identity contract (distinct from the primary private corpus).
    fixture = ladder.get("feasibility_fixture")
    if not isinstance(fixture, dict):
        raise ProtocolValidationError("feasibility fixture identity contract is missing")
    if not isinstance(fixture.get("fixture_id"), str) or not fixture.get("fixture_id"):
        raise ProtocolValidationError("feasibility fixture must declare a fixture_id")
    if fixture.get("is_primary_private_corpus") is not False:
        raise ProtocolValidationError(
            "feasibility fixture must be explicitly distinct from the primary private corpus"
        )
    for sha_field in (
        "content_sha256",
        "rendered_examples_sha256",
        "tokenizer_content_sha256",
        "chat_template_sha256",
    ):
        value = fixture.get(sha_field)
        if value != FIXTURE_SHA_SENTINEL and not (
            isinstance(value, str) and SHA256_RE.fullmatch(value)
        ):
            raise ProtocolValidationError(
                f"feasibility fixture {sha_field} must be '{FIXTURE_SHA_SENTINEL}' or a SHA-256"
            )
    if fixture.get("license_evidence_required") is not True:
        raise ProtocolValidationError("feasibility fixture must require license evidence")
    if fixture.get("fixed_row_order") is not True:
        raise ProtocolValidationError("feasibility fixture must fix row order")
    if fixture.get("packing") is not False:
        raise ProtocolValidationError("feasibility fixture must disable packing")
    if fixture.get("truncation") is not False:
        raise ProtocolValidationError("feasibility fixture must disable truncation")

    # (3) fixed ladder configuration - exact controlling values.
    config = ladder.get("fixed_ladder_configuration")
    if not isinstance(config, dict):
        raise ProtocolValidationError("feasibility ladder fixed configuration is missing")
    if config.get("sequence_length_rungs") != LADDER_RUNGS:
        raise ProtocolValidationError(
            f"feasibility ladder rungs must be exactly {LADDER_RUNGS} in that order"
        )
    fixed_scalars = {
        "rung_order": LADDER_RUNG_ORDER,
        "microbatch": 1,
        "gradient_accumulation": 1,
        "bounded_optimizer_steps": 12,
        "offload": "none",
        "automatic_workload_retry_count": 0,
        "math_runs": "first-once",
        "flash_runs": "at-most-once",
    }
    for field, expected in fixed_scalars.items():
        if config.get(field) != expected:
            raise ProtocolValidationError(
                f"feasibility ladder fixed configuration {field} must be {expected!r}"
            )

    # (4) flash-eligibility rule - exactly bound (no false negative on a survivable math OOM).
    flash = ladder.get("flash_eligibility")
    if not isinstance(flash, dict):
        raise ProtocolValidationError("feasibility ladder flash eligibility is missing")
    if flash.get("flash_condition") != FLASH_CONDITION_REQUIRED:
        raise ProtocolValidationError(
            f"feasibility ladder flash_condition must be {FLASH_CONDITION_REQUIRED!r}"
        )
    if flash.get("run_when_math_succeeds") is not True:
        raise ProtocolValidationError("flash must run when math succeeds")
    if flash.get("run_when_math_ends_clean_kernel_specific_oom_or_timeout") is not True:
        raise ProtocolValidationError(
            "flash must run after a clean, conclusively math-specific OOM or timeout"
        )
    if set(flash.get("eligible_math_terminal_classes") or []) != ELIGIBLE_MATH_TERMINAL_CLASSES:
        raise ProtocolValidationError(
            "flash eligibility must bind exactly the kernel-specific OOM and timeout terminal classes"
        )
    if (
        set(flash.get("clean_math_specific_failure_preconditions") or [])
        != CLEAN_MATH_SPECIFIC_PRECONDITIONS
    ):
        raise ProtocolValidationError(
            "flash eligibility must bind exactly the clean math-specific failure preconditions"
        )
    if set(flash.get("withhold_flash_failure_classes") or []) != WITHHOLD_FLASH_FAILURE_CLASSES:
        raise ProtocolValidationError(
            "flash eligibility must withhold flash for exactly the enumerated failure classes"
        )
    if flash.get("withheld_flash_status") != "NOT_RUN":
        raise ProtocolValidationError("a withheld flash must be recorded NOT_RUN")

    # (5) separate rung-result definitions.
    results = ladder.get("rung_result_definitions")
    if not isinstance(results, dict):
        raise ProtocolValidationError("feasibility ladder rung result definitions are missing")
    result_defs = {
        "kernel_success": KERNEL_SUCCESS_DEFINITION,
        "rung_success": RUNG_SUCCESS_DEFINITION,
        "matched_pair_success": MATCHED_PAIR_DEFINITION,
        "sequence_4096_feasibility_claim": SEQ_4096_CLAIM_DEFINITION,
    }
    for field, expected in result_defs.items():
        if results.get(field) != expected:
            raise ProtocolValidationError(
                f"feasibility ladder {field} definition must be {expected!r}"
            )
    if results.get("flash_failure_does_not_erase_math_success") is not True:
        raise ProtocolValidationError("a flash failure must not erase a valid math success")
    if results.get("math_fail_then_flash_success_is_rung_success_not_matched_pair") is not True:
        raise ProtocolValidationError(
            "math-fail then flash-success must be rung success but not matched-pair success"
        )

    # (6) progression and stopping rule.
    progression = ladder.get("progression")
    if not isinstance(progression, dict):
        raise ProtocolValidationError("feasibility ladder progression is missing")
    progression_flags = {
        "advance_only_after_rung_success": True,
        "stop_after_rung_with_no_kernel_success": True,
        "impute_longer_rungs": False,
        "impute_any_result": False,
        "shared_path_failure_stops_ladder_immediately": True,
        "preserve_every_terminal_result": True,
    }
    for field, expected in progression_flags.items():
        if progression.get(field) is not expected:
            raise ProtocolValidationError(
                f"feasibility ladder progression {field} must be {expected}"
            )
    if progression.get("longer_rungs_status_after_stop") != NOT_RUN_LONGER_STATUS:
        raise ProtocolValidationError(
            f"longer rungs after a stop must be {NOT_RUN_LONGER_STATUS!r}"
        )

    # (7) exact per-kernel success criteria; sequence length 4096 no weaker.
    if list(ladder.get("rung_success_requires") or []) != list(RUNG_SUCCESS_CRITERIA):
        raise ProtocolValidationError(
            "feasibility ladder rung_success_requires is not the exact required criteria set"
        )
    if list(ladder.get("sequence_length_4096_success_requires") or []) != list(RUNG_SUCCESS_CRITERIA):
        raise ProtocolValidationError(
            "sequence-length-4096 success criteria must be no weaker than the rung criteria"
        )


def validate_candidate_identities(
    candidate_path: Path,
    reserved: dict[str, Any],
    effective: dict[str, Any],
) -> None:
    candidate = _load_json(candidate_path)
    identity_classes = reserved["disjointness_required_for_new_evidence"]
    expected_fields = {"schema_version", "stage", *identity_classes}
    if set(candidate) != expected_fields:
        missing = sorted(expected_fields - set(candidate))
        extra = sorted(set(candidate) - expected_fields)
        raise ProtocolValidationError(
            f"candidate identity fields do not match the stage schema "
            f"(missing={missing}, extra={extra})"
        )
    if candidate["schema_version"] != "1.0.0":
        raise ProtocolValidationError("candidate identity schema_version must be 1.0.0")
    stage = candidate["stage"]
    if stage not in STAGE_REQUIRED_NONEMPTY:
        raise ProtocolValidationError(f"unknown candidate identity stage: {stage!r}")

    for field in identity_classes:
        values = candidate[field]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ProtocolValidationError(f"candidate identity class is malformed: {field}")
        if values != sorted(set(values)):
            raise ProtocolValidationError(
                f"candidate identities must be sorted and unique: {field}"
            )
        if field in STAGE_REQUIRED_NONEMPTY[stage] and not values:
            raise ProtocolValidationError(
                f"candidate stage {stage!r} requires a nonempty {field} inventory"
            )
        if field in STAGE_REQUIRED_EMPTY[stage] and values:
            raise ProtocolValidationError(
                f"candidate stage {stage!r} requires an empty {field} inventory"
            )
        if field in HASH_IDENTITY_FIELDS and not all(SHA256_RE.fullmatch(value) for value in values):
            raise ProtocolValidationError(
                f"candidate {field} must contain lowercase SHA-256 values"
            )
        if field in ID_IDENTITY_FIELDS and not all(IDENTITY_RE.fullmatch(value) for value in values):
            raise ProtocolValidationError(
                f"candidate {field} must contain canonical ASCII identities"
            )
        overlap = sorted(set(values).intersection(reserved[field]))
        if field in PATH_IDENTITY_FIELDS:
            for value in values:
                candidate_path_value = PurePosixPath(value)
                if (
                    not candidate_path_value.is_absolute()
                    or ".." in candidate_path_value.parts
                    or "." in candidate_path_value.parts
                    or str(candidate_path_value) != value
                ):
                    raise ProtocolValidationError(
                        f"candidate {field} must contain canonical absolute POSIX paths"
                    )
                for prior_value in reserved[field]:
                    prior = PurePosixPath(prior_value)
                    if (
                        candidate_path_value == prior
                        or candidate_path_value.is_relative_to(prior)
                        or prior.is_relative_to(candidate_path_value)
                    ):
                        overlap.append(value)
                        break
            overlap = sorted(set(overlap))
        if overlap:
            raise ProtocolValidationError(
                f"candidate reuses reserved {field}: {', '.join(overlap)}"
            )

    if (
        len(candidate["environment_ids"]) != len(candidate["environment_lock_hashes"])
        and stage != "environment_plan"
    ):
        raise ProtocolValidationError(
            "candidate environment IDs and lock hashes must have equal cardinality"
        )
    if len(candidate["plan_ids"]) != len(candidate["plan_hashes"]):
        raise ProtocolValidationError("candidate plan IDs and hashes must have equal cardinality")
    if len(candidate["execution_configuration_ids"]) != len(
        candidate["execution_configuration_hashes"]
    ):
        raise ProtocolValidationError(
            "candidate execution configuration IDs and hashes must have equal cardinality"
        )
    if len(candidate["plan_ids"]) != len(candidate["execution_configuration_ids"]):
        raise ProtocolValidationError(
            "candidate plan and execution configuration inventories must have equal cardinality"
        )
    if len(candidate["worker_wheel_sha256"]) != 1:
        raise ProtocolValidationError("candidate must bind exactly one shared worker wheel")

    paths = effective.get("environment_admission", {}).get("sealed_environment_ids", {})
    expected_environments = {paths.get("math"), paths.get("flash")}
    if not all(isinstance(value, str) for value in expected_environments):
        raise ProtocolValidationError("effective matrix has malformed sealed environment IDs")
    observed_environments = set(candidate["environment_ids"])
    if stage in {"environment_plan", "runplan"} and observed_environments != expected_environments:
        raise ProtocolValidationError(
            "candidate matched-pair stage must bind both exact amended environment IDs"
        )
    if stage == "trial" and (
        len(observed_environments) != 1
        or not observed_environments.issubset(expected_environments)
    ):
        raise ProtocolValidationError(
            "candidate trial stage must bind one exact amended environment ID"
        )
    if stage == "runplan" and len(candidate["plan_ids"]) != len(observed_environments):
        raise ProtocolValidationError(
            "candidate matched RunPlan pair must contain one plan per environment"
        )
    if stage == "trial" and (
        len(candidate["plan_ids"]) != 1 or len(candidate["run_ids"]) != 1
    ):
        raise ProtocolValidationError("candidate trial must bind exactly one plan and one run")


def validate(
    *,
    candidate_identities: Path | None = None,
    verify_host_evidence: bool = False,
) -> dict[str, str]:
    manifest = _load_json(AMENDMENT_MANIFEST)
    reserved = _load_json(RESERVED_IDENTITIES)
    effective = _load_json(EFFECTIVE_MATRIX)
    base = _load_yaml(BASE_MATRIX)

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ProtocolValidationError("amendment manifest files object is missing")
    checks = {
        "base_protocol": (BASE_PROTOCOL, files.get("base_protocol_sha256")),
        "base_matrix": (BASE_MATRIX, files.get("base_matrix_sha256")),
        "effective_matrix": (EFFECTIVE_MATRIX, files.get("effective_matrix_sha256")),
        "narrative": (AMENDMENT, files.get("narrative_sha256")),
        "reserved_identities": (
            RESERVED_IDENTITIES,
            files.get("reserved_identities_sha256"),
        ),
        "validator": (Path(__file__), files.get("validator_sha256")),
    }
    observed: dict[str, str] = {}
    for label, (path, expected_hash) in checks.items():
        actual = _sha256(path)
        _require_hash(actual, expected_hash, label)
        observed[f"{label}_sha256"] = actual

    expected_effective = _build_expected_effective(base, manifest)
    if effective != expected_effective:
        raise ProtocolValidationError(
            "effective matrix differs from the exact base-plus-amendment construction"
        )
    _validate_reserved(reserved)
    _validate_reserved_superset(reserved)
    _validate_supersession(manifest)
    _validate_authored_at(manifest)
    _validate_affected_counts(effective)
    _validate_lineage_change_classification(effective)
    _validate_seven_b_feasibility_ladder(effective)

    non_reuse = effective.get("historical_identity_non_reuse")
    if not isinstance(non_reuse, dict):
        raise ProtocolValidationError("effective matrix omits historical non-reuse")
    _require_hash(
        observed["reserved_identities_sha256"],
        non_reuse.get("manifest_sha256"),
        "effective reserved-identities binding",
    )
    if candidate_identities is not None:
        validate_candidate_identities(candidate_identities, reserved, effective)

    if verify_host_evidence:
        source_manifests = reserved.get("source_manifests")
        if not isinstance(source_manifests, list):
            raise ProtocolValidationError("reserved source manifests are missing")
        for source in source_manifests:
            if not isinstance(source, dict):
                raise ProtocolValidationError("reserved source manifest entry is malformed")
            path = Path(str(source.get("path", "")))
            if not path.is_file():
                raise ProtocolValidationError(f"preserved source manifest is missing: {path}")
            _require_hash(_sha256(path), source.get("sha256"), f"preserved source {path}")

    return observed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-identities", type=Path)
    parser.add_argument("--verify-host-evidence", action="store_true")
    args = parser.parse_args(argv)
    try:
        observed = validate(
            candidate_identities=args.candidate_identities,
            verify_host_evidence=args.verify_host_evidence,
        )
    except (OSError, ProtocolValidationError) as exc:
        print(f"protocol validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "valid", **observed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
