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
# Current (newest) amendment: 0007 -> effective matrix 1.7.0, reserved-identity registry v7.
EFFECTIVE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.v1.7.0.json"
AMENDMENT = STUDY_ROOT / (
    "amendments/0007-2026-07-17-exact-length-fixture-binding.md"
)
AMENDMENT_MANIFEST = STUDY_ROOT / (
    "amendments/0007-2026-07-17-exact-length-fixture-binding.manifest.json"
)
RESERVED_IDENTITIES = STUDY_ROOT / "amendments/RESERVED_IDENTITIES.v7.json"
# Frozen prior amendment (0006 -> effective matrix 1.6.0). The current amendment supersedes it; the
# chain is verified below so 0006 stays byte-frozen and the amendment ordering is provable.
PRIOR_AMENDMENT = STUDY_ROOT / (
    "amendments/0006-2026-07-17-validator-hardening.md"
)
PRIOR_AMENDMENT_MANIFEST = STUDY_ROOT / (
    "amendments/0006-2026-07-17-validator-hardening.manifest.json"
)
PRIOR_EFFECTIVE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.v1.6.0.json"
PRIOR_RESERVED_IDENTITIES = STUDY_ROOT / "amendments/RESERVED_IDENTITIES.v6.json"
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
FLASH_CONDITION_REQUIRED = "run-after-math-success-or-mapped-clean-math-terminal-taxonomy-and-stage"
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
# Flash eligibility after a MATH FAILURE is bound to the real FailureRecord evidence (taxonomy + stage),
# not to invented terminal classes. It is a SCHEDULING decision (is flash worth trying?), never proof
# the math kernel caused the failure. The mapping is fail-closed: only these exact taxonomy/stage
# combinations keep flash eligible; every other, unknown, missing, or unmapped combination withholds
# flash and stops.
MATH_TERMINAL_FLASH_ELIGIBILITY_KEY = "math_terminal_flash_eligibility"
MATH_TERMINAL_ELIGIBLE_REQUIRED = {
    "OOM": frozenset({"forward", "backward"}),
    "KERNEL_STALL": frozenset({"forward", "backward"}),
}
MATH_TERMINAL_ELIGIBLE_STAGES = frozenset({"forward", "backward"})
MATH_TERMINAL_DEFAULT_ACTION = "withhold-flash-and-stop"
MATH_TERMINAL_UNMAPPED_ACTION = "NOT_RUN-stop-fail-closed"
# The existing FailureTaxonomy values whose treatment the mapping must make explicit. The engine test
# suite additionally binds the declared known-taxonomy/known-stage snapshots to the live enums exactly.
REQUIRED_TAXONOMY_TREATMENTS = frozenset({"OOM", "TIMEOUT", "KERNEL_STALL"})
FORBIDDEN_INVENTED_TERMINAL_CLASSES = ("kernel_specific_oom", "kernel_specific_timeout")
KERNEL_SUCCESS_DEFINITION = "one-kernel-satisfies-every-rung_success_requires-condition"
RUNG_SUCCESS_DEFINITION = "at-least-one-executed-kernel-succeeds"
MATCHED_PAIR_DEFINITION = "both-math-and-flash-succeed"
SEQ_4096_CLAIM_DEFINITION = "at-least-one-kernel-succeeds-at-rung-4096"
NOT_RUN_LONGER_STATUS = "NOT_RUN_PRIOR_RUNG_NO_SUCCESS"
# Execution-implying phrasings a reason code must NOT use when it denies a worker-execution change
# (a broader denylist than the single "worker-execution" token, so a reworded claim cannot dodge it).
_WORKER_EXECUTION_CLAIM_TOKENS = (
    "worker-execution",
    "worker-runtime",
    "worker-child",
    "worker-bytes",
    "worker-code",
    "runtime-bytes",
    "execution-bytes",
    "execution-change",
)
# Map an evidence sub-field (by exact name or ``_``-suffix) to the reserved-identity class it must be a
# member of, so a completed-run identity documented as history cannot be left reusable at plan time.
_EVIDENCE_FIELD_RESERVED_CLASS = {
    "worker_wheel_sha256": "worker_wheel_sha256",
    "run_id": "run_ids",
    "environment_id": "environment_ids",
    "plan_id": "plan_ids",
    "artifact_id": "artifact_ids",
}

# Amendment 0007 - exact-length (non-padding) sequence feasibility. The feasibility fixture is rebound to
# a per-rung, license-clear, exact-length chat fixture so a real mb=1 microbatch carries EXACTLY the rung
# in non-padding tokens (proven pre-dispatch by the actual TRL collator, CPU-only). This is a
# fixture-identity change ONLY: no worker execution change, no new wheel/environment lineage, no
# primary-matrix change. These bindings hold the new fixture semantics shut by exact controlling values.
NONPADDING_ARM_NAME = "seven_b_native_linux_nonpadding_sequence_feasibility"
NONPADDING_FIXTURE_ID = "cs-ieee-linux-7b-nonpadding-seq-fixture-v1"
NONPADDING_MODEL_REPOSITORY = "Qwen/Qwen2.5-7B-Instruct"
NONPADDING_MODEL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
NONPADDING_MODEL_LICENSE = "apache-2.0"
NONPADDING_FIXTURE_LICENSE = "CC0-1.0"
NONPADDING_OBJECTIVE_MODE = "full_language_model_supervision"
NONPADDING_RUNG_EQUALITY = "rung"  # admission binds equality to the rung, never a weaker "<=" bound
NONPADDING_CONFORMANCE_STATUS = "PREDISPATCH_RUNTIME_COLLATOR_CONFORMANCE_PASS"
NONPADDING_WIDTH_CLAIM_SOURCE = "raw_run_event_measurements"
# The primary private corpus id the feasibility fixture must never be conflated with.
PRIMARY_PRIVATE_DATASET_ID = "cs-ieee-linux-sft-v1"


def _require_full_sha256(value: object, label: str) -> None:
    """A bound hash must be a full 64-hex lowercase SHA-256 - an abbreviated hash is refused so a
    truncated digest can never enter the effective matrix."""

    if not (isinstance(value, str) and SHA256_RE.fullmatch(value)):
        raise ProtocolValidationError(f"{label} must be a full 64-hex SHA-256 (no abbreviation)")


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
        # Type-check before the sorted/unique check so a mixed-type reserved file fails as a clean
        # ProtocolValidationError rather than an unhandled TypeError from sorting mixed types.
        if not all(isinstance(value, str) and value for value in values):
            raise ProtocolValidationError(f"reserved identity values must be nonempty strings: {field}")
        if values != sorted(set(values)):
            raise ProtocolValidationError(f"reserved identities must be sorted and unique: {field}")
    if reserved.get("reuse_authorized") is not False:
        raise ProtocolValidationError("historical identity reuse must remain unauthorized")


def _validate_preserved_evidence_reserved(
    effective: dict[str, Any], reserved: dict[str, Any]
) -> None:
    """Every completed-run identity the effective matrix documents as history (in a
    ``preserved_*_evidence`` block) must actually be in the reserved-identity registry. This closes the
    gap the one-hop append-only superset check does not cover for identities NEW in this version: the
    newest lineage's real run/wheel/environment ids cannot be recorded as history yet left reusable at
    plan time."""

    for key, block in effective.items():
        if not (key.startswith("preserved_") and key.endswith("_evidence")):
            continue
        if not isinstance(block, dict):
            raise ProtocolValidationError(f"{key} must be an object")
        for sub_key, value in block.items():
            if not isinstance(value, str):
                continue
            reserved_class: str | None = None
            for suffix, cls in _EVIDENCE_FIELD_RESERVED_CLASS.items():
                if sub_key == suffix or sub_key.endswith("_" + suffix):
                    reserved_class = cls
                    break
            if reserved_class is None:
                continue
            pool = reserved.get(reserved_class)
            if not isinstance(pool, list) or value not in pool:
                raise ProtocolValidationError(
                    f"{key}.{sub_key}={value!r} is documented as history but is not reserved in the "
                    f"{reserved_class} registry"
                )


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
    if supersedes.get("effective_protocol_version") != "1.6.0":
        raise ProtocolValidationError("amendment must supersede exactly effective version 1.6.0")
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
    # When the classification denies a worker-execution change, the reason code must not claim one under
    # any of the known execution-implying phrasings (not just the exact "worker-execution" token).
    if not worker_execution:
        folded = reason_code.casefold()
        if any(token in folded for token in _WORKER_EXECUTION_CLAIM_TOKENS):
            raise ProtocolValidationError(
                "reason code claims a worker-execution change while the classification denies one"
            )


def _validate_math_terminal_flash_eligibility(effective: dict[str, Any]) -> None:
    """Flash eligibility after a math failure is bound to the actual ``FailureRecord`` evidence -
    ``taxonomy`` (FailureTaxonomy) and ``stage`` (StageMarker) - not to invented terminal classes. This
    is a fail-closed SCHEDULING decision (is flash worth trying at this rung?), never a claim that the
    math kernel caused the failure. Exactly OOM and KERNEL_STALL at forward/backward, under confirmed
    forced-math/no-fallback and all clean-failure preconditions, keep flash eligible; a generic TIMEOUT
    is withheld because no machine-readable field proves it occurred within the math attention execution;
    every other, unknown, missing, or unmapped taxonomy/stage combination withholds flash and stops."""

    mapping = effective.get(MATH_TERMINAL_FLASH_ELIGIBILITY_KEY)
    if not isinstance(mapping, dict):
        raise ProtocolValidationError("effective matrix omits math_terminal_flash_eligibility")

    # The declared taxonomy/stage snapshots (bound to the live enums exactly by the engine test suite).
    known_taxonomy = mapping.get("known_failure_taxonomy")
    known_stages = mapping.get("known_stage_markers")
    if not isinstance(known_taxonomy, list) or not all(
        isinstance(v, str) and v for v in known_taxonomy
    ):
        raise ProtocolValidationError("known_failure_taxonomy is malformed")
    if not isinstance(known_stages, list) or not all(isinstance(v, str) and v for v in known_stages):
        raise ProtocolValidationError("known_stage_markers is malformed")
    known_taxonomy_set = set(known_taxonomy)
    known_stage_set = set(known_stages)
    # Bind the declared snapshots to the LIVE runtime enums (fail-closed): the mapping is only "grounded
    # in the real contracts" if its known_failure_taxonomy / known_stage_markers equal, exactly and in
    # order, corpus_studio.platform.enums.FailureTaxonomy / StageMarker. A drifted or fabricated snapshot
    # is refused by the validator itself, not only by a CI test. The enums are torch-free; if they cannot
    # be imported the grounding cannot be verified, so validation refuses rather than trusting the file.
    try:
        from corpus_studio.platform.enums import (  # noqa: PLC0415
            FailureTaxonomy,
            StageMarker,
        )
    except ImportError as exc:  # pragma: no cover - exercised only outside an engine environment
        raise ProtocolValidationError(
            "cannot import the runtime FailureTaxonomy/StageMarker enums to verify the "
            "math_terminal_flash_eligibility taxonomy/stage snapshots"
        ) from exc
    if known_taxonomy != [member.value for member in FailureTaxonomy]:
        raise ProtocolValidationError(
            "known_failure_taxonomy does not equal the live FailureTaxonomy enum (exact order)"
        )
    if known_stages != [member.value for member in StageMarker]:
        raise ProtocolValidationError(
            "known_stage_markers does not equal the live StageMarker enum (exact order)"
        )
    if not REQUIRED_TAXONOMY_TREATMENTS <= known_taxonomy_set:
        missing = sorted(REQUIRED_TAXONOMY_TREATMENTS - known_taxonomy_set)
        raise ProtocolValidationError(
            f"known_failure_taxonomy is incomplete; missing required values: {missing}"
        )
    if not MATH_TERMINAL_ELIGIBLE_STAGES <= known_stage_set:
        raise ProtocolValidationError("known_stage_markers omits the forward/backward stages")

    # Eligible entries: exactly OOM and KERNEL_STALL, each at exactly {forward, backward}.
    eligible = mapping.get("eligible")
    if not isinstance(eligible, list):
        raise ProtocolValidationError("math_terminal_flash_eligibility.eligible must be a list")
    parsed: dict[str, frozenset[str]] = {}
    for entry in eligible:
        if not isinstance(entry, dict) or set(entry) != {"taxonomy", "stages"}:
            raise ProtocolValidationError(
                "an eligible entry is malformed (must have exactly taxonomy and stages)"
            )
        taxonomy = entry["taxonomy"]
        stages = entry["stages"]
        if taxonomy not in known_taxonomy_set:
            raise ProtocolValidationError(f"eligible entry uses an unknown taxonomy: {taxonomy}")
        if not isinstance(stages, list) or not stages:
            raise ProtocolValidationError(f"eligible taxonomy {taxonomy} needs a nonempty stage list")
        for stage in stages:
            if stage not in known_stage_set:
                raise ProtocolValidationError(f"eligible entry uses an unknown stage: {stage}")
        if taxonomy in parsed:
            raise ProtocolValidationError(f"eligible taxonomy {taxonomy} appears more than once")
        parsed[taxonomy] = frozenset(stages)
    # KERNEL_STALL treatment must be present (precise message before the exact-set check).
    if "KERNEL_STALL" not in parsed:
        raise ProtocolValidationError("eligible mapping omits KERNEL_STALL treatment")
    # OOM at a shared stage (e.g. model_load) must not be eligible.
    if "model_load" in parsed.get("OOM", frozenset()):
        raise ProtocolValidationError("OOM at model_load must not be eligible")
    if parsed != MATH_TERMINAL_ELIGIBLE_REQUIRED:
        raise ProtocolValidationError(
            "eligible mapping must be exactly OOM and KERNEL_STALL at {forward, backward}"
        )

    # Generic TIMEOUT is withheld fail-closed: it must not be eligible and must be explicitly decided.
    if "TIMEOUT" in parsed:
        raise ProtocolValidationError(
            "generic TIMEOUT must not be eligible without machine-readable math-attention-stage evidence"
        )
    if mapping.get("timeout_decision") != "withhold":
        raise ProtocolValidationError("TIMEOUT decision must be 'withhold' (fail-closed)")
    if not isinstance(mapping.get("timeout_evidence_basis"), str) or not mapping.get(
        "timeout_evidence_basis"
    ):
        raise ProtocolValidationError("TIMEOUT withholding must record its evidence basis")

    # Default and unmapped actions must be fail-closed.
    if mapping.get("default_action") != MATH_TERMINAL_DEFAULT_ACTION:
        raise ProtocolValidationError(
            f"math_terminal_flash_eligibility.default_action must be {MATH_TERMINAL_DEFAULT_ACTION!r}"
        )
    if mapping.get("unmapped_combination_action") != MATH_TERMINAL_UNMAPPED_ACTION:
        raise ProtocolValidationError(
            f"unmapped combination action must be {MATH_TERMINAL_UNMAPPED_ACTION!r}"
        )

    # Scheduling decision, not causation; grounded in the real contracts; with required guards.
    if mapping.get("decision_is_scheduling_eligibility_not_math_kernel_causation") is not True:
        raise ProtocolValidationError(
            "the mapping must state it is a scheduling eligibility decision, not math-kernel causation"
        )
    if mapping.get("requires_confirmed_forced_math_no_fallback") is not True:
        raise ProtocolValidationError("eligibility must require confirmed forced-math with no fallback")
    if mapping.get("requires_all_clean_failure_preconditions") is not True:
        raise ProtocolValidationError("eligibility must require all clean-failure preconditions")

    # No invented terminal classes may reappear anywhere in the mapping.
    blob = json.dumps(mapping)
    for invented in FORBIDDEN_INVENTED_TERMINAL_CLASSES:
        if invented in blob:
            raise ProtocolValidationError(f"invented terminal class must not reappear: {invented}")


def _validate_seven_b_feasibility_ladder(effective: dict[str, Any]) -> None:
    """The 7B feasibility ladder is a separate, non-paper arm. This binds its semantics shut, by exact
    controlling values (not descriptive prose): (0) non-paper separation; (1) the model reference must
    resolve to exactly one ``models[].id`` and match its repository (a repository-shaped id is refused);
    (2) the feasibility fixture carries its own identity contract, distinct from the private corpus;
    (3) the fixed ladder configuration (rung list and order, microbatch 1, gradient accumulation 1, 12
    bounded steps, no offload, zero automatic retries, math first-once, flash at-most-once); (4) the
    flash-eligibility rule (flash runs after math success OR when the math FailureRecord taxonomy/stage
    matches the grounded ``math_terminal_flash_eligibility`` mapping, and is withheld fail-closed
    otherwise - so a math OOM at forward/backward that flash would survive is not a false negative, while
    a generic TIMEOUT with no math-attention-stage evidence stays withheld); (5) the separate
    rung-result definitions (kernel success, rung
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

    # (4) flash-eligibility rule - grounded in the real FailureRecord taxonomy + stage mapping.
    flash = ladder.get("flash_eligibility")
    if not isinstance(flash, dict):
        raise ProtocolValidationError("feasibility ladder flash eligibility is missing")
    if flash.get("flash_condition") != FLASH_CONDITION_REQUIRED:
        raise ProtocolValidationError(
            f"feasibility ladder flash_condition must be {FLASH_CONDITION_REQUIRED!r}"
        )
    if flash.get("run_when_math_succeeds") is not True:
        raise ProtocolValidationError("flash must run when math succeeds")
    if flash.get("run_when_math_failure_matches_mapped_terminal_taxonomy_and_stage") is not True:
        raise ProtocolValidationError(
            "flash-after-math-failure must be governed by the mapped terminal taxonomy/stage rule"
        )
    if flash.get("terminal_taxonomy_mapping") != MATH_TERMINAL_FLASH_ELIGIBILITY_KEY:
        raise ProtocolValidationError(
            "flash eligibility must reference the math_terminal_flash_eligibility mapping"
        )
    if flash.get("confirmed_forced_math_no_fallback_required") is not True:
        raise ProtocolValidationError(
            "flash eligibility must require confirmed forced-math with no fallback"
        )
    if (
        set(flash.get("clean_math_specific_failure_preconditions") or [])
        != CLEAN_MATH_SPECIFIC_PRECONDITIONS
    ):
        raise ProtocolValidationError(
            "flash eligibility must bind exactly the clean math-specific failure preconditions"
        )
    if flash.get("withheld_flash_status") != "NOT_RUN":
        raise ProtocolValidationError("a withheld flash must be recorded NOT_RUN")
    if flash.get("decision_is_scheduling_eligibility_not_math_kernel_causation") is not True:
        raise ProtocolValidationError(
            "flash eligibility must state it is a scheduling decision, not math-kernel causation"
        )
    # No invented terminal classes may reappear anywhere in the ladder.
    ladder_blob = json.dumps(ladder)
    for invented in FORBIDDEN_INVENTED_TERMINAL_CLASSES:
        if invented in ladder_blob:
            raise ProtocolValidationError(
                f"invented terminal class must not reappear in the ladder: {invented}"
            )

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


def _validate_nonpadding_sequence_feasibility(effective: dict[str, Any]) -> None:
    """Amendment 0007 binds the exact-length (non-padding) feasibility fixture shut, by exact controlling
    values (not prose): (A) the fixture is the license-clear, per-rung exact-length chat fixture, distinct
    from the private corpus, with 12 rows/rung, fixed row order, no packing/truncation, and full canonical
    hashes for the generator, fixture-root SHA256SUMS, chat template, tokenizer content, and per-rung
    dataset/rendered/token-id aggregates - every per-rung ``exact_non_padding_length`` equal to its rung;
    (B) the model binding pins the 40-hex revision, model/tokenizer/template aggregates, apache-2.0, and
    trust_remote_code false; (C) the training objective is full-language-model supervision with the
    expected supervised-token count equal to the rung; (D) the pre-dispatch conformance is the actual-TRL
    collator PASS, explicitly NOT a completed worker execution or GPU result; (E) the execution-time
    admission requires, per optimizer step, observed_microbatches exactly 1, non-padding tokens EQUAL to
    the rung (never a weaker bound), supervised tokens equal to the rung, positive step time, and no
    truncation/packing/fallback with finite loss - a step whose non-padding tokens are below the rung
    invalidates that rung's sequence-width claim, and the claim is sourced from the raw RunEvent
    measurements, not sequence_len / fixture metadata / the collator report; and (F) the change is
    classified fixture-identity-only, with no worker-execution change and no new environment lineage."""

    ladder = effective.get(SEVEN_B_LADDER_KEY)
    if not isinstance(ladder, dict):
        raise ProtocolValidationError("seven-B feasibility ladder is missing or malformed")

    # (F) fixture-only classification - and it must not claim a worker change or a new lineage.
    classification = ladder.get("fixture_change_classification")
    if not isinstance(classification, dict):
        raise ProtocolValidationError("feasibility ladder is missing fixture_change_classification")
    for flag in (
        "FIXTURE_IDENTITY_CHANGE_ONLY",
        "NO_WORKER_CHANGE",
        "NO_NEW_ENVIRONMENT_LINEAGE",
        "NO_PRIMARY_MATRIX_CHANGE",
    ):
        if classification.get(flag) is not True:
            raise ProtocolValidationError(
                f"fixture_change_classification.{flag} must be true for this fixture-only amendment"
            )
    if ladder.get("arm_name") != NONPADDING_ARM_NAME:
        raise ProtocolValidationError(
            f"feasibility ladder arm_name must be {NONPADDING_ARM_NAME!r}"
        )
    folded_ladder = json.dumps(ladder).casefold()
    for token in _WORKER_EXECUTION_CLAIM_TOKENS:
        if token in folded_ladder:
            raise ProtocolValidationError(
                "a fixture-only amendment must not use worker-execution-change wording in the ladder"
            )
    if "new_lineage" in folded_ladder or "new-lineage" in folded_ladder:
        raise ProtocolValidationError(
            "a fixture-only amendment must not claim a new lineage in the ladder"
        )

    # (A) fixture identity contract - exact-length, per-rung, license-clear, distinct from the corpus.
    fixture = ladder.get("feasibility_fixture")
    if not isinstance(fixture, dict):
        raise ProtocolValidationError("feasibility fixture identity contract is missing")
    if fixture.get("fixture_id") != NONPADDING_FIXTURE_ID:
        raise ProtocolValidationError(
            f"feasibility fixture_id must be the exact-length fixture {NONPADDING_FIXTURE_ID!r}"
        )
    if fixture.get("fixture_id") == PRIMARY_PRIVATE_DATASET_ID:
        raise ProtocolValidationError("the feasibility fixture must not be the private corpus")
    if fixture.get("is_primary_private_corpus") is not False:
        raise ProtocolValidationError("feasibility fixture must not be the primary private corpus")
    if fixture.get("distinct_from_primary_private_corpus") is not True:
        raise ProtocolValidationError("feasibility fixture must be distinct from the private corpus")
    if fixture.get("license") != NONPADDING_FIXTURE_LICENSE:
        raise ProtocolValidationError(
            f"feasibility fixture license must be {NONPADDING_FIXTURE_LICENSE!r}"
        )
    if fixture.get("packing") is not False:
        raise ProtocolValidationError("feasibility fixture must disable packing")
    if fixture.get("truncation") is not False:
        raise ProtocolValidationError("feasibility fixture must disable truncation")
    if fixture.get("fixed_row_order") is not True:
        raise ProtocolValidationError("feasibility fixture must fix row order")
    if fixture.get("rows_per_rung") != 12:
        raise ProtocolValidationError("feasibility fixture must bind exactly 12 rows per rung")
    for sha_field in ("generator_sha256", "fixture_root_sha256sums",
                      "chat_template_sha256", "tokenizer_content_sha256"):
        _require_full_sha256(fixture.get(sha_field), f"feasibility fixture {sha_field}")

    # (B) exact model binding.
    binding = fixture.get("model_binding")
    if not isinstance(binding, dict):
        raise ProtocolValidationError("feasibility fixture model_binding is missing")
    if binding.get("repository") != NONPADDING_MODEL_REPOSITORY:
        raise ProtocolValidationError(
            f"feasibility fixture model repository must be {NONPADDING_MODEL_REPOSITORY!r}"
        )
    revision = binding.get("revision")
    if not (isinstance(revision, str) and NONPADDING_MODEL_REVISION_RE.fullmatch(revision)):
        raise ProtocolValidationError("feasibility fixture model revision must be a 40-hex commit")
    for sha_field in ("model_aggregate_sha256", "tokenizer_aggregate_sha256",
                      "chat_template_sha256", "acquisition_sha256sums"):
        _require_full_sha256(binding.get(sha_field), f"feasibility fixture model_binding {sha_field}")
    if binding.get("license") != NONPADDING_MODEL_LICENSE:
        raise ProtocolValidationError(
            f"feasibility fixture model license must be {NONPADDING_MODEL_LICENSE!r}"
        )
    if binding.get("trust_remote_code") is not False:
        raise ProtocolValidationError("feasibility fixture model trust_remote_code must be false")

    # (A cont.) per-rung exact-length identity: keys are exactly the rungs, each length equals its rung.
    per_rung = fixture.get("per_rung")
    if not isinstance(per_rung, dict):
        raise ProtocolValidationError("feasibility fixture per_rung binding is missing")
    if [str(rung) for rung in LADDER_RUNGS] != list(per_rung):
        raise ProtocolValidationError(
            f"feasibility fixture per_rung keys must be exactly {LADDER_RUNGS} in that order"
        )
    for rung in LADDER_RUNGS:
        entry = per_rung.get(str(rung))
        if not isinstance(entry, dict):
            raise ProtocolValidationError(f"feasibility fixture per_rung {rung} entry is malformed")
        if entry.get("exact_non_padding_length") != rung:
            raise ProtocolValidationError(
                f"feasibility fixture per_rung {rung} exact_non_padding_length must equal {rung}"
            )
        for sha_field in ("dataset_sha256", "rendered_examples_aggregate_sha256",
                          "token_id_aggregate_sha256"):
            _require_full_sha256(entry.get(sha_field), f"feasibility fixture per_rung {rung} {sha_field}")

    # (C) training objective - full-language-model supervision, supervised tokens equal to the rung.
    objective = ladder.get("training_objective")
    if not isinstance(objective, dict):
        raise ProtocolValidationError("feasibility ladder training_objective is missing")
    if not objective.get("mode"):
        raise ProtocolValidationError("feasibility ladder training_objective must declare a mode")
    if objective.get("mode") != NONPADDING_OBJECTIVE_MODE:
        raise ProtocolValidationError(
            f"feasibility ladder training objective mode must be {NONPADDING_OBJECTIVE_MODE!r}"
        )
    if objective.get("expected_supervised_tokens_per_microbatch") != NONPADDING_RUNG_EQUALITY:
        raise ProtocolValidationError(
            "feasibility ladder training objective must expect supervised tokens equal to the rung"
        )

    # (D) pre-dispatch conformance - the actual TRL collator PASS, NOT a completed worker/GPU result.
    conformance = ladder.get("predispatch_conformance")
    if not isinstance(conformance, dict):
        raise ProtocolValidationError("feasibility ladder predispatch_conformance is missing")
    if conformance.get("status") != NONPADDING_CONFORMANCE_STATUS:
        raise ProtocolValidationError(
            f"predispatch conformance status must be {NONPADDING_CONFORMANCE_STATUS!r}"
        )
    _require_full_sha256(
        conformance.get("conformance_evidence_sha256sums"),
        "predispatch conformance evidence SHA256SUMS",
    )
    if conformance.get("is_not_a_completed_worker_execution_or_gpu_result") is not True:
        raise ProtocolValidationError(
            "pre-dispatch collator conformance must be recorded as NOT a completed execution or GPU result"
        )

    # (E) execution-time admission - equality to the rung, observed_microbatches exactly 1, raw-event claim.
    admission = ladder.get("execution_time_admission")
    if not isinstance(admission, dict):
        raise ProtocolValidationError("feasibility ladder execution_time_admission is missing")
    per_step = admission.get("per_optimizer_step")
    if not isinstance(per_step, dict):
        raise ProtocolValidationError("execution_time_admission.per_optimizer_step is missing")
    if per_step.get("observed_microbatches") != 1:
        raise ProtocolValidationError("execution-time admission observed_microbatches must be exactly 1")
    if per_step.get("nonpadding_tokens_equals") != NONPADDING_RUNG_EQUALITY:
        raise ProtocolValidationError(
            "execution-time admission must require non-padding tokens EQUAL to the rung (not a weaker bound)"
        )
    if per_step.get("supervised_tokens_equals") != NONPADDING_RUNG_EQUALITY:
        raise ProtocolValidationError(
            "execution-time admission must require supervised tokens equal to the rung"
        )
    if per_step.get("step_time_seconds_gt") != 0:
        raise ProtocolValidationError("execution-time admission must require positive step time")
    for flag in ("no_truncation", "no_packing", "no_fallback", "finite_loss"):
        if per_step.get(flag) is not True:
            raise ProtocolValidationError(f"execution-time admission {flag} must be true")
    if admission.get("nonpadding_below_rung_invalidates_rung_sequence_width_claim") is not True:
        raise ProtocolValidationError(
            "admission must state that non-padding tokens below the rung invalidate the width claim"
        )
    if admission.get("sequence_width_claim_source") != NONPADDING_WIDTH_CLAIM_SOURCE:
        raise ProtocolValidationError(
            f"the sequence-width claim source must be {NONPADDING_WIDTH_CLAIM_SOURCE!r}"
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
        if field in ID_IDENTITY_FIELDS:
            # ID classes are compared case-INSENSITIVELY: a case-variant of a reserved id (e.g. an
            # uppercased-hex UUID) is not a fresh identity, so it must not slip past disjointness.
            reserved_folded = {str(item).casefold() for item in reserved[field]}
            overlap = sorted(value for value in values if value.casefold() in reserved_folded)
        else:
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
    _validate_preserved_evidence_reserved(effective, reserved)
    _validate_supersession(manifest)
    _validate_authored_at(manifest)
    _validate_affected_counts(effective)
    _validate_lineage_change_classification(effective)
    _validate_math_terminal_flash_eligibility(effective)
    _validate_seven_b_feasibility_ladder(effective)
    _validate_nonpadding_sequence_feasibility(effective)

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
