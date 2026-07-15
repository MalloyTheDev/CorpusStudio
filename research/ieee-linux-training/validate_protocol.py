"""Fail-closed validator for the append-only native-Linux research specification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = Path(__file__).resolve().parent
BASE_PROTOCOL = STUDY_ROOT / "PROTOCOL.md"
BASE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.yaml"
# Current (newest) amendment: 0002 -> effective matrix 1.2.0, reserved-identity registry v2.
EFFECTIVE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.v1.2.0.json"
AMENDMENT = STUDY_ROOT / "amendments/0002-2026-07-15-post-audit-v5-identities.md"
AMENDMENT_MANIFEST = STUDY_ROOT / (
    "amendments/0002-2026-07-15-post-audit-v5-identities.manifest.json"
)
RESERVED_IDENTITIES = STUDY_ROOT / "amendments/RESERVED_IDENTITIES.v2.json"
# Frozen prior amendment (0001 -> effective matrix 1.1.0). The current amendment supersedes it; the
# chain is verified below so 0001 stays byte-frozen and the amendment ordering is provable.
PRIOR_AMENDMENT = STUDY_ROOT / "amendments/0001-2026-07-15-manager-1.3-blue-green-identities.md"
PRIOR_AMENDMENT_MANIFEST = STUDY_ROOT / (
    "amendments/0001-2026-07-15-manager-1.3-blue-green-identities.manifest.json"
)
PRIOR_EFFECTIVE_MATRIX = STUDY_ROOT / "EXPERIMENT_MATRIX.v1.1.0.json"
PRIOR_RESERVED_IDENTITIES = STUDY_ROOT / "amendments/RESERVED_IDENTITIES.v1.json"
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
    """Bind the frozen prior amendment by exact hash so the amendment chain is ordered and 0001 is
    provably unmodified. A superseding amendment records the prior effective version and the raw-byte
    hashes of the prior manifest, narrative, effective matrix, and reserved-identity set."""

    supersedes = manifest.get("supersedes")
    if not isinstance(supersedes, dict):
        raise ProtocolValidationError("amendment must record the superseded prior amendment")
    if supersedes.get("effective_protocol_version") != "1.1.0":
        raise ProtocolValidationError("amendment must supersede exactly effective version 1.1.0")
    prior_files = {
        "prior_amendment_manifest_sha256": PRIOR_AMENDMENT_MANIFEST,
        "prior_narrative_sha256": PRIOR_AMENDMENT,
        "prior_effective_matrix_sha256": PRIOR_EFFECTIVE_MATRIX,
        "prior_reserved_identities_sha256": PRIOR_RESERVED_IDENTITIES,
    }
    for field, path in prior_files.items():
        _require_hash(_sha256(path), supersedes.get(field), f"superseded {field}")


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
    _validate_affected_counts(effective)

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
