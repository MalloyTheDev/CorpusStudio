/* GENERATED from docs/contracts/ObjectiveCompatibilityReport.schema.json — do not edit. Run: npm run gen:contracts */

export type Evidence = string[];
export type Reasons = string[];
export type ObjectiveCompatibilityStatus =
  "declared_compatible" | "verified_compatible" | "incompatible" | "unverified" | "not_applicable";
export type BackendId = string | null;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ContractVersion = "1.0.0";
export type DatasetSchemaId = string | null;
export type DatasetSchemaVersion = string | null;
export type ModelId = string | null;
export type Notes = string[];
export type ObjectiveVersion = string;

/**
 * Independent compatibility axes. A static backend declaration can earn only
 * ``declared_compatible``; a matching functional capability report is required for verified
 * compatibility.
 */
export interface ObjectiveCompatibilityReport {
  backend: ObjectiveCompatibilityAxis;
  backend_id?: BackendId;
  capability_environment_ref?: Ref | null;
  contract_version?: ContractVersion;
  dataset: ObjectiveCompatibilityAxis;
  dataset_schema_id?: DatasetSchemaId;
  dataset_schema_version?: DatasetSchemaVersion;
  model: ObjectiveCompatibilityAxis;
  model_id?: ModelId;
  notes?: Notes;
  objective_ref: Ref;
  objective_version: ObjectiveVersion;
  overall_status: ObjectiveCompatibilityStatus;
}
export interface ObjectiveCompatibilityAxis {
  evidence?: Evidence;
  reasons?: Reasons;
  status: ObjectiveCompatibilityStatus;
}
/**
 * A stable reference to another contract instance by id, optionally pinned to a content hash so
 * the reference cannot silently re-point.
 */
export interface Ref {
  hash?: HashRef | null;
  id: Id;
}
/**
 * An algorithm-tagged digest. The engine emits sha256 today; the algo tag makes a future
 * migration additive (cf. versions/version_registry.FINGERPRINT_ALGO).
 */
export interface HashRef {
  algo?: Algo;
  value?: Value;
}
