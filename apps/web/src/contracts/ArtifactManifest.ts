/* GENERATED from docs/contracts/ArtifactManifest.schema.json — do not edit. Run: npm run gen:contracts */

export type ArtifactId = string;
export type BaseModel = string | null;
export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type CheapFingerprint = string | null;
export type ContentHash = string | null;
export type CurrentIntegrity = "ok" | "missing" | "modified" | "unknown";
export type Kind = "adapter" | "checkpoint" | "merged_model" | "gguf" | "onnx" | "quantized" | "other";
export type Notes = string;
export type Path = string;
export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ReloadVerified = boolean;
export type Status = "candidate" | "kept" | "rejected";
export type UpdatedAt = string | null;

/**
 * A first-class record of a weight artifact a run produced. Formalizes
 * artifact_registry.ModelArtifactRecord + its two-tier integrity model. The platform NEVER
 * moves/copies/deletes the underlying weights — the manifest only references + re-checks them.
 */
export interface ArtifactManifest {
  artifact_id: ArtifactId;
  base_model?: BaseModel;
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  integrity?: ArtifactIntegrity | null;
  kind?: Kind;
  notes?: Notes;
  path: Path;
  producer_run_ref: Ref;
  reload_verified?: ReloadVerified;
  status?: Status;
  updated_at?: UpdatedAt;
}
/**
 * Two-tier integrity: cheap size+mtime fingerprint powers the fast LIST; content_hash (sha256
 * over weight bytes) powers the promote GATE. ``current_integrity`` is computed LIVE at read.
 */
export interface ArtifactIntegrity {
  cheap_fingerprint?: CheapFingerprint;
  content_hash?: ContentHash;
  current_integrity?: CurrentIntegrity;
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
