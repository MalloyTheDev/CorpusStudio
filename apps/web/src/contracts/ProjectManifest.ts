/* GENERATED from docs/contracts/ProjectManifest.schema.json — do not edit. Run: npm run gen:contracts */

export type ContractVersion = "1.0.0";
export type CreatedAt = string | null;
export type DatasetPath = string;
export type Id = string;
export type Labels = string[];
export type Name = string;
export type DatasetVersionsDir = string;
export type GateReportsDir = string;
export type GateThresholdsFile = string;
export type ModelArtifactsDir = string;
export type TrainingRunsDir = string;
export type SchemaId = string;
export type Seed = number;
export type TrainRatio = number;
export type ValidationRatio = number;
export type UpdatedAt = string | null;

/**
 * The top-level descriptor of a project/workspace. Formalizes storage/project.DatasetProject
 * (project.json) + SplitSettings, promoting it to the workspace-primary manifest a UI shell opens;
 * everything else resolves live from the referenced registries.
 */
export interface ProjectManifest {
  contract_version?: ContractVersion;
  created_at?: CreatedAt;
  dataset_path?: DatasetPath;
  id: Id;
  labels?: Labels;
  name: Name;
  registries?: ProjectRegistries;
  schema_id: SchemaId;
  split_settings?: SplitSettings;
  updated_at?: UpdatedAt;
}
/**
 * Project-relative directories the engine already uses for durable records. Pointers only;
 * contents are resolved live.
 */
export interface ProjectRegistries {
  dataset_versions_dir?: DatasetVersionsDir;
  gate_reports_dir?: GateReportsDir;
  gate_thresholds_file?: GateThresholdsFile;
  model_artifacts_dir?: ModelArtifactsDir;
  training_runs_dir?: TrainingRunsDir;
}
/**
 * Verbatim from storage/project.SplitSettings. Seed pins deterministic splitting.
 */
export interface SplitSettings {
  seed?: Seed;
  train_ratio?: TrainRatio;
  validation_ratio?: ValidationRatio;
}
