/* GENERATED from docs/contracts/CheckpointManifest.schema.json — do not edit. Run: npm run gen:contracts */

export type Algo = "sha256" | "sha256-ordered-exact-v1" | "blake3" | "none";
export type Value = string | null;
export type Id = string;
export type ChatTemplateSha256 = string | null;
export type DataSeed = number;
export type EnvironmentLockHash = string | null;
export type ExecutionConfigurationHash = string;
export type FormatterSha256 = string | null;
export type PlanHash = string;
export type Seed = number;
export type WorkerWheelSha256 = string | null;
export type CheckpointId = string;
export type CheckpointManifestHash = string;
export type Complete = boolean;
export type ContractVersion = "1.0.0";
export type CreatedAt = string;
/**
 * @minItems 1
 */
export type Files = [CheckpointFileEntry, ...CheckpointFileEntry[]];
export type Path = string;
export type Role =
  "optimizer" | "scheduler" | "scaler" | "rng" | "sampler" | "trainer_state" | "adapter_weights" | "other";
export type Sha256 = string;
export type SizeBytes = number;
export type ParentCheckpointHash = string | null;
export type ParentCheckpointId = string | null;
export type SchemaKind = "checkpoint-manifest-v1";
export type SourceRunId = string;
export type ConsumedMicrosteps = number;
export type Epoch = number;
export type GlobalOptimizerStep = number;
export type GradientAccumulationSteps = number;
export type MicrostepWithinStep = number;
export type OptimizerCaptured = true;
export type RngAlgorithm = string | null;
export type RngCaptured = boolean;
export type SamplerStateCaptured = boolean;
export type ScalerCaptured = boolean;
export type SchedulerCaptured = boolean;

/**
 * A single hash-sealed checkpoint instance. ``complete`` is the atomic completion marker: it is
 * False until every file has been hashed and the manifest sealed, so a torn write is never mistaken
 * for a resumable checkpoint. ``checkpoint_manifest_hash`` is the canonical digest of the manifest
 * body (every field except the hash itself), and ``parent_checkpoint_hash`` chains lineage back to
 * the parent checkpoint by that same digest.
 */
export interface CheckpointManifest {
  bound: CheckpointBoundIdentities;
  checkpoint_id: CheckpointId;
  checkpoint_manifest_hash: CheckpointManifestHash;
  complete?: Complete;
  contract_version?: ContractVersion;
  created_at: CreatedAt;
  files: Files;
  parent_checkpoint_hash?: ParentCheckpointHash;
  parent_checkpoint_id?: ParentCheckpointId;
  schema_kind?: SchemaKind;
  source_run_id: SourceRunId;
  state: SealedTrainingState;
}
/**
 * Everything a resumed run must match to reuse this checkpoint. A mismatch on ANY field makes the
 * checkpoint inadmissible (incompatible) - the resume fails closed. ``environment_lock_hash`` /
 * ``worker_wheel_sha256`` are null only for an unmanaged (profile-snapshot) source run.
 */
export interface CheckpointBoundIdentities {
  backend_ref: Ref;
  chat_template_sha256?: ChatTemplateSha256;
  data_seed: DataSeed;
  dataset_ref: Ref;
  environment_lock_hash?: EnvironmentLockHash;
  execution_configuration_hash: ExecutionConfigurationHash;
  formatter_sha256?: FormatterSha256;
  model_ref: Ref;
  objective_ref: Ref;
  plan_hash: PlanHash;
  seed: Seed;
  tokenizer_ref: Ref;
  worker_wheel_sha256?: WorkerWheelSha256;
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
/**
 * One required file inside a checkpoint, pinned by exact bytes. ``path`` is relative to the
 * checkpoint directory and must be a canonical, non-escaping POSIX path.
 */
export interface CheckpointFileEntry {
  path: Path;
  role: Role;
  sha256: Sha256;
  size_bytes: SizeBytes;
}
/**
 * The resumable training state the worker sealed, described WITHOUT torch. Presence flags assert
 * each component was actually captured (its bytes live in a :class:`CheckpointFileEntry`); the
 * counters place the checkpoint exactly on the optimizer-step / microstep / epoch timeline so a
 * resume continues from the precise position, never an approximate one.
 */
export interface SealedTrainingState {
  consumed_microsteps: ConsumedMicrosteps;
  epoch: Epoch;
  global_optimizer_step: GlobalOptimizerStep;
  gradient_accumulation_steps: GradientAccumulationSteps;
  microstep_within_step: MicrostepWithinStep;
  optimizer_captured?: OptimizerCaptured;
  rng_algorithm?: RngAlgorithm;
  rng_captured: RngCaptured;
  sampler_state_captured: SamplerStateCaptured;
  scaler_captured: ScalerCaptured;
  scheduler_captured: SchedulerCaptured;
}
