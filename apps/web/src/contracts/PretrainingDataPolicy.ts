/* GENERATED from docs/contracts/PretrainingDataPolicy.schema.json — do not edit. Run: npm run gen:contracts */

export type ContractVersion = "1.0.0";
export type DataSeed = number;
export type DocumentBoundaries = boolean;
export type Epochs = number | null;
export type GlobalBatchSize = number;
export type Packing = "none" | "concat_and_split" | "best_fit";
export type ContentSha256 = string;
export type Location = string;
export type RowCount = number;
export type ShardId = string;
export type Source = string;
export type TokenCount = number;
export type Shards = PretrainingShard[];
export type Streaming = boolean;
export type TokenBudget = number | null;

/**
 * Additive, dense/MoE-safe pretraining data policy (#487), PARALLEL to the SFT-only
 * ``TrainingDataPolicy`` - never reuse the SFT contract for a sharded / streamed / mixture-weighted
 * corpus. It declares a content-hashed shard set, streaming, per-source mixture weights, document
 * boundaries, pretraining packing, a seeded deterministic global order, and a stop condition (token
 * budget and/or epochs) so a run stops at the budget and never silently truncates. The runtime
 * per-rank data cursor + streaming resume is a separate (worker) slice.
 */
export interface PretrainingDataPolicy {
  contract_version?: ContractVersion;
  data_seed: DataSeed;
  document_boundaries?: DocumentBoundaries;
  epochs?: Epochs;
  global_batch_size: GlobalBatchSize;
  mixture_weights?: MixtureWeights;
  packing?: Packing;
  shards: Shards;
  streaming?: Streaming;
  token_budget?: TokenBudget;
}
export interface MixtureWeights {
  [k: string]: number;
}
/**
 * One content-hashed corpus shard in a :class:`PretrainingDataPolicy`: a stable id + location, its
 * row and token counts, its sha256, and the mixture source it belongs to. The token count feeds the
 * token budget; the sha256 pins the exact bytes so a resumed stream reads the same shard.
 */
export interface PretrainingShard {
  content_sha256: ContentSha256;
  location: Location;
  row_count: RowCount;
  shard_id: ShardId;
  source?: Source;
  token_count: TokenCount;
}
