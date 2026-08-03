/* GENERATED from docs/contracts/PreferenceDataPolicy.schema.json — do not edit. Run: npm run gen:contracts */

export type ChatTemplateSha256 = string | null;
export type ContractVersion = "1.0.0";
export type DataSeed = number;
export type FormatterId = string;
export type FormatterSha256 = string;
export type MaxLength = number;
export type MaxPromptLength = number;
export type PairSchema = "chosen_rejected" | "preference_pair";
export type TruncationPolicy = "refuse" | "allow";

/**
 * Additive, dense/MoE-safe preference-pair data policy (S2 / DPO), PARALLEL to the SFT-only
 * ``TrainingDataPolicy`` - never reuse the SFT contract for preference pairs. It pins the exact
 * chosen/rejected pair schema + formatter + chat template + the DPO prompt/response length budget, so a
 * preference run formats every pair identically and refuses (never silently truncates) an over-length
 * prompt or response. The reference model + DPO loss hyperparameters live on the DPO execution seal
 * (a separate worker slice), not here - this is only the data contract.
 */
export interface PreferenceDataPolicy {
  chat_template_sha256?: ChatTemplateSha256;
  contract_version?: ContractVersion;
  data_seed?: DataSeed;
  formatter_id: FormatterId;
  formatter_sha256: FormatterSha256;
  max_length: MaxLength;
  max_prompt_length: MaxPromptLength;
  pair_schema?: PairSchema;
  truncation_policy?: TruncationPolicy;
}
