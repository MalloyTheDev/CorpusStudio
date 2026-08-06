/* GENERATED from docs/contracts/ModelCodeVettingReport.schema.json — do not edit. Run: npm run gen:contracts */

export type AnalyzerVersion = string;
export type BundleSha256 = string;
export type ContractVersion = "1.0.0";
export type EntrySymbol = string;
export type Code = string;
export type Lineno = number | null;
export type Message = string;
export type Severity = "error" | "warning";
export type Findings = VettingFinding[];
export type InterfaceVersion = "custom_decoder_v1";
export type Verdict = "admitted" | "rejected";

/**
 * The recorded result of STATICALLY screening a local custom-block bundle - the auditable admission
 * token for the mode-3 'your own model code' path (your own IMPLEMENTATION, not a borrowed family nor a
 * composed-from-standard-blocks design).
 *
 * Content-addressed evidence: it pins the exact bundle bytes it screened (``bundle_sha256``), so a plan
 * can bind admission to those bytes and nothing else. A static pre-screen is NECESSARY, NOT SUFFICIENT:
 * it executes nothing and does not prove the code safe - runtime containment is the (gated) worker
 * sandbox, and admission stays human-gated. This path never uses HF ``trust_remote_code`` (which stays
 * ``Literal[False]``); the module is loaded locally, by path, from the pinned bundle. ``verdict`` is
 * ``admitted`` iff there are no error-severity findings.
 */
export interface ModelCodeVettingReport {
  analyzer_version: AnalyzerVersion;
  bundle_sha256: BundleSha256;
  contract_version?: ContractVersion;
  entry_symbol: EntrySymbol;
  findings?: Findings;
  interface_version: InterfaceVersion;
  verdict: Verdict;
}
/**
 * A single finding from statically screening a custom-block bundle (nested in ModelCodeVettingReport,
 * not a root contract).
 */
export interface VettingFinding {
  code: Code;
  lineno?: Lineno;
  message: Message;
  severity: Severity;
}
