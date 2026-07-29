/* GENERATED from docs/contracts/BackendExecutionVariant.schema.json — do not edit. Run: npm run gen:contracts */

export type BackendId = string;
export type ContractVersion = "1.0.0";
/**
 * How far an execution variant has been proven, as a closed ladder (a Boolean 'verified' is too
 * easy to set without saying what was proven). ``declared`` = exists in the capability model only;
 * ``contract_validated`` = its descriptor + invariants are test-covered but no worker path is
 * established; ``worker_implemented`` = an execution path exists but the real-workload evidence has
 * not passed; ``workload_verified`` = the real training workload + artifact-verification gates
 * passed. Only ``workload_verified`` may pass the execution admission gate.
 */
export type ExecutionVariantSupport = "declared" | "contract_validated" | "worker_implemented" | "workload_verified";
/**
 * An execution variant of the ONE canonical training harness (not a separate harness). The
 * dense-QLoRA-SFT variant is the sealed, workload-verified path (``ResolvedExecutionConfiguration``);
 * the others are contract-expressible modes that are NOT executable until separately implemented,
 * measured, and admitted. The set is deliberately small - only variants justified by current scope.
 */
export type ExecutionVariantKind = "dense_qlora_sft" | "dense_full_finetune" | "pretraining" | "moe";

/**
 * A control-plane CAPABILITY DESCRIPTOR: a backend's support for one execution variant of the
 * canonical training harness. It is NOT the sealed worker configuration
 * (:class:`ResolvedExecutionConfiguration`) and is never passed to a worker as one; the variant's
 * capability envelope (task type, PEFT/full-parameter, dataset cardinality) is DERIVED from
 * ``variant_kind`` (see ``execution_variants.variant_envelope``), so a contradictory envelope cannot
 * be expressed. Declaring a variant does not make it executable - only ``workload_verified`` passes
 * the execution admission gate.
 */
export interface BackendExecutionVariant {
  backend_id: BackendId;
  contract_version?: ContractVersion;
  support?: ExecutionVariantSupport;
  variant_kind: ExecutionVariantKind;
}
