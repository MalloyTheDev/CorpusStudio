/* GENERATED from docs/contracts/WorkerMessage.schema.json — do not edit. Run: npm run gen:contracts */

export type Body = {
  [k: string]: unknown;
} | null;
export type CorrelationId = string | null;
export type Direction = "core_to_worker" | "worker_to_core";
export type MessageId = string;
export type ProtocolVersion = string;
export type SentAt = string | null;
export type Type =
  | "hello"
  | "capability_probe_request"
  | "capability_report"
  | "run_dispatch"
  | "run_accepted"
  | "run_rejected"
  | "run_control"
  | "event"
  | "heartbeat"
  | "terminal_result"
  | "failure";

/**
 * The versioned envelope for the core↔worker channel — realizes the 'immutable RunPlan IN,
 * structured RunEvent stream OUT' boundary. NEW; no worker/core protocol exists in the engine
 * today (the desktop owns the trainer process directly). ``protocol_version`` lets the two sides
 * negotiate compatibility independently of any single contract's version. The body shape is
 * selected by ``type`` (see :data:`WORKER_BODY_BY_TYPE`).
 */
export interface WorkerMessage {
  body?: Body;
  correlation_id?: CorrelationId;
  direction: Direction;
  message_id: MessageId;
  protocol_version: ProtocolVersion;
  sent_at?: SentAt;
  type: Type;
}
