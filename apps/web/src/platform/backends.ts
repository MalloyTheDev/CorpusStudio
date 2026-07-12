import type { BackendManifest } from "../contracts";
import { isTauri, type PlatformSnapshot } from "./api";
import backendsSample from "./backends.sample.json";

/** The registered training backends — "pick your framework". Browser build renders the committed
 *  engine snapshot (a REAL `corpus-studio platform-backends --json`); inside Tauri it invokes the
 *  live command so the registry stays the engine's, never the shell's. */
export async function loadBackends(): Promise<BackendManifest[]> {
  if (isTauri()) {
    const { invoke } = await import("@tauri-apps/api/core");
    return invoke<BackendManifest[]>("platform_backends");
  }
  return backendsSample as unknown as BackendManifest[];
}

/** The requirements the engine's resolved plan imposes on a backend — read straight off the snapshot
 *  the planner + calibrator already produced. The engine stays authoritative; this just surfaces what
 *  it decided so the picker can show which frameworks fit. */
export interface PlanRequirements {
  os: string;
  device: string;
  taskType: string;
  precision: string;
  quantization: string;
  adapter: string;
  attention: string;
}

export function planRequirements(snap: PlatformSnapshot): PlanRequirements {
  return {
    os: snap.profile.host.os,
    device: snap.profile.gpus?.[0]?.kind ?? "cpu",
    taskType: snap.plan.task_type ?? "sft",
    precision: snap.plan.precision,
    quantization: snap.plan.quantization,
    adapter: snap.plan.adapter.method,
    attention: snap.plan.attention_backend,
  };
}

/** The reasons a backend can't run the resolved plan — empty when it can. A pure read of the
 *  backend's DECLARED support that mirrors the engine's `backends.unmet_requirements`. The planner is
 *  the authority (it already validated the chosen backend and forced `math` on Blackwell); this is the
 *  preview the picker renders so the honest filtering — e.g. Unsloth off a Blackwell math plan — is
 *  visible before you pick. */
export function unmetRequirements(b: BackendManifest, r: PlanRequirements): string[] {
  const reasons: string[] = [];
  const has = (arr: readonly string[] | undefined, v: string): boolean => (arr ?? []).includes(v);
  if (!has(b.supported_os, r.os)) reasons.push(`OS '${r.os}' not supported`);
  if (!has(b.supported_devices, r.device)) reasons.push(`device '${r.device}' not supported`);
  if (!has(b.task_types, r.taskType)) reasons.push(`task '${r.taskType}' not supported`);
  if (!has(b.precision_modes, r.precision)) reasons.push(`precision '${r.precision}' not supported`);
  if (!has(b.quantization_modes, r.quantization))
    reasons.push(`quantization '${r.quantization}' not supported`);
  if (!has(b.adapter_methods, r.adapter)) reasons.push(`adapter '${r.adapter}' not supported`);
  if (!has(b.attention_impls, r.attention)) reasons.push(`attention '${r.attention}' not supported`);
  return reasons;
}
