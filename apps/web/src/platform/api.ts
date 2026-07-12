import sample from "./sample.json";
import type {
  CapabilityReport,
  EnvironmentProfile,
  FitClassification,
  RunEvent,
  RunManifest,
  RunPlan,
} from "../contracts";

/** One end-to-end platform snapshot: the host, its proven capabilities, the resolved plan, the
 *  predicted fit, and a run's manifest + event stream — the run lifecycle in one object. */
export interface PlatformSnapshot {
  profile: EnvironmentProfile;
  capabilities: CapabilityReport;
  plan: RunPlan;
  fit: FitClassification;
  manifest: RunManifest;
  events: RunEvent[];
}

/** True when running inside the Tauri webview (vs. a plain browser dev/preview build). */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** Load the platform snapshot to render. Today this is a REAL engine-generated bundle (a 5070 /
 *  Blackwell scenario produced by the actual planner + calibrator + supervisor, committed as
 *  sample.json) so the contract → UI pipeline is exercised end-to-end. Wiring the live host flow
 *  (probe → plan → run against your machine, via the Tauri commands below) is the next slice. */
export async function loadSnapshot(): Promise<PlatformSnapshot> {
  return sample as unknown as PlatformSnapshot;
}

/** Profile the real host via the Tauri command (which shells `corpus-studio platform-probe --json`).
 *  Only callable inside Tauri; the browser dev build renders the sample instead. */
export async function probeHost(): Promise<{
  profile: EnvironmentProfile;
  capabilities: CapabilityReport;
}> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<{ profile: EnvironmentProfile; capabilities: CapabilityReport }>("platform_probe");
}
