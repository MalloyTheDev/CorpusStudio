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
 *  predicted fit, and — once a run exists — its manifest + event stream. A LIVE pre-run snapshot
 *  (probe → plan, nothing executed yet) leaves manifest/events undefined; the sample carries a
 *  completed demo run. */
export interface PlatformSnapshot {
  profile: EnvironmentProfile;
  capabilities: CapabilityReport;
  plan: RunPlan;
  fit: FitClassification;
  manifest?: RunManifest;
  events?: RunEvent[];
}

/** Inputs the host can't decide — the "goal + data" the user supplies to resolve a plan. */
export interface PlanInputs {
  baseModel: string;
  dataset: string;
  sequenceLen: number;
  backend: string;
}

/** True when running inside the Tauri webview (vs. a plain browser dev/preview build). */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** The committed demo snapshot — a REAL engine-generated 5070/Blackwell bundle (produced by the
 *  actual planner + calibrator + supervisor). The browser build always renders this; inside Tauri it
 *  is the initial view until you probe the live host. */
export async function loadSnapshot(): Promise<PlatformSnapshot> {
  return sample as unknown as PlatformSnapshot;
}

/** Profile the real host via the Tauri command (shells `corpus-studio platform-probe --json --cache`).
 *  Only callable inside Tauri; the browser build renders the sample instead. */
export async function probeHost(): Promise<{
  profile: EnvironmentProfile;
  capabilities: CapabilityReport;
}> {
  const { invoke } = await import("@tauri-apps/api/core");
  const bundle = await invoke<{
    environment_profile: EnvironmentProfile;
    capability_report: CapabilityReport;
  }>("platform_probe");
  return { profile: bundle.environment_profile, capabilities: bundle.capability_report };
}

/** Resolve a hash-sealed RunPlan + its predicted fit for the given inputs, on the chosen backend
 *  (shells `corpus-studio platform-plan --json --backend …`). Only callable inside Tauri. Rejects
 *  with the engine's message when the host is unready or the backend can't run the plan. */
export async function resolvePlan(inputs: PlanInputs): Promise<{
  plan: RunPlan;
  fit: FitClassification;
}> {
  const { invoke } = await import("@tauri-apps/api/core");
  const bundle = await invoke<{ run_plan: RunPlan; fit_classification: FitClassification }>(
    "platform_plan",
    {
      baseModel: inputs.baseModel,
      dataset: inputs.dataset,
      sequenceLen: inputs.sequenceLen,
      backend: inputs.backend,
    },
  );
  return { plan: bundle.run_plan, fit: bundle.fit_classification };
}

/** The full live pre-run snapshot for the real host: probe it, then resolve a plan against it. No run
 *  has executed, so manifest/events are absent (the view shows a "not launched" run state). */
export async function buildLiveSnapshot(inputs: PlanInputs): Promise<PlatformSnapshot> {
  const { profile, capabilities } = await probeHost();
  const { plan, fit } = await resolvePlan(inputs);
  return { profile, capabilities, plan, fit };
}
