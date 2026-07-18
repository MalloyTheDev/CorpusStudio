import { useEffect, useMemo, useState } from "react";
import type { BackendManifest } from "../contracts";
import type { PlatformSnapshot } from "../platform/api";
import { loadBackends, planRequirements, unmetRequirements } from "../platform/backends";
import { Card, Chips, Eyebrow } from "./ui";

/** "Pick your framework." Lists the registered training backends and shows, honestly, which ones
 *  DECLARE support for the plan the engine already resolved — a Blackwell math plan filters Unsloth
 *  out (its fused kernels can't do math), it isn't silently downgraded. Picking a fitting backend
 *  reveals its export formats, optimizers, and pre-declared hazards. */
export function BackendPicker({
  snap,
  onPick,
  busy = false,
}: {
  snap: PlatformSnapshot;
  /** When provided (live host flow), picking a fitting backend re-resolves the plan through the
   *  engine. Absent (sample) → picking only highlights + previews. */
  onPick?: (backendId: string) => void;
  /** A probe or re-plan is in flight — disables picking so it can't race the request. */
  busy?: boolean;
}) {
  const [backends, setBackends] = useState<BackendManifest[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reqs = useMemo(() => planRequirements(snap), [snap]);
  // The picked backend follows the resolved plan, so a live re-plan keeps the selection in sync.
  const picked = snap.plan.backend_ref.id;
  const [previewed, setPreviewed] = useState<string>(picked);
  const selected = onPick ? picked : previewed;

  useEffect(() => {
    loadBackends()
      .then(setBackends)
      .catch((e) => setError(String(e)));
  }, []);

  const choose = (backendId: string): void => {
    if (onPick) onPick(backendId);
    else setPreviewed(backendId);
  };

  if (error) {
    return (
      <section className="cs-backends-section">
        <div className="cs-error" role="alert">
          Could not load backends: {error}
        </div>
      </section>
    );
  }
  if (!backends) return null;
  const pickedBackend = backends.find((b) => b.backend_id === selected) ?? null;

  return (
    <section className="cs-backends-section">
      <Eyebrow>Pick your framework · declared support vs the resolved plan</Eyebrow>
      <div className="cs-backends">
        {backends.map((b) => {
          const unmet = unmetRequirements(b, reqs);
          const fits = unmet.length === 0;
          const isPicked = selected === b.backend_id;
          return (
            <button
              key={b.backend_id}
              type="button"
              className={`cs-backend${isPicked ? " picked" : ""}${fits ? "" : " unfit"}`}
              onClick={() => fits && !busy && choose(b.backend_id)}
              disabled={!fits || busy}
              aria-pressed={isPicked}
            >
              <div className="cs-backend-head">
                <span className="cs-backend-name">{b.display_name ?? b.backend_id}</span>
                <span className={`cs-chip ${fits ? "ok" : "bad"}`}>
                  {fits ? "fits this plan" : "can’t run"}
                </span>
              </div>
              <div className="cs-backend-caps">
                <Chips items={[...(b.supported_devices ?? [])]} tone="neutral" />
                <Chips
                  items={[...(b.precision_modes ?? []), ...(b.quantization_modes ?? [])]}
                  tone="neutral"
                />
                <Chips
                  items={[...(b.adapter_methods ?? []), ...(b.attention_impls ?? [])]}
                  tone="neutral"
                />
              </div>
              {fits ? null : (
                <ul className="cs-backend-reasons">
                  {unmet.map((why) => (
                    <li key={why}>{why}</li>
                  ))}
                </ul>
              )}
            </button>
          );
        })}
      </div>

      {pickedBackend ? (
        <Card title={`${pickedBackend.display_name ?? pickedBackend.backend_id} · details`}>
          <div className="cs-backend-detail">
            <div className="cs-key" style={{ marginBottom: 5 }}>
              export formats
            </div>
            <Chips items={[...(pickedBackend.export_formats ?? [])]} />
            <div className="cs-key" style={{ margin: "12px 0 5px" }}>
              optimizers
            </div>
            <Chips items={[...(pickedBackend.optimizers ?? [])]} tone="neutral" />
            {(pickedBackend.known_failure_modes ?? []).length ? (
              <>
                <div className="cs-key" style={{ margin: "14px 0 5px" }}>
                  declared hazards
                </div>
                {(pickedBackend.known_failure_modes ?? []).map((fm) => (
                  <div className="cs-hazard" key={`${fm.taxonomy}-${fm.condition}`}>
                    <span className="cs-chip warn">{fm.taxonomy}</span>
                    <div>
                      <div className="cs-hazard-cond">{fm.condition}</div>
                      {fm.mitigation ? <div className="cs-hazard-mit">→ {fm.mitigation}</div> : null}
                    </div>
                  </div>
                ))}
              </>
            ) : null}
          </div>
        </Card>
      ) : null}

      <p className="cs-note">
        The planner is authoritative — it already validated{" "}
        <strong>{snap.plan.backend_ref.id}</strong> for this plan and resolved{" "}
        <code>{reqs.attention}</code> attention
        {reqs.attention === "math" ? " (Blackwell sm_120 → math)" : ""}. This shows which registered
        backends <em>declare</em> support for the resolved plan; one that doesn’t (Unsloth’s fused
        kernels can’t do the math path) is filtered out, not silently downgraded.{" "}
        {onPick
          ? "Picking a fitting backend re-plans through the engine."
          : "This is the committed demo snapshot; picking previews a backend (live re-planning runs in the Tauri app)."}
      </p>
    </section>
  );
}
