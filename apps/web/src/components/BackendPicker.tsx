import { useEffect, useMemo, useState } from "react";
import type { BackendManifest } from "../contracts";
import type { PlatformSnapshot } from "../platform/api";
import { loadBackends, planRequirements, unmetRequirements } from "../platform/backends";
import { Card, Chips, Eyebrow } from "./ui";

/** "Pick your framework." Lists the registered training backends and shows, honestly, which ones
 *  DECLARE support for the plan the engine already resolved — a Blackwell math plan filters Unsloth
 *  out (its fused kernels can't do math), it isn't silently downgraded. Picking a fitting backend
 *  reveals its export formats, optimizers, and pre-declared hazards. */
export function BackendPicker({ snap }: { snap: PlatformSnapshot }) {
  const [backends, setBackends] = useState<BackendManifest[] | null>(null);
  const reqs = useMemo(() => planRequirements(snap), [snap]);
  const [picked, setPicked] = useState<string>(snap.plan.backend_ref.id);

  useEffect(() => {
    loadBackends().then(setBackends);
  }, []);

  if (!backends) return null;
  const pickedBackend = backends.find((b) => b.backend_id === picked) ?? null;

  return (
    <section className="cs-backends-section">
      <Eyebrow>Pick your framework · declared support vs the resolved plan</Eyebrow>
      <div className="cs-backends">
        {backends.map((b) => {
          const unmet = unmetRequirements(b, reqs);
          const fits = unmet.length === 0;
          const isPicked = picked === b.backend_id;
          return (
            <button
              key={b.backend_id}
              type="button"
              className={`cs-backend${isPicked ? " picked" : ""}${fits ? "" : " unfit"}`}
              onClick={() => fits && setPicked(b.backend_id)}
              disabled={!fits}
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
        {reqs.attention === "math" ? " (Blackwell sm_120 → math)" : ""}. This preview shows which
        registered backends <em>declare</em> support for the resolved plan; one that doesn’t (Unsloth’s
        fused kernels can’t do the math path) is filtered out, not silently downgraded. In the live
        flow, picking a fitting backend re-plans through the engine.
      </p>
    </section>
  );
}
