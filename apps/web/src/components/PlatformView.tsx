import type { PlatformSnapshot } from "../platform/api";
import type { FitClass } from "../contracts/FitClassification";
import { BackendPicker } from "./BackendPicker";
import { Card, Chip, Chips, Eyebrow, Hash, Row, type Tone } from "./ui";

const gb = (bytes: number | null | undefined): string =>
  bytes == null ? "—" : `${(bytes / 1_000_000_000).toFixed(1)} GB`;

const readinessTone = (readiness: string): Tone =>
  readiness === "ready" ? "ok" : readiness === "cpu_toy_only" ? "warn" : "bad";

const probeTone = (outcome: string): Tone => (outcome === "PASS" ? "ok" : "bad");

// A predicted-fit verdict is coloured by how worried we should be — never "safe green" from an
// estimate (the calibrator never emits NATIVE_SAFE without a measured run).
const fitTone = (c: FitClass): Tone => {
  if (c === "NATIVE_SAFE") return "ok";
  if (c.startsWith("NATIVE_") || c === "MARGINAL") return c === "MARGINAL" ? "warn" : "neutral";
  if (c.startsWith("CONTROLLED_")) return "warn";
  return "bad"; // ACCIDENTAL_* / THRASHING / FAIL
};

const Stage = ({ label }: { label: string }) => (
  <div className="cs-stage">
    <span className="cs-dot" />
    {label}
  </div>
);

export function PlatformView({
  snap,
  onPickBackend,
  busy = false,
}: {
  snap: PlatformSnapshot;
  onPickBackend?: (backendId: string) => void;
  /** A probe or re-plan is in flight — disables the backend picker so it can't race the request. */
  busy?: boolean;
}) {
  const gpu = snap.profile.gpus?.[0];
  const eff = snap.capabilities.effective_capabilities;
  const fit = snap.fit;
  const manifest = snap.manifest;

  return (
    <div className="cs-body">
      <div className="cs-lifecycle">
        <Stage label="Profile" />
        <span className="cs-connector" />
        <Stage label="Plan" />
        <span className="cs-connector" />
        <Stage label="Fit" />
        <span className="cs-connector" />
        <Stage label="Run" />
      </div>

      <Eyebrow>Run lifecycle · from language-neutral contracts</Eyebrow>
      <div className="cs-grid">
        <Card title="Environment">
          <Row k="GPU">{gpu?.name ?? "no GPU detected"}</Row>
          <Row k="VRAM">{gb(gpu?.vram_total_bytes)}</Row>
          <Row k="Compute capability">{gpu?.compute_capability ?? "—"}</Row>
          <Row k="Residency">{snap.profile.host.memory_residency_model ?? "unknown"}</Row>
          <Row k="Readiness">
            <Chip tone={readinessTone(snap.capabilities.readiness)}>
              {snap.capabilities.readiness}
            </Chip>
          </Row>
        </Card>

        <Card title="Proven capabilities">
          <div style={{ marginBottom: 10 }}>
            <div className="cs-key" style={{ marginBottom: 5 }}>
              precision · quantization · attention
            </div>
            <Chips
              items={[
                ...(eff?.precision_modes ?? []),
                ...(eff?.quantization_modes ?? []),
                ...(eff?.attention_impls ?? []),
              ]}
            />
          </div>
          {(snap.capabilities.probe_results ?? []).map((p) => (
            <Row key={p.probe} k={p.probe}>
              <Chip tone={probeTone(p.outcome)}>{p.outcome}</Chip>
            </Row>
          ))}
        </Card>

        <Card title="Resolved run plan">
          <Row k="Base model">{snap.plan.base_model}</Row>
          <Row k="Precision · quant">
            {snap.plan.precision} · {snap.plan.quantization}
          </Row>
          <Row k="Attention">
            {snap.plan.attention_backend}
            {snap.plan.attention_backend === "math" ? (
              <span className="cs-honest">Blackwell → math</span>
            ) : null}
          </Row>
          <Row k="Adapter">{snap.plan.adapter.method}</Row>
          <Row k="Sequence length">{snap.plan.sequence.max_sequence_len}</Row>
          <Row k="Plan hash">
            <Hash value={snap.plan.plan_hash} />
          </Row>
        </Card>

        <Card title="Predicted fit">
          <Row k="Verdict">
            <Chip tone={fitTone(fit.classification)}>{fit.classification}</Chip>
            <span className="cs-honest">predicted, not measured</span>
          </Row>
          <Row k="Estimated peak">{gb(fit.estimated_peak_bytes)}</Row>
          <Row k="Device capacity">{gb(fit.device_capacity_bytes)}</Row>
          <Row k="Headroom">{gb(fit.headroom_bytes)}</Row>
          {fit.rationale ? <p className="cs-rationale">{fit.rationale}</p> : null}
        </Card>

        <Card title="Run">
          {manifest ? (
            <>
              <Row k="State">
                <Chip tone={manifest.state === "succeeded" ? "ok" : "bad"}>{manifest.state}</Chip>
              </Row>
              <Row k="Run id">{manifest.run_id}</Row>
              <Row k="Runner">{manifest.target}</Row>
              <Row k="Artifacts">{(manifest.artifact_ids ?? []).join(", ") || "—"}</Row>
            </>
          ) : (
            <>
              <Row k="State">
                <Chip tone="neutral">not launched</Chip>
              </Row>
              <p className="cs-note">
                Plan resolved — nothing has executed yet. Launch it from the desktop shell
                (<span className="cs-mono">platform-run --runner training</span>); the live RunEvent
                stream lands here.
              </p>
            </>
          )}
        </Card>

        <Card title="Event stream">
          {snap.events?.length ? (
            <div className="cs-events">
              {snap.events.map((e) => (
                <div className="cs-event" key={e.seq}>
                  <span className="seq">{e.seq}</span>
                  <span className="kind">{e.event_type}</span>
                  <span>
                    {e.stage ?? ""}
                    {e.optimizer_step != null ? ` step ${e.optimizer_step}` : ""}
                    {e.metrics?.loss != null ? ` loss=${e.metrics.loss.toFixed(4)}` : ""}
                    {e.message ? ` ${e.message}` : ""}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="cs-note">No run yet — events appear once a run is launched.</p>
          )}
        </Card>
      </div>

      <BackendPicker snap={snap} onPick={onPickBackend} busy={busy} />

      <p className="cs-note">
        Rendered from the engine's language-neutral JSON-Schema contracts (docs/contracts) — the same
        boundary the Rust core and the Avalonia head consume.{" "}
        {onPickBackend
          ? "This is a live probe → plan against your host; launch a run from the desktop shell to populate the run stream."
          : "This is a real engine-generated snapshot; switch to “Live host” inside the desktop shell to probe your own machine."}
      </p>
    </div>
  );
}
