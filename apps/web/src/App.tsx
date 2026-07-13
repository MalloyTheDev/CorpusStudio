import { useEffect, useState } from "react";
import {
  buildLiveSnapshot,
  isTauri,
  loadSnapshot,
  resolvePlan,
  type PlanInputs,
  type PlatformSnapshot,
} from "./platform/api";
import { PlatformView } from "./components/PlatformView";

type Theme = "dark" | "light";
type Mode = "sample" | "live";

export default function App() {
  const [sampleSnap, setSampleSnap] = useState<PlatformSnapshot | null>(null);
  const [liveSnap, setLiveSnap] = useState<PlatformSnapshot | null>(null);
  const [mode, setMode] = useState<Mode>("sample");
  const [theme, setTheme] = useState<Theme>("dark");
  const [inputs, setInputs] = useState<PlanInputs>({
    baseModel: "Qwen/Qwen2.5-7B-Instruct",
    modelRevision: "a09a35458c702b33eeacc393d103063234e8bc28",
    dataset: "data/train.jsonl",
    sequenceLen: 4096,
    backend: "corpus_studio",
  });
  const [busy, setBusy] = useState(false);
  const [replanning, setReplanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const live = isTauri();

  useEffect(() => {
    loadSnapshot().then(setSampleSnap);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const probeAndPlan = async (): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      setLiveSnap(await buildLiveSnapshot(inputs));
    } catch (e) {
      setError(String(e));
      setLiveSnap(null);
    } finally {
      setBusy(false);
    }
  };

  // Picking a backend in live mode re-resolves the plan against the already-probed host.
  const replan = async (backend: string): Promise<void> => {
    setReplanning(true);
    setError(null);
    try {
      const { plan, fit } = await resolvePlan({ ...inputs, backend });
      // Commit the backend to inputs ONLY on success, so a refused backend doesn't linger and get
      // silently reused by the next "Probe & plan".
      setInputs((i) => ({ ...i, backend }));
      setLiveSnap((s) => (s ? { ...s, plan, fit } : s));
    } catch (e) {
      setError(String(e));
    } finally {
      setReplanning(false);
    }
  };

  // Serialize probe + replan: while either is in flight, the form submit and the picker are disabled,
  // so a concurrent probe can't clobber a replan's plan (or vice-versa).
  const pending = busy || replanning;

  const switchMode = (next: Mode): void => {
    setMode(next);
    setError(null); // a live-host error must not bleed onto the committed sample view (or vice-versa)
  };

  const shown = mode === "live" ? liveSnap : sampleSnap;

  return (
    <div className="cs-app">
      <header className="cs-topbar">
        <div className="cs-brand">C</div>
        <div>
          <div className="cs-title">Corpus Studio</div>
          <div className="cs-subtitle">Platform · run lifecycle</div>
        </div>
        <div className="cs-spacer" />
        <div className="cs-segment" role="tablist" aria-label="Data source">
          <button
            className={`cs-seg${mode === "sample" ? " on" : ""}`}
            role="tab"
            aria-selected={mode === "sample"}
            onClick={() => switchMode("sample")}
          >
            Sample
          </button>
          <button
            className={`cs-seg${mode === "live" ? " on" : ""}`}
            role="tab"
            aria-selected={mode === "live"}
            disabled={!live}
            title={live ? "Probe this host" : "Live host runs inside the desktop shell"}
            onClick={() => switchMode("live")}
          >
            Live host
          </button>
        </div>
        <button className="cs-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "☾ Dark" : "☀ Light"}
        </button>
      </header>

      {mode === "live" ? (
        <form
          className="cs-planbar"
          onSubmit={(e) => {
            e.preventDefault();
            void probeAndPlan();
          }}
        >
          <label className="cs-field cs-field-grow">
            <span>Base model</span>
            <input
              value={inputs.baseModel}
              onChange={(e) => setInputs((i) => ({ ...i, baseModel: e.target.value }))}
            />
          </label>
          <label className="cs-field cs-field-grow">
            <span>Dataset</span>
            <input
              value={inputs.dataset}
              onChange={(e) => setInputs((i) => ({ ...i, dataset: e.target.value }))}
            />
          </label>
          <label className="cs-field cs-field-grow">
            <span>Model revision (immutable)</span>
            <input
              value={inputs.modelRevision}
              onChange={(e) => setInputs((i) => ({ ...i, modelRevision: e.target.value }))}
              placeholder="40-character Hub commit; blank for a local model directory"
            />
          </label>
          <label className="cs-field">
            <span>Seq len</span>
            <input
              type="number"
              min={1}
              value={inputs.sequenceLen}
              onChange={(e) =>
                setInputs((i) => ({ ...i, sequenceLen: Number(e.target.value) || 1 }))
              }
            />
          </label>
          <button className="cs-btn cs-btn-primary" type="submit" disabled={pending}>
            {busy ? "Probing…" : "Probe & plan"}
          </button>
        </form>
      ) : null}

      {mode === "live" && error ? (
        <div className="cs-body">
          <div className="cs-error">
            <strong>Engine error.</strong> {error}
          </div>
        </div>
      ) : null}

      {shown ? (
        <PlatformView
          snap={shown}
          onPickBackend={mode === "live" ? replan : undefined}
          busy={pending}
        />
      ) : mode === "live" ? (
        <div className="cs-body cs-note">
          Enter a base model + dataset, then probe your host to resolve a plan.
        </div>
      ) : (
        <div className="cs-body">Loading…</div>
      )}
    </div>
  );
}
