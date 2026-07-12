import { useEffect, useState } from "react";
import { loadSnapshot, type PlatformSnapshot } from "./platform/api";
import { PlatformView } from "./components/PlatformView";

type Theme = "dark" | "light";

export default function App() {
  const [snap, setSnap] = useState<PlatformSnapshot | null>(null);
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    loadSnapshot().then(setSnap);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <div className="cs-app">
      <header className="cs-topbar">
        <div className="cs-brand">C</div>
        <div>
          <div className="cs-title">Corpus Studio</div>
          <div className="cs-subtitle">Platform · run lifecycle</div>
        </div>
        <div className="cs-spacer" />
        <button className="cs-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "☾ Dark" : "☀ Light"}
        </button>
      </header>
      {snap ? <PlatformView snap={snap} /> : <div className="cs-body">Loading…</div>}
    </div>
  );
}
