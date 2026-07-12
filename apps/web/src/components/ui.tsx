import type { ReactNode } from "react";

export type Tone = "accent" | "ok" | "warn" | "bad" | "neutral";

export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="cs-eyebrow">{children}</p>;
}

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="cs-card">
      {title ? <h2 className="cs-card-title">{title}</h2> : null}
      {children}
    </section>
  );
}

export function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="cs-row">
      <span className="cs-key">{k}</span>
      <span className="cs-val">{children}</span>
    </div>
  );
}

export function Chip({ tone = "accent", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`cs-chip ${tone}`}>{children}</span>;
}

export function Chips({ items, tone = "accent" }: { items: string[]; tone?: Tone }) {
  return (
    <div className="cs-chips">
      {items.length ? (
        items.map((it) => (
          <Chip key={it} tone={tone}>
            {it}
          </Chip>
        ))
      ) : (
        <span className="cs-note">none</span>
      )}
    </div>
  );
}

/** A short middle-truncated hash for a mono readout. */
export function Hash({ value }: { value: string }) {
  const short = value.length > 20 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
  return <span className="cs-mono">{short}</span>;
}
