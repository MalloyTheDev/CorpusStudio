# Corpus Studio — Tauri + React shell

A cross-platform client over the platform's **language-neutral contracts**. This is the eventual
production head (see [`docs/design/`](../../docs/design/)); the Avalonia head under
[`apps/desktop/`](../desktop/) is the interim. Both consume the **same design tokens** and the **same
JSON-Schema contracts** the engine emits, so this ports ~1:1.

## Architecture

```
engine (Python)  ──JSON-Schema contracts──►  TS types  ──►  React UI
  platform-*                docs/contracts/    src/contracts/   src/components/
  CLI (stdout=JSON)     (single source of truth, generated)
```

- **Contract-first.** `src/contracts/*.ts` is **generated** from `docs/contracts/*.schema.json` (the
  schemas the engine emits via `corpus-studio platform-schemas`). Regenerate with
  `npm run gen:contracts` — never hand-edit them. CI diffs the regenerated output to catch drift.
- **Nocturne tokens.** `src/theme/nocturne.css` is the web half of
  [`docs/design/nocturne-tokens.json`](../../docs/design/nocturne-tokens.json) — the same token set
  the Avalonia head uses as brushes. Theme = a `data-theme` swap (dark / light).
- **Engine as the source of truth.** The Tauri commands in `src-tauri/src/lib.rs` shell out to
  `corpus-studio platform-*` and return the parsed contract. The shell contains **no platform logic**.

## What it renders

The **run lifecycle** — profile → plan → predicted-fit → run — from real contract data: the host +
proven capabilities, the resolved `RunPlan` (with the Blackwell → math note), the **predicted**
`FitClassification` (honestly labeled *not measured*; a WDDM spill is distinguished from an OOM), and
the run's `RunManifest` + `RunEvent` stream. `src/platform/sample.json` is a real engine-generated
snapshot (a 5070 / Blackwell scenario) so the contract → UI pipeline is exercised end to end.

## Develop

```bash
npm install
npm run gen:contracts   # regenerate TS types from docs/contracts (committed; run after a contract change)
npm run typecheck       # tsc --noEmit
npm run build           # tsc + vite build (the CI-gated frontend build)
npm run dev             # vite dev server on :1420

# Desktop shell (needs the Rust toolchain + a WebView runtime):
npm run tauri dev       # run the Tauri window
npm run tauri build     # bundle the desktop app
```

## Status

Slice 1: the contract → TS → Nocturne-React pipeline + the platform lifecycle view (verified: type-check
+ vite build + a real engine-generated render). **Next:** wire the live host flow (probe → plan → run
against the local machine via the Tauri commands), then port the Studio screens 1:1 from the Avalonia head.
