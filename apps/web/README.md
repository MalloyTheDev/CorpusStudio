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
  `FitClassification` (honestly labeled *not measured*; a WDDM spill is distinguished from an OOM),
  the sealed effective attention kernel/device map/configuration hash, and the run's `RunManifest` +
  `RunEvent` stream. Live planning requires an immutable Hub model revision. The committed
  `src/platform/sample.json` is a pre-Phase-9B engine snapshot retained as migration evidence; the UI
  labels it `legacy plan - regenerate before execution` instead of treating it as current proof.

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

The contract → TS → Nocturne-React pipeline and live probe → immutable-revision plan flow are wired.
The lifecycle view displays the effective kernel, explicit device map, and nested execution hash, and
warns on legacy plans. The Tauri shell remains a thin CLI client; live run streaming/cancellation and
the remaining Studio screen port are future work.
