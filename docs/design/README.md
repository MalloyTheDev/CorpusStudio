# Corpus Studio — Production UI (Nocturne)

The production design system + information architecture for the Corpus Studio desktop app. This is
the **single design source of truth for both shells**: the current **Avalonia** interim head and the
eventual **Tauri 2 + React** shell. Because the tokens + IA are framework-agnostic, the Avalonia UI
is being built to match this design so it **ports over to Tauri/React close to 1:1**.

## Files
- **[`nocturne-tokens.json`](nocturne-tokens.json)** — the canonical, framework-agnostic tokens
  (dark + light palettes, typography, radii, spacing, elevation). Avalonia consumes them as
  `ResourceDictionary` brushes (`App.axaml`); a React shell would consume them as CSS custom
  properties. Change a token here → update both consumers.
- **[`handoff/`](handoff/)** — the original high-fidelity design handoff (reference, not code):
  - `Corpus Studio.dc.html` — the full interactive reference (all screens, both themes). Open in a
    browser to view. It uses a proprietary `support.js` runtime **only** to render standalone —
    **ignore/do not port that runtime**; reimplement each surface in the shell's own components.
  - `README.md` — the authoritative spec (all tokens, every screen, interactions, state shape).
  - `_ds/.../styles.css` — the Nocturne base tokens (accent ramp, spacing, elevation).

## Design system: Nocturne
A quiet, compact UI — near-neutral blue-grey grounds, **Inter** throughout, 8px radii, a single
blurple accent used as line/glow (not floods), hairline elevation. Dark is primary; a full light
theme is a token-set swap (`.cs-root` / `.cs-root.cs-light` in the reference; `ThemeDictionaries`
Dark/Light in Avalonia). Icons: **Phosphor**.

## Information architecture (the core change)
The old flat 15-tab strip is replaced by a **grouped section → subsection sidebar** by workflow
phase, each group header carrying a per-phase status glyph reflecting pipeline readiness:

- **OVERVIEW** — Dashboard
- **AUTHOR** ✓ — Writing Studio · Examples · Import & Quarantine · Preference Review
- **MEASURE** ⚠ — Quality · Dataset Debt (grade badge) · Splits
- **EVALUATE** 🔒 (gated) — Evaluation · Model Arena · Suites
- **TRAIN** 🔒 (gated) — Training · Artifacts
- footer — Versions · Settings

Every panel is **live data from the engine** — the sample numbers in the reference mirror real engine
outputs (`build_debt_report`, `build_basic_quality_report`, `GateReport`, eval/arena/suite reports,
run/version records). The live surfaces map onto the platform contracts in
[`../contracts/`](../contracts/): Dashboard/Quality/Debt ← dataset+gate reports; Training/Artifacts ←
`RunManifest`/`RunEvent`/`ArtifactManifest`; Evaluation/Suites ← `EvaluationResult`.

## Migration status
- **Slice 1 (this):** tokens in the repo + handoff committed + the Nocturne palette applied to the
  Avalonia head (`App.axaml` ThemeDictionaries + shell chrome).
- **Next:** the grouped-workflow IA (replace the tab strip with the sectioned sidebar), then
  per-screen Nocturne fidelity (cards, status chips, hero/lifecycle strips). See
  [`AVALONIA_MIGRATION_PLAN.md`](AVALONIA_MIGRATION_PLAN.md).
