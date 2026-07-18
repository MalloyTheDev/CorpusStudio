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
- **[`handoff-v2/`](handoff-v2/)** — the v2 handoff assets: the actual **Phosphor SVG glyphs**
  (regular/fill/bold), the icon inventory/manifest (`ICONS.md` / `icons.json`), the Nocturne
  `styles.css` token source, and the full `SPEC.md`. These are the glyphs the Avalonia
  `Styles/Icons.axaml` inlines and that the future React shell imports via `@phosphor-icons/react`.

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
The Nocturne re-skin has landed across the cross-platform (Avalonia) shell:
- **Design system:** tokens + the full handoff (incl. the exact Phosphor SVGs, see
  [`handoff-v2/`](handoff-v2/)) committed; the `App.axaml` ThemeDictionaries carry the whole token set.
- **Shell:** the grouped workflow-phase sidebar (Overview · Author · Measure · Evaluate · Train)
  replaced the flat tab strip; a Phosphor icon system (inlined `StreamGeometry`), the activity bar,
  context bar, engine-status footer, and the contextual Quality rail.
- **Screens:** Nocturne card fidelity across Dashboard (hero + honest lifecycle strip), Quality,
  Dataset Debt, Splits, Writing Studio, Examples, Preference Review, Evaluation, Model Arena,
  Settings, Versions, Import & Quarantine, Suites, and Artifacts.
- **Remaining polish:** Writing-Studio structured-form split, Examples search/filter pills, a
  Dashboard recent-activity feed, the Splits proportion bar, and the docked-console styling.

The interim Avalonia head is being matured to this design so it ports ~1:1 toward the Tauri/React
endpoint. See [`AVALONIA_MIGRATION_PLAN.md`](../AVALONIA_MIGRATION_PLAN.md).
