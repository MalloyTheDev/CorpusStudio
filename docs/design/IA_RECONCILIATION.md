# Nocturne reference -> 7-area IA reconciliation

The **Nocturne** design under [`docs/design/`](README.md) is the UI/UX reference for the Tauri 2 +
React client (`apps/web`). It was drawn for the now-removed Avalonia head (#545) and predates the
**seven co-equal product areas** ([`../PRODUCT_AREAS.md`](../PRODUCT_AREAS.md)). Per the design
README, treat it as a **reference source, not a fixed spec**: the design *system* carries over almost
unchanged, but the top-level *information architecture* must be reworked. Where the reference conflicts
with the 7-area model or the live contracts (`../contracts/`), the current model wins.

This note is the map: what to **reuse as-is**, what to **adapt**, and what is **net-new**, so a UI
slice starts from a decision instead of re-deriving one.

## 1. Reuse as-is - the design system (framework-agnostic)

These port ~1:1 and are already partly implemented in `apps/web/src/theme/nocturne.css` +
`components.css`. Keep them as the shared substrate for every area, old screen or new:

- **Tokens** - [`nocturne-tokens.json`](nocturne-tokens.json) (dark + light palettes, typography,
  radii, spacing, elevation) -> CSS custom properties.
- **Type / shape / accent** - Inter throughout; 8px radii; a single blurple accent used as line/glow,
  never as floods; hairline elevation; dark primary with a light theme as a token-set swap.
- **Icons** - the Phosphor glyph set in [`handoff-v2/assets/`](handoff-v2/assets/) (regular/fill/bold),
  consumed via `@phosphor-icons/react`.
- **Component language** - cards, key/value rows, chips, contextual rails, the "honest lifecycle"
  strip. `apps/web/src/components/ui.tsx` is the current expression of this.

Do **not** port the handoff's `support.js` runtime (it exists only to render the standalone HTML);
reimplement each surface in the shell's own components.

## 2. Rework - information architecture (5 workflow groups -> 7 co-equal areas)

The prototype nav is a **workflow-phase** sidebar, data-authoring-centric, with training and
evaluation demoted to *gated* sub-sections:

> OVERVIEW · AUTHOR (checkmark) · MEASURE (warn) · EVALUATE (gated) · TRAIN (gated) · footer

The target nav is **seven first-class equals** ([`../PRODUCT_AREAS.md`](../PRODUCT_AREAS.md)) - dataset
engineering, evaluation, release, Behavior Lab, and evidence are **not** subordinate to training:

> Data Studio · Training Studio · Evaluation Studio · Behavior Lab (gated) · Model & Release Studio ·
> Environment & Hardware · Evidence & Experiments

The reference's per-group **status glyph** (a header icon reflecting pipeline readiness) is a good
idea worth keeping - re-scope it to a **per-area** readiness indicator rather than a linear-pipeline
gate.

## 3. Screen -> area disposition

| Nocturne screen | Prototype group | -> Product area | Disposition |
| --- | --- | --- | --- |
| Dashboard | OVERVIEW | Cross-area home | **Adapt** - reframe the data-first landing into a 7-area overview (per-area readiness at a glance); reuse the hero + honest-lifecycle strip |
| Writing Studio | AUTHOR | Data Studio | **Reuse** (visual + interaction) |
| Examples | AUTHOR | Data Studio | **Reuse** - `apps/web/src/components/DataStudio.tsx` is the current start |
| Import & Quarantine | AUTHOR | Data Studio (import & ingestion; validation) | **Reuse** |
| Preference Review | AUTHOR | Data Studio (annotation) | **Reuse** |
| Quality | MEASURE | Data Studio (validation) | **Reuse** |
| Dataset Debt | MEASURE | Data Studio (validation / provenance) | **Reuse** |
| Splits | MEASURE | Data Studio (versioning / validation) | **Reuse** |
| Evaluation | EVALUATE | Evaluation Studio | **Reuse** - un-gate |
| Model Arena | EVALUATE | Evaluation Studio (A/B comparison) | **Reuse** - un-gate |
| Suites | EVALUATE | Evaluation Studio (benchmarks / regression) | **Reuse** - un-gate |
| Training | TRAIN | Training Studio | **Adapt** - un-gate to co-equal; fold in the Platform run-lifecycle (section 4) |
| Artifacts | TRAIN | Model & Release Studio | **Adapt** - re-home from Train to Release (inspection, merge, quantize, convert, model cards, export) |
| Versions | footer | Data Studio + Evidence & Experiments | **Adapt** - split: dataset versions -> Data Studio; run/model lineage -> Evidence & Experiments |
| Settings | footer | Cross-cutting (+ Environment & Hardware) | **Reuse** - keep cross-cutting; env/runtime settings sit under Environment & Hardware |

## 4. Net-new - no prototype equivalent

- **Behavior Lab** (gated) - activation analysis, steering, causal attribution, behavior modification,
  weight surgery, capability-preservation testing. Design fresh within Nocturne; gated per
  [`../PRODUCT_VS_RESEARCH.md`](../PRODUCT_VS_RESEARCH.md).
- **Environment & Hardware** as a top-level area - dependency isolation, reproducible environments,
  GPU capability checks, resource planning, runtime health (the Environment Manager). The Platform
  view's **Environment** and **Proven capabilities** cards are a partial seed, not the whole area.
- **Evidence & Experiments** as a top-level area - run lineage, telemetry, reproducibility, scientific
  comparison, and the opt-in IEEE research overlay. The prototype's footer **Versions** is only a seed.
- **Platform run-lifecycle** surface - `apps/web/src/components/PlatformView.tsx`
  (probe -> plan -> predict-fit -> run, plus the backend picker and the live `RunEvent` stream wired in
  #513). This is **already built** and has no prototype screen; it spans Training Studio (run
  management), Environment & Hardware (capability), and Evidence & Experiments (telemetry).

## 5. Using this map

- **Porting a screen that exists in Nocturne:** reuse its visual + interaction design, re-home it under
  its product area from the table above, and bind it to the live JSON-Schema contract in
  [`../contracts/`](../contracts/) (the reference's sample numbers mirror real engine reports).
- **Building an area with no prototype:** design fresh *within* the Nocturne system (tokens +
  component language), following that area's responsibilities in `PRODUCT_AREAS.md`.
- **Navigation:** build the 7-area primary nav first; do not reproduce the 5-group workflow strip.
- **Honesty carries over:** the reference's "every panel is live engine data" and the
  predicted-vs-measured / gated-capability cues are invariants, not decoration - keep them.

The full visual reference is [`handoff/Corpus Studio.dc.html`](handoff/) (all screens, both themes);
the written spec is [`handoff/README.md`](handoff/README.md) and
[`handoff-v2/SPEC.md`](handoff-v2/SPEC.md).
