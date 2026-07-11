# Corpus Studio — Production UI handoff (v2)

The **v2** design handoff for the Corpus Studio production UI (Nocturne). Where the v1
[`../handoff/`](../handoff/) bundle shipped the written spec plus the interactive HTML reference,
**this bundle adds the durable, reusable assets** the v1 bundle lacked: the actual **Phosphor icon
SVGs**, the icon **inventory/manifest**, and the **Nocturne token source** (`styles.css`). It is the
source-of-truth glyph + token set that both shells consume.

## What's here

| Path | What it is |
|---|---|
| [`assets/icons/{regular,fill,bold}/*.svg`](assets/icons/) | The **71 Phosphor glyphs** (59 icons × weight variants) used across the design, as raw SVG (`fill="currentColor"`, 256×256 viewBox). Phosphor Icons **v2.1.1**, MIT. |
| [`assets/ICONS.md`](assets/ICONS.md) | Icon inventory: each glyph → its UI role → its `@phosphor-icons/react` component name, plus React/font usage notes. |
| [`assets/icons.json`](assets/icons.json) | Machine-readable manifest of the same (name → React component → weights → role → import line). |
| [`_ds/nocturne-557b8352-4f53-4f98-9e8f-4873cfb3aa97/styles.css`](_ds/nocturne-557b8352-4f53-4f98-9e8f-4873cfb3aa97/styles.css) | The **Nocturne token source** — the canonical OKLCH ramps, accent, spacing, and elevation the app layers its semantic tokens on top of. |
| [`SPEC.md`](SPEC.md) | The full written design spec (tokens, app shell, IA, every screen, interactions, state). Same authoritative spec as the v1 handoff, kept alongside these assets so the bundle is self-describing. |

## How the two shells use these glyphs

- **Avalonia interim head (today):** [`apps/desktop/CorpusStudio.Avalonia/Styles/Icons.axaml`](../../../apps/desktop/CorpusStudio.Avalonia/Styles/Icons.axaml)
  **inlines these exact glyphs** as `StreamGeometry` resources — the resource path data is the
  byte-for-byte Phosphor path from the matching SVG here (e.g. the `IcoDashboard` geometry is
  `assets/icons/regular/gauge.svg`'s path). The `.axaml` resource keys are short app-local names
  (`Ico…`), not the Phosphor names; use `ICONS.md`/`icons.json` to map a glyph back to its Phosphor
  identity. To add a glyph, drop its SVG here and add a matching `StreamGeometry`.
- **Future React / Tauri 2 shell:** do **not** hand-copy these SVGs — run `npm i @phosphor-icons/react`
  and import each icon by the component name listed in `ICONS.md` (e.g. `Gauge`, `PencilSimpleLine`,
  `WarningCircle`), setting `weight="regular|fill|bold"` and letting `currentColor` pick up the
  semantic token. The loose SVGs here are the reference / inline-or-sprite fallback.

## Deliberately excluded (kept out of the repo)

- **`Corpus Studio.dc.html`** and **`support.js`** — the standalone HTML reference and its proprietary
  templating runtime. The spec is explicit that `support.js` is **not to be ported**; the interactive
  reference already lives in the v1 [`../handoff/`](../handoff/) bundle, so it is not duplicated here.
- **`_ds/…/_ds_bundle.js`** — the Nocturne component runtime bundle. Only the token **source**
  (`styles.css`) is durable; the JS runtime is reference-only and intentionally omitted.

## License

Phosphor Icons are **MIT** licensed (© Phosphor Icons). The Nocturne tokens are the project's own
design system. See [`../README.md`](../README.md) for how the tokens flow into
[`nocturne-tokens.json`](../nocturne-tokens.json) and the Avalonia `App.axaml`.
