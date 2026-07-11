# Icons & assets used

All UI icons are **Phosphor Icons v2.1.1** (MIT license). This bundle includes the **exact 59 icons** (71 name+weight variants) used across the design, as raw SVGs under `assets/icons/<weight>/`.

## Using them in the React / Tauri app (recommended)

Install the React package and import by name — you do **not** need these loose SVGs unless you prefer them:

```bash
npm i @phosphor-icons/react
```
```tsx
import { Gauge, Broom, WarningCircle } from '@phosphor-icons/react';

<Gauge weight="fill" size={16} />        {/* ph-fill ph-gauge */}
<Broom size={16} />                       {/* ph ph-broom (regular) */}
<WarningCircle weight="fill" color="var(--warn)" />
```

Weight maps directly: `ph ph-x` → `weight="regular"`, `ph-fill ph-x` → `weight="fill"`, `ph-bold ph-x` → `weight="bold"`. Color follows `currentColor`, so set `color` / CSS `color` to the semantic token (`--ok`/`--warn`/`--bad`/`--accent`).

The raw SVGs in `assets/icons/` are the same glyphs (`fill="currentColor"`, 256×256 viewBox) for anyone who wants to inline or sprite them instead.

## Fonts

**Inter** (400/500/600/700) is the only typeface, loaded by Nocturne's `styles.css` from Google Fonts. For an offline desktop build, vendor it with `npm i @fontsource/inter` (or download the woff2s from Google Fonts). Monospace uses the OS stack `ui-monospace, Menlo, monospace` — no font file needed.

## No other raster/vector assets

There are **no logos, photos, or icon images** to ship: the brand mark is a CSS tile ("C"), and the only inline SVG is the Training loss-curve `<polyline>` (in the design file). The debt trend bars are plain divs.

## Full icon inventory

| Icon | Weights | React import | Used for |
|---|---|---|---|
| `arrow-left` | regular | `<ArrowLeft/>` | Wizard back |
| `arrow-right` | regular | `<ArrowRight/>` | Wizard continue / mapping arrows |
| `arrows-clockwise` | regular | `<ArrowsClockwise/>` | Re-check debt / refresh |
| `broom` | regular | `<Broom/>` | Quality nav / run quality |
| `caret-down` | regular | `<CaretDown/>` | Dropdowns / console collapse |
| `caret-right` | regular | `<CaretRight/>` | Collapsed tree folder |
| `caret-up-down` | regular | `<CaretUpDown/>` | Project switcher |
| `chart-pie-slice` | regular | `<ChartPieSlice/>` | Splits nav / quick action |
| `chat-text` | regular | `<ChatText/>` | Instruction schema (wizard) |
| `chats` | regular | `<Chats/>` | Chat schema (wizard) |
| `check` | bold, regular | `<Check/>` | Lifecycle/step complete |
| `check-circle` | regular, fill, bold | `<CheckCircle/>` | Clean/valid/pass status |
| `cloud-arrow-down` | regular | `<CloudArrowDown/>` | Import from Hugging Face |
| `code` | regular | `<Code/>` | Code schema (wizard) |
| `copy` | regular | `<Copy/>` | Exact duplicates (quality) |
| `copy-simple` | fill, regular | `<CopySimple/>` | Near-duplicate flag |
| `cube` | fill, regular | `<Cube/>` | Artifacts nav + artifact cards |
| `export` | regular | `<Export/>` | Export button |
| `file-arrow-up` | regular | `<FileArrowUp/>` | Import file drop |
| `flask` | regular | `<Flask/>` | Evaluation nav / quick action |
| `floppy-disk` | regular | `<FloppyDisk/>` | Save example |
| `folder` | fill, regular | `<Folder/>` | Explorer folders |
| `folder-open` | regular | `<FolderOpen/>` | Open Project (Start Center) |
| `gauge` | fill, regular | `<Gauge/>` | Dashboard nav + readiness |
| `gear` | regular | `<Gear/>` | Settings (activity bar + nav) |
| `git-branch` | regular | `<GitBranch/>` | Versions nav |
| `git-commit` | regular | `<GitCommit/>` | Capture version |
| `git-diff` | regular | `<GitDiff/>` | Regression guard suite |
| `graduation-cap` | regular | `<GraduationCap/>` | Training nav / quick action |
| `heartbeat` | regular | `<Heartbeat/>` | Dataset Debt nav |
| `list-checks` | regular | `<ListChecks/>` | Suites nav |
| `list-dashes` | regular | `<ListDashes/>` | Output (activity bar) |
| `lock-simple` | regular | `<LockSimple/>` | Locked pipeline phase (Evaluate/Train) |
| `magnifying-glass` | regular | `<MagnifyingGlass/>` | Search (activity bar + Examples search) |
| `minus` | regular | `<Minus/>` | Titlebar minimize |
| `moon-stars` | regular | `<MoonStars/>` | Theme toggle (to dark) |
| `pencil-simple-line` | regular | `<PencilSimpleLine/>` | Writing Studio nav / author action |
| `play` | regular | `<Play/>` | Run (evaluation / rail) |
| `plugs-connected` | regular | `<PlugsConnected/>` | Engine status (status bar) |
| `plus` | regular | `<Plus/>` | Author example / new |
| `plus-circle` | regular | `<PlusCircle/>` | New Dataset Project (Start Center) |
| `rows` | regular | `<Rows/>` | Examples nav |
| `scales` | regular | `<Scales/>` | Preference Review nav |
| `seal-check` | fill | `<SealCheck/>` | Artifact integrity: present |
| `shield-check` | regular, fill | `<ShieldCheck/>` | Run gates button + safety suite |
| `shield-warning` | fill | `<ShieldWarning/>` | PII alert (right rail) |
| `shuffle` | regular | `<Shuffle/>` | Generate splits |
| `square` | regular | `<Square/>` | Titlebar maximize |
| `squares-four` | fill | `<SquaresFour/>` | Studio (activity bar) |
| `thumbs-down` | fill | `<ThumbsDown/>` | Preference: rejected |
| `thumbs-up` | fill | `<ThumbsUp/>` | Preference: chosen |
| `tray-arrow-down` | regular | `<TrayArrowDown/>` | Import nav + Import button |
| `tree-view` | regular | `<TreeView/>` | Explorer (activity bar) |
| `trophy` | regular, fill | `<Trophy/>` | Model Arena nav + standings |
| `warning` | fill, regular | `<Warning/>` | Warn severity (quality/debt/gates) |
| `warning-circle` | fill | `<WarningCircle/>` | Empty-row / moderate flag |
| `warning-diamond` | regular | `<WarningDiamond/>` | Problems (activity bar) + open-problems stat |
| `x` | bold, regular | `<X/>` | Titlebar close, tab close, step blocked |
| `x-circle` | fill, regular | `<XCircle/>` | Error/PII/fail status |
