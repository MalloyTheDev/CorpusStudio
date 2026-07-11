# Handoff: Corpus Studio — Production UI

## Overview
Corpus Studio is a **local-first desktop app for creating fine-tuning datasets**: author → validate → clean/gate → split → evaluate → train → version, entirely on the user's machine. This handoff is a **production redesign of the whole app shell and every surface**, replacing the previous 15-tab dashboard with a grouped, workflow-phase information architecture.

Target stack (per the team): **Rust + Python engine with a React front-end, packaged with Tauri 2.** The design is framed as a desktop window.

## About the design files
The files in this bundle are **design references authored in HTML** (a single "Design Component" prototype, `Corpus Studio.dc.html`). They demonstrate the intended look, layout, IA, and interaction — **they are not production code to copy**. The task is to **recreate these designs in the React/Tauri app** using the codebase's own component patterns, router, and state libraries. The `.dc.html` file uses a small proprietary templating runtime (`support.js`) only so it can render standalone; **ignore that runtime** and reimplement the UI as normal React components. All visual values below are the source of truth.

## Fidelity
**High-fidelity.** Final colors, typography, spacing, and interaction behavior. Recreate pixel-accurately using the exact tokens in the **Design Tokens** section. Both a **dark (primary)** and a **light** theme are specified; implement theming as CSS-variable swaps.

## Opening the reference
Open `Corpus Studio.dc.html` in a browser (it pulls Phosphor icons from a CDN; the Nocturne stylesheet + bundle are included under `_ds/`). It opens at min-width 1180px — view it in a wide window. Navigate via the activity bar (left), the grouped sidebar, the docked console tabs, and Settings → Appearance (theme).

---

## Design system: Nocturne
The visual system is **Nocturne** — a quiet, compact dark UI: near-neutral blue-grey grounds, **Inter** throughout, 8px radii, a single blurple accent used as line/glow (not floods), soft elevation (hairline edge + ambient shadow). The full token/source tree is included at `_ds/nocturne-557b8352-4f53-4f98-9e8f-4873cfb3aa97/` (`styles.css` is the source of truth for base tokens). The app layers its own semantic + chrome tokens on top of Nocturne, listed below.

### Design tokens — app layer (CSS custom properties)
These are defined on a root `.cs-root` element and overridden on `.cs-root.cs-light`. Reimplement as a theme object / CSS vars.

**Dark (default)**
| Token | Value | Use |
|---|---|---|
| `--win` | `#12141f` | window body ground |
| `--chrome` | `#0e1019` | titlebar, activity bar, console |
| `--panel` | `#232532` | cards / surfaces (Nocturne `--color-surface`) |
| `--panel-2` | `#1b1d29` | sidebar, insets, inputs |
| `--line` | `rgba(233,233,237,.12)` | hairline dividers |
| `--line-2` | `#2c2f3e` | solid borders, input borders |
| `--text` | `#e9e9ed` | primary text |
| `--t2` | `#b2b6ca` | secondary text |
| `--t3` | `#9397ab` | tertiary/muted |
| `--t4` | `#75798c` | faint labels, disabled icons |
| `--accent` | `#968ae0` | accent (Nocturne accent-500) |
| `--accent-strong` | `#b5abfc` | accent hover / emphasis (accent-400) |
| `--accent-ink` | `#e7e5fe` | text on accent-soft (accent-200) |
| `--accent-soft` | `color-mix(in srgb, var(--accent) 16%, transparent)` | accent tint fills, active nav |
| `--hover` | `color-mix(in srgb, var(--text) 7%, transparent)` | hover tint |
| `--track` | `#33384a` | progress/bar tracks |
| `--ok` | `#6bbf9a` | pass / clean / success |
| `--warn` | `#d9a35f` | warn / moderate |
| `--bad` | `#d76d6d` | block / error / critical |
| `--grade` | `#d98f52` | dataset-debt grade badge (D-grade orange) |

**Light (`.cs-light` overrides)**
| Token | Value |
|---|---|
| `--win` | `#eef0f5` |
| `--chrome` | `#e6e8ef` |
| `--panel` | `#ffffff` |
| `--panel-2` | `#f4f6fa` |
| `--line` | `rgba(26,28,40,.10)` |
| `--line-2` | `#e0e3ec` |
| `--text` | `#1a1c28` |
| `--t2` | `#3f434f` |
| `--t3` | `#5b6070` |
| `--t4` | `#868b9c` |
| `--accent` | `#6c5fd0` |
| `--accent-strong` | `#5d5294` |
| `--accent-ink` | `#463c7d` |
| `--accent-soft` | `color-mix(in srgb, var(--accent) 12%, transparent)` |
| `--hover` | `color-mix(in srgb, #1a1c28 5%, transparent)` |
| `--track` | `#dfe3ec` |
| `--ok` | `#3f9b73` | 
| `--warn` | `#b8792e` |
| `--bad` | `#c85552` |
| `--grade` | `#c07636` |

Status colors are semantic and largely consistent across themes (tuned for contrast on each ground). On a dark ground, badge text placed on a solid `--grade`/`--ok`/`--accent` fill uses a near-black ink (`#161826` / `#1a1206`).

### Typography
- **Family:** Inter (weights 400/500/600/700), `system-ui` fallback. Load Inter (already imported by Nocturne's `styles.css`).
- **Scale (px / weight):** big metric 26/600; grade badge 28–34/700; screen title (context bar) 15/600; card title 13–13.5/600; body 12.5/400–500; nav item 13/500; small label 11–11.5; micro 10.5; group header 10/600 uppercase, letter-spacing `.12em`.
- Tabular numerals (`font-variant-numeric: tabular-nums`) on all counts/metrics.
- Monospace (`ui-monospace, Menlo`) for file paths, fingerprints, JSONL, CLI, run ids.

### Radii, elevation, spacing
- Radii: cards 11–12px, buttons/inputs 8px, chips/badges 5–6px, pills 12–14px, nav rows 7px, avatars 7–9px.
- Elevation: Nocturne `--shadow-sm` = `0 0 0 1px #3f424d` (hairline). Cards use `box-shadow: var(--shadow-sm)`. Dialogs use `--shadow-lg`. No heavy drop shadows on dark.
- Spacing is compact (Nocturne density 0.7×). Card padding 13–18px; section padding 18px 20px; gaps 8–14px.

### Icons
**Phosphor Icons** (regular / fill / bold), loaded from `unpkg.com/@phosphor-icons/web@2.1.1`. In React use `@phosphor-icons/react`. Glyphs used: nav — `gauge, pencil-simple-line, rows, tray-arrow-down, scales, broom, heartbeat, chart-pie-slice, flask, trophy, list-checks, graduation-cap, cube, git-branch, gear`; activity bar — `tree-view, squares-four(fill), magnifying-glass, warning-diamond, list-dashes, sun, moon-stars`; status — `check(bold), x(bold), warning(fill), check-circle(fill), x-circle(fill), warning-circle(fill), shield-warning(fill), shield-check, seal-check(fill), lock-simple`; misc — `caret-up-down, caret-down/right, arrow-right/left, plus, plus-circle, play, floppy-disk, arrows-clockwise, git-commit, git-diff, shuffle, thumbs-up/down(fill), copy, copy-simple, folder(fill), file-arrow-up, cloud-arrow-down, folder-open, trophy, chats, chat-text, code`.

---

## App shell (chrome — persistent on every Studio/Explorer view)

Fixed desktop window, **min-width 1180px, min-height 660px**; below that the window scrolls rather than reflowing. Vertical stack: titlebar → body → (console lives inside the Studio main column).

| Region | Size | Notes |
|---|---|---|
| **Titlebar** | height 36px, `--chrome` | macOS traffic-light dots (`#e0655b/#e0b055/#5fb87e`), "Corpus Studio — Support Chatbot", right-aligned version `v0.9.4` (mono). |
| **Activity bar** | width 56px, `--chrome`, right border `--line` | Top: brand tile "C" (34px, `--accent` bg, opens Start Center) → Explorer (`tree-view`) → Studio (`squares-four` fill) → divider → Search → Problems (`warning-diamond`, red count badge "3") → Output (`list-dashes`). Bottom: theme toggle (`sun`/`moon-stars`) → Settings (`gear`). Active item: `--accent` icon on `--accent-soft` 40×40 rounded-9 tile. Idle icon `--t4`. |
| **Primary sidebar** | width 250px, `--panel-2`, right border `--line` | Project switcher card at top (28px "SC" avatar, name + "instruction · 19 rows", `caret-up-down`). Grouped nav (see IA). Footer: Versions, Settings, then a green "Local-first · engine ready" status dot row. |
| **Context bar** | height 54px, bottom border `--line` | Left: screen title (15/600) + subtitle (11.5, `--t4`). Right: action buttons — `Import` (ghost), `Run gates` (ghost), `Author example` (accent-outline primary). |
| **Right rail (contextual)** | width 316px, `--panel-2`, left border | **Only shown on Dashboard, Writing Studio, Quality, Dataset Debt, Splits.** Quality summary: header + "Run" button, a red PII alert card, then a metric list (Examples 19, Empty rows 2⚠, Exact duplicates 2⚠, Near-duplicates 2⚠, Low-information 0✓, Possible PII 1✗), divider, Validation card (green "Passed · 0 schema errors · 3h ago"). |
| **Docked console** | height 176px, `--chrome`, top border | Tabs: **Problems** (red count 3), **Output**, **Terminal**; right: "2 block · 1 warn" + collapse caret. Active tab = `--text` text + 2px `--accent` underline. Collapsible. |

### Buttons
- **Primary (accent-outline):** text `#161826` on solid `--accent` bg (filled variant) OR `--accent-ink` text on `--accent-soft` with 1px `--accent` border (soft variant). Hover → `--accent-strong`. Radius 8px, padding ~7–8px×14px, 12.5/600.
- **Secondary/ghost:** `--t2` text, transparent bg, 1px `--line-2` border, hover bg `--hover`.
- Focus-visible ring: `2px solid var(--accent)`, offset 2px (Nocturne default).

---

## Information architecture (the core change)
The old 15-tab strip is replaced by a **grouped section→subsection sidebar** with a **per-phase status glyph** on each group header:

- **OVERVIEW** — Dashboard
- **AUTHOR**  ✓ (green check) — Writing Studio · Examples `19` · Import & Quarantine `2`(amber) · Preference Review
- **MEASURE**  ⚠ (amber) — Quality · Dataset Debt `D`(grade badge) · Splits
- **EVALUATE**  🔒 (locked, `--t4`) — Evaluation · Model Arena · Suites
- **TRAIN**  🔒 (locked) — Training · Artifacts
- **(footer)** — Versions · Settings

Group status reflects pipeline readiness (Author complete, Measure has open issues, Evaluate/Train gated until issues clear). Active nav row: `--accent-ink` text on `--accent-soft` with a 2px inset accent left-bar. Idle: `--t2`, hover `--hover`. Count/grade badges sit right-aligned in the row.

---

## Screens / views
All Studio screens render in the content-scroll area between the sidebar and the (contextual) right rail. Section root padding is `18px 20px` unless noted. Exact copy is realistic sample content drawn from the project's example dataset ("Support Chatbot", instruction schema, 19 rows, debt grade **D** caused by a leaked email).

### 1. Dashboard (hero)
- **Purpose:** project overview + train-readiness at a glance.
- **Readiness hero card** (`--panel`, radius 12): 52px `--grade` "D" badge, "Support Chatbot" + `instruction` tag + subtitle "19 examples · debt grade D · leaked personal data + 3 issues…", "Re-check debt" ghost button. Below: a 7-node **lifecycle strip** — Author✓ Validate✓ Quality⚠ Gate✗ Split— Evaluate— Train🔒 — 26px status circles joined by 2px connectors (`--ok`/`--track`).
- **Stat cards** (4, `repeat(auto-fit,minmax(150px,1fr))`): Examples 19 (+7 this week, `--ok`); Dataset debt D (`--grade`, "4 items · fix first"); Open problems 3 (`--bad`, "2 block · 1 warn"); Last evaluation — ("not run yet").
- **Quick actions** (6, `auto-fit minmax(210px,1fr)`): Author examples, Run quality, Generate splits, Run evaluation, Training config, Capture version — each an icon tile (`--accent-soft` 32px) + title + caption; navigates to that section on click.
- **Recent activity** list (4 rows with icons, description, relative time).

### 2. Writing Studio
- **Purpose:** author + validate one example. Max-width 820px.
- Top row: `instruction schema` tag, "Draft · not saved", `Validate` + `Save example` buttons.
- Fields: **Instruction** (textarea), **Input** (optional, textarea), **Output** (textarea, min 120px) — all `--panel` bg, 1px `--line-2`, radius 9. Sample content = password-reset example.
- Metadata row (`auto-fit`): Tags (chips + "add"), Source ("handwritten"), License ("CC-BY-4.0").
- **Validation result** card (`--ok-soft`): "Draft is valid — required fields present · output 34 tokens · no duplicates · no PII."

### 3. Examples
- **Purpose:** browse/inspect rows. Full height, list + detail.
- Toolbar: search field ("Search 19 examples…"), filter segmented (All 19 / Clean 15 / Flagged 4).
- **List** (290px, `--panel`): rows with index, truncated instruction, status icon — clean (`--ok` check-circle), near-duplicate (`--warn` copy-simple), empty (`--warn` warning-circle), PII (`--bad` x-circle). First row selected (`--accent-soft`).
- **Detail** (`--panel`): "Example 1" + CLEAN chip + edit/trash icons; INSTRUCTION + OUTPUT blocks; meta row (Tags, Source, Tokens, Added).

### 4. Import & Quarantine
- **Purpose:** bring rows in + repair rejects. Max-width 880px.
- **Import card:** source segmented (JSONL/CSV · Hugging Face · Folder), dashed file drop ("support_tickets.csv · 312 rows detected"), **field mapping** (`subject→instruction`, `context→input`, `reply→output` with accent chips), "Import & validate" primary.
- **Quarantine:** header + "2 rejected" chip + "Last import: 12 accepted · 2 quarantined". Two rejected-row cards (missing `output`; invalid JSON) each with a "Repair" button.

### 5. Preference Review
- **Purpose:** rank chosen vs rejected. Full height, pairs list + detail.
- Header: "14 pairs · 9 ranked · 5 pending", Contrast filter, "Export ranking".
- **List** (240px): pairs with contrast/ranked labels.
- **Detail:** PROMPT card; two response cards side-by-side — **CHOSEN** (1px `--ok` border, thumbs-up) vs **REJECTED** (`--line-2`, thumbs-down); REASON card.

### 6. Quality
- **Purpose:** full quality report (rail shown).
- Header: "15 of 19 rows clean · 4 flagged", last-run time, "Run quality" primary.
- **Signal cards** (`auto-fit minmax(232px,1fr)`): Empty rows 2 (HIGH, 10.5%, rows 12/18), Exact duplicates 2 (HIGH, rows 6/14), Near-duplicates 2 (MOD), Possible PII 1 (HIGH, email, row 15), Low-information 0 (✓), Synthetic patterns 0 (✓). Each: icon, title, severity chip (color-coded), big count + rate, affected/repair caption.
- Synthetic-issue triage card (green, "no issues · severities advisory").

### 7. Dataset Debt
- **Purpose:** graded, ranked remediation ledger (rail shown).
- **Grade hero:** 64px `--grade` "D" badge, "Not train-ready — fix high-severity items first", explanation (debt normalizes counts→rates, ranks, recommends paydown; adds no new detection), a small 5-bar trend sparkline.
- **Ledger** (highest severity first): Personal data — 1 email (HIGH, presence-graded, row 15); Exact duplicates — 2 (HIGH, 10.5%); Empty rows — 2 (HIGH, 10.5%); Near-duplicates — 2 (MOD). Each card: severity chip, title, remediation text, right-aligned rate/row.
- Grade rule: F if any critical; else D if any high; else C moderate; else B low; else A. Secrets/PII graded by presence, not rate.

### 8. Splits
- **Purpose:** leakage-checked train/val/test (rail shown). Max-width 760px.
- **Config card:** Train% 80 / Validation% 10 / Test% 10 / Seed 42, "Generate splits" primary.
- **Result card:** "No leakage" green chip; stacked proportion bar (79% train `--accent` / 10.5% val `--ok` / 10.5% test `--warn`); counts 15/2/2 + "0 shared rows".

### 9. Evaluation (hero)
- **Purpose:** score the dataset against a model.
- **Config card:** Backend (ollama, green dot), Model (llama3.1:8b), Judge model (optional, "keyword-overlap (none)"), Threshold 70, "Run evaluation" primary.
- **Summary cards:** Average score 78.4; Pass rate 74% (`--ok`, "14/19 ≥ 70"); Evaluated 19; Metric "keyword overlap".
- **Per-example results:** filter chips (All / Pass 14 / Fail 5) + a list of rows with index, prompt, score bar (`--ok`/`--warn`), score, PASS/FAIL chip.

### 10. Model Arena
- **Purpose:** head-to-head model comparison.
- Header: "14 prompts · 3 models · pairwise judged", "Run arena" primary.
- **Standings** (3 ranked cards): llama3.1:8b 62% (8W·3L·3T, trophy), qwen2.5:7b 57%, mistral:7b 43% — each a win-rate bar.
- **Head-to-head** table (llama3.1 vs qwen2.5): per-prompt winner chips ("llama3.1 wins" / "tie" / "qwen2.5 wins").

### 11. Suites
- **Purpose:** saved evaluation suites. Max-width 840px.
- Header + "New suite". Three suite cards: **Support smoke** (instruction · llama3.1:8b · keyword-overlap · runs on save · last 78.4), **Refusal & safety** (chat suite · llm_judge · manual · last 91), **Regression guard** (compares to v2 · blocks on −2.0 drop). Each has a "Run" button.

### 12. Training
- **Purpose:** configure, launch & track runs. Two-column (`auto-fit minmax(300px,1fr)`).
- **Config card:** Method segmented (QLoRA/LoRA/Full), Base model (llama-3.1-8b), Epochs 3, Learning rate 2e-4, Batch size 8, LoRA rank 16, Exporter axolotl, "Launch training run" primary.
- **Run card:** run-0007 "complete" chip, final loss 0.412 / steps 720 / wall 14m, a loss-curve SVG polyline, and a **regression-gate** warning card ("unverified linkage — after-eval targeted the base model").

### 13. Artifacts
- **Purpose:** trained weights + provenance. Max-width 860px.
- Note: weights referenced by path, never modified; "Keep" runs the promote gate first.
- Two artifact cards: **run-0007** (integrity: present ✓, base llama-3.1-8b, 168MB adapter, eval 81.2; Keep / Weight card / Compare buttons) and **run-0005** (dimmed, integrity: modified ⚠, "Promote gate blocks — weights changed since eval").

### 14. Versions
- **Purpose:** snapshots + restore. Max-width 820px. Vertical timeline (2px rail, dot per node).
- v3 (CURRENT, accent border): "Imported support_tickets.csv (+12 rows)", 19 rows, fp a3f9c1, grade D. v2: "Rewrote 4 outputs for tone", 7 rows, fp 71b0e8, grade B, Restore. v1: "Initial 7 authored examples", fp 0c4d22, grade A, Restore.

### 15. Settings
- **Purpose:** project & engine config. Max-width 720px. Cards:
  - **Appearance:** Theme segmented **Dark / Light** (wired — this toggles the theme).
  - **Engine:** Python 3.11.7 (ready), Ollama localhost:11434 (detected) — green status dots.
  - **Provider policy:** OpenAI + Anthropic = "evaluator-only" (grey); Ollama = "generate ✓" (green). Local-first: cloud providers can never generate training rows.
  - **Gate thresholds:** Block on exact duplicates (on), Block on PII/secrets (on), Max regression score drop 2.0.

### Explorer (Files activity)
VS-Code-style file view: 262px tree sidebar (workspace root "support-chatbot" → data/examples.jsonl selected, schema.json, splits/, eval_reports/, gate_reports/, DATASET_CARD.md, project.json) + editor pane with a document tab, an amber "single source of rows" caution banner, and a JSONL viewer (line numbers; the PII row 15 highlighted `--bad`).

### Start Center (full-window, opened via brand tile)
Radial accent-tinted ground. Brand "C" + "Corpus Studio / Local-first dataset creation studio" + intro. Two columns: **START** (New Dataset Project [accent], Import Dataset, Open Project…, Import from Hugging Face) and **RECENT** (Support Chatbot D, Code Assistant B, Chat Tutor A — avatar + schema/rows/opened + grade badge) + a green "Engine ready · Python 3.11 · Ollama detected" strip.

### New Project wizard (modal over Start Center / shell)
660px dialog, `--shadow-lg`. Header + close. **4-step stepper** (Name · Schema · Source · Review) with numbered dots + connectors (done/active/idle states). Steps: (1) name + location; (2) schema picker grid — Instruction (selected), Chat, Preference, Code; (3) starting point — Start empty (selected), Import a file, From Hugging Face; (4) review summary. Footer: Cancel / Back / Continue → "Create project".

---

## Interactions & behavior
- **Activity bar:** brand → Start Center; Explorer → Files view; Studio → last Studio section; Problems/Output → open console to that tab; theme toggle → flip dark/light; Settings → Settings section.
- **Sidebar nav:** selects a section; sets active styling; contextual right rail appears only on Dashboard/Writing/Quality/Debt/Splits.
- **Context bar buttons & Dashboard quick-actions & rail "Run":** navigate to the relevant section (Author example→Writing, Run quality→Quality, etc.).
- **Console:** tab switch (Problems/Output/Terminal); collapse caret hides the panel.
- **Theme:** dark↔light swaps the CSS-variable set on the root; persists per session.
- **Wizard:** Continue/Back move steps 1–4 (clamped); Back disabled on step 1; "Create project" closes and lands on Dashboard.
- **Hover states** on every interactive element (nav rows, buttons, cards, list rows) via the `--hover`/`--accent-strong` tints.
- Transitions are subtle (background/border/color ~120ms). No large motion.

## State management
The prototype uses one shell state object — reimplement with your store/router:
- `screen`: `'app' | 'start'` (Start Center is full-window)
- `activity`: `'studio' | 'files'`
- `view`: active Studio section id (`dashboard, writing, examples, import, preference, quality, debt, splits, evaluation, arena, suites, training, artifacts, versions, settings`)
- `theme`: `'dark' | 'light'`
- `consoleOpen`: boolean; `consoleTab`: `'problems' | 'output' | 'terminal'`
- `wizardOpen`: boolean; `wizardStep`: 1–4

Derived: `showRail = activity==='studio' && view ∈ {dashboard,writing,quality,debt,splits}`; context title/subtitle per `view`; nav active = `view===id`. In production, most panel content is **live data from the Rust/Python engine** (quality report, debt ledger, gate reports, eval reports, training runs, versions) — the sample numbers here mirror the engine's real outputs (e.g. `build_debt_report`, `build_basic_quality_report`, `GateReport`).

## Assets
- **Icons:** Phosphor (`@phosphor-icons/react` in a React app). No custom SVG assets except the small inline loss-curve polyline (Training) and the debt trend bars (plain divs).
- **No raster images / logos** — the brand mark is a CSS tile ("C"). No photography.
- **Fonts:** Inter (Google Fonts), already imported by Nocturne's `styles.css`.

## Files in this bundle
- `Corpus Studio.dc.html` — the full interactive design reference (all screens, both themes).
- `support.js` — the prototype's rendering runtime (**reference only; do not port**).
- `_ds/nocturne-557b8352-4f53-4f98-9e8f-4873cfb3aa97/styles.css` — Nocturne base tokens + component classes (source of truth for base colors/type/spacing).
- `_ds/nocturne-557b8352-4f53-4f98-9e8f-4873cfb3aa97/_ds_bundle.js` — Nocturne component bundle (reference).
