# Workspace System

Corpus Studio is evolving from a fixed set of dataset tabs into an **IDE-like
workspace** for datasets — think VS Code / Cursor, but for dataset creation,
validation, review, export, evaluation, and training orchestration.

The guiding rule: **one universal workspace, many schemas, many file viewers** —
*not* one hardcoded UI per dataset type.

- The **file kind** controls the viewer/editor (text, JSON, image, …).
- The **schema kind** controls dataset-row behavior.
- The **engine** validates, measures, splits, gates, exports, evaluates, and
  prepares training.
- The **desktop** owns safe, user-facing workspace file editing and never
  silently mutates dataset content.

Corpus Studio orchestrates the user's installed trainer; it never becomes a deep
learning framework.

## Project manifest — `.corpus/project.json`

A workspace is identified by a manifest at `.corpus/project.json`. It is the
primary way to open a workspace, and it *points at* the authoritative dataset
files (e.g. `examples_file`) without replacing them — the dataset content under
the workspace root stays the source of truth, and `examples.jsonl` keeps its
single writer (the desktop).

Fields: `format` (`"corpus_studio_project"`), `format_version`, `project_id`,
`name`, `schema_id`, `template_id`, `created_at`, `last_opened_at`,
`examples_file` (default `examples.jsonl`), `asset_root` (default `assets`),
`notes`. Reading is tolerant (a missing/malformed manifest is a clear error, not a
crash) and forward-compatible (a newer `format_version` still opens).

## Recent Workspaces

A user-local registry (stored outside the repo under `%LOCALAPPDATA%/CorpusStudio/`)
tracks recently opened workspaces: `name`, `path`, `schema_id`, `last_opened_at`,
`is_pinned`, and a live-computed `missing` state. A missing/corrupt registry
recovers to an empty list without crashing startup, and a workspace whose folder
has moved is **kept and flagged** (not silently dropped) so the user can re-open,
re-pin, or remove it. Pinned entries are always retained; unpinned entries fill
the remaining slots most-recent-first (capped at 50).

## Project templates

The New Dataset Project flow chooses a **schema** (row behavior) *and* a
**template** (folder scaffold) independently:

- **Empty Workspace** — just `.corpus/project.json`.
- **Minimal Dataset Project** — `examples.jsonl`, `README.md`, `assets/`.
- **Standard Dataset Project** — adds `.corpus/workspace.json`, `dataset_card.json`,
  `imports/quarantine/`, `splits/`, `reports/`, `exports/`, `training_configs/`.
- **Full Dataset-to-Model Project** — adds per-kind asset folders, per-kind report
  folders, and `evaluation_reports/`, `arena_reports/`, `training_runs/`,
  `model_artifacts/`, `dataset_versions/`.
- **Schema-Specific Starter** — Standard structure plus schema-appropriate asset
  folders and **one valid starter row** where safe. `image_caption` is the
  exception: `examples.jsonl` is left **empty** (placeholder image paths would fail
  validation) and starter guidance is written to `README.md` instead.

`ProjectTemplateService.BuildPlan` is pure, so the wizard renders the exact
folder/file structure as a live preview *before* anything is written.

## Open / Initialize Folder

Opening an existing folder detects one of four cases and picks the safe action:

1. **Manifest present** (`.corpus/project.json`) → open directly, no changes.
2. **Dataset files, no manifest** (e.g. `examples.jsonl`) → offer to *initialize*
   (adds only `.corpus/`; existing files are untouched).
3. **Empty folder** → offer to create a project there (explicit confirmation).
4. **Unrecognized / non-empty** → inspect only; never mutate.

## File kinds

File classification is pure and deterministic (extension → kind), so the explorer
and document viewers always agree:

`Folder`, `Jsonl`, `Json`, `Markdown`, `Text`, `Yaml`, `Toml`, `Code`
(`.py/.cs/.cpp/.c/.h/.hpp/.js/.ts/.rs/.java/.go`), `Image`
(`.png/.jpg/.jpeg/.webp/.gif`), `AudioFuture`, `VideoFuture`, `Binary`, `Unknown`.

## Universal Workspace Explorer

`WorkspaceExplorerService.BuildTree` produces a deterministic tree (directories
first, then files, each alphabetical / case-insensitive), rooted at the workspace,
that:

- stays within the workspace root (never walks outside it),
- skips VCS / build / OS junk (`.git`, `node_modules`, `bin`, `obj`, `__pycache__`,
  …) but keeps the workspace's own `.corpus/`,
- does not follow symlinks/junctions and guards against directory loops,
- flags generated-artifact nodes and dataset-core files for the UI.

`CreateFile` / `CreateFolder` resolve through the path-safety layer, refuse
traversal/absolute paths, and refuse to overwrite an existing entry. There is **no
delete** in this system (a later slice may add move-to-trash with confirmation).

## File viewers / editor tabs

`WorkspaceDocumentService.Open` selects a viewer by file kind and returns an
`OpenWorkspaceDocument`:

- **Text-editable** kinds load their content into an editor with explicit **Save**
  (atomic temp+move) and dirty tracking.
- **Generated artifacts** (`reports/`, `training_runs/`, `model_artifacts/`,
  `dataset_versions/`, `evaluation_reports/`, `arena_reports/`) open **read-only**.
- **Over-large** files (> 2 MB) open as a **read-only preview** (a virtualized
  editor is a later slice).
- **Images** open a preview + metadata (opening never creates a dataset row).
- **Binary / unknown** files show a metadata-only panel.
- **`examples.jsonl`** opens as editable text but is flagged as the single-writer
  core file; it is never auto-formatted and only ever written by an explicit Save.

## Safety rules

Every workspace file operation resolves through a path-safety layer that:

- normalizes to a full path and enforces the **workspace-root boundary** (with a
  trailing-separator guard so `.../ws` is never treated as a parent of
  `.../ws-other`),
- **rejects path traversal** (`..` escapes) and absolute/rooted child paths,
- sanitizes new file/folder names (invalid characters and separators → `_`,
  reserved/degenerate names refused).

Additional hard rules the system upholds: never silently modify `examples.jsonl`;
open generated reports read-only by default; no permanent delete; never mutate a
folder without explicit confirmation.

## Implemented

**Slice 1 — foundation (shipped).** `WorkspaceProjectManifest`,
`RecentWorkspaceRecord`, `WorkspaceFileKind` (+ classifier);
`WorkspaceManifestService`, `RecentWorkspaceService`, `WorkspacePathSafety`.

**Slices 3-5 — scaffolding, explorer, documents (this change).** Pure, unit-tested
services and models — **no change to the existing project flow**:

- `Models/WorkspaceTemplateDefinition`, `Models/WorkspaceTreeNode`,
  `Models/OpenWorkspaceDocument`, `Models/WorkspaceFileMetadata`.
- `Services/WorkspaceLayout` (generated-dir / core-file / ignored-dir knowledge),
  `Services/ProjectTemplateService` (pure `BuildPlan` + guarded `Scaffold`),
  `Services/WorkspaceExplorerService` (deterministic tree + guarded create),
  `Services/WorkspaceDocumentService` (safe open + explicit atomic save + metadata).
- Tests in `CorpusStudio.Desktop.Tests/WorkspaceSystemTests.cs`.

The **Start Center, New Project wizard, Open/Initialize flow, VS Code-style
Explorer view, and editor/viewer tabs** are fully specified by an interactive
design prototype (`Corpus Studio Workspace.dc.html`) and are wired to these
services in the view layer as the next step.

## Roadmap (documented, not yet built)

- **Problems panel** (schema errors, invalid rows, missing/unreferenced assets,
  split leakage, PII/secrets, stale reports) and **Output / Logs panel** (engine,
  validation, export, evaluation, AI Assist, training logs).
- **Command Palette**, **Quick Open** (Ctrl+P), workspace **search**.
- **Dataset indexing** (row counts, line offsets, fingerprints, asset references),
  **schema-aware `examples.jsonl` row grid**, large-file **virtualized viewer**.
- **Import wizard** upgrades, **asset integrity scanner** (missing / unreferenced
  assets), audio/video preview, object-detection boxes, segmentation masks,
  COCO/YOLO export, safe rename/move reference updater.
- **Layout persistence** (`.corpus/workspace.json`: open files, expanded folders,
  panel sizes) and a per-project **Project Health** dashboard card.
