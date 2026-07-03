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

## File kinds

File classification is pure and deterministic (extension → kind), so the explorer
and document viewers always agree:

`Folder`, `Jsonl`, `Json`, `Markdown`, `Text`, `Yaml`, `Toml`, `Code`
(`.py/.cs/.cpp/.c/.h/.hpp/.js/.ts/.rs/.java/.go`), `Image`
(`.png/.jpg/.jpeg/.webp/.gif`), `AudioFuture`, `VideoFuture`, `Binary`, `Unknown`.

## Safety rules

Every workspace file operation resolves through a path-safety layer that:

- normalizes to a full path and enforces the **workspace-root boundary** (with a
  trailing-separator guard so `.../ws` is never treated as a parent of
  `.../ws-other`),
- **rejects path traversal** (`..` escapes) and absolute/rooted child paths,
- sanitizes new file/folder names (invalid characters and separators → `_`,
  reserved/degenerate names refused).

Additional hard rules the system upholds: never silently modify `examples.jsonl`;
open generated reports read-only by default; no permanent delete (a later slice
may add move-to-trash with explicit confirmation); never mutate a folder without
explicit confirmation.

## Implemented today (slice 1 — foundation)

Pure, unit-tested foundation services and models — **no UI yet, and no change to
the existing project flow**:

- `Models/WorkspaceProjectManifest`, `Models/RecentWorkspaceRecord`,
  `Models/WorkspaceFileKind` (+ classifier).
- `Services/WorkspaceManifestService` (tolerant read, atomic write),
  `Services/RecentWorkspaceService` (app-data registry, corrupt/missing recovery,
  add/pin/remove/cap, missing-path detection),
  `Services/WorkspacePathSafety` (root-boundary, traversal rejection, name
  sanitization).

## Roadmap (documented, not yet built)

- **Start Center** shown when no workspace is active (New Project / Open Folder /
  Recent Workspaces).
- **Template-driven New Project** (Empty / Minimal / Standard / Full /
  Schema-Specific) with a preview of the folder structure it will create.
- **Open / Initialize Folder** flows (manifest present → open; dataset files but
  no manifest → offer to initialize; empty → offer to create; random → inspect
  only, never mutate).
- **Universal Workspace Explorer** (deterministic tree, New File / New Folder,
  refresh, reveal-in-explorer, copy relative path).
- **Editor / viewer tabs** (text edit + explicit save, read-only report viewer,
  image preview, binary/unknown metadata panel, dirty-document tracking).
- **Problems panel**, **Output / Logs panel**, **Command Palette**, **Quick
  Open**, workspace search.
- **Dataset indexing**, **schema-aware `examples.jsonl` row grid**, large-file
  virtualized viewer.
- **Import wizard** upgrades, **asset integrity scanner** (missing / unreferenced
  assets), audio/video preview, object-detection boxes, segmentation masks,
  COCO/YOLO export, safe rename/move reference updater.
