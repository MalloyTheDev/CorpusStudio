# Dataset Version History & Lineage (v1.0)

A **dataset version** is a durable, inspectable *lineage anchor*: it records the
identity of a project's dataset at a moment in time and pins the artifacts that
co-existed with it. It answers "**which exact dataset state produced this model /
evaluation?**" and "**has the dataset changed since?**" — without copying the
dataset. It is the dataset-side mirror of the training-run and model-artifact
registries.

## What a version stores

Records live at `dataset_versions/{version_id}.json` (one atomic JSON per
version, newest-first because `version_id` is timestamp-prefixed) — beside
`training_runs/` and `model_artifacts/`.

| Field | Meaning |
|---|---|
| `version_id` | `YYYYMMDDThhmmss-<suffix>`, validated `^[A-Za-z0-9._-]+$` (injective filename). |
| `created_at` / `updated_at` | ISO-8601 timestamps. |
| `label` / `trigger` | User label; what produced it (`manual`, `manual_add`, `import_commit`, `pre_training`). |
| `row_count` | Rows in `examples.jsonl` at capture time. |
| `content_fingerprint` | 64-char SHA-256 over the **ordered** per-row exact signatures, or `null` if the dataset was missing/unreadable. |
| `fingerprint_algo` | `sha256-ordered-exact-v1` — versioned so a future identity scheme is additive, never a silent reinterpretation. |
| `row_signature_kind` | `exact` (reserved: `normalized` later). |
| `source_run_ids` / `artifact_ids` | Pinned links to training runs / model artifacts. |
| `eval_report_path` / `gate_report_path` | Pinned links to an evaluation report (absolute path) / a dataset-scope gate report. |
| `notes` | Free text. |

**Nothing derivable is stored.** Eval scores, base model, artifact integrity,
and gate status are resolved *live* when the version card is rendered, so a
record can never drift from the state it points at (the same discipline as the
weight card).

### The fingerprint

`content_fingerprint` is a streaming SHA-256 fed the canonical per-row signature
(`exact_row_signature` — `json.dumps(row, sort_keys=True, separators=(',',':'))`,
the *same* primitive used by cleaning/quality/leakage) joined by newlines. It is
deliberately **order-sensitive**: rows have no stable identity today, so a
version means "these exact rows in this exact order." Reordering, adding,
removing, or editing any row changes the fingerprint. It streams in O(1) memory
and never hashes anything but the dataset rows; a missing/unreadable dataset
yields `null` rather than a false alarm.

## Live integrity (`matches` / `drifted` / `unreadable`)

Listing and the card recompute the fingerprint of the *current* `examples.jsonl`
and compare it to the stored one:

- **matches** — the dataset is exactly the state this version recorded.
- **drifted** — the dataset has changed since; linked runs/evals may describe a
  different state (the card leads with this warning).
- **unreadable** — the dataset is missing/unreadable, or no fingerprint was
  recorded.

## CLI

```
# Capture a version (fingerprint + row count of examples.jsonl, with links)
python -m corpus_studio.cli dataset-version-create <project-dir> \
  --label "before cleaning" --trigger pre_training \
  --link-run <run_id> --link-artifact <artifact_id> \
  [--eval-report-path <abs>] [--gate-report-path <path>] [--stamp-run <run_id>]

# List versions (newest first), each annotated with current integrity
python -m corpus_studio.cli dataset-version-list <project-dir>

# Render the live version card (Markdown, or --json for the resolved card)
python -m corpus_studio.cli dataset-version-show <project-dir> --version-id <id> [--json]

# Diff two versions (added/removed/common rows; needs both captured with rows)
python -m corpus_studio.cli dataset-version-diff <project-dir> \
  --version-id <base> --other <other> [--samples 5] [--json]

# Restore a version's rows to a file (never examples.jsonl; verified by default)
python -m corpus_studio.cli dataset-version-restore <project-dir> \
  --version-id <id> --output <path> [--force] [--no-verify] [--json]

# Prune row-store rows that no version manifest references (fail-closed)
python -m corpus_studio.cli dataset-version-gc <project-dir> [--dry-run] [--json]
```

`dataset-version-create` computes the fingerprint + row count from the project's
`examples.jsonl`, auto-links the newest dataset-scope gate report already on disk
(deterministic — it never *runs* a gate), and with `--stamp-run` writes
`source_snapshot_id=<version_id>` onto that run so the dataset→run link closes in
both directions. The `--stamp-run` target is checked before anything is captured,
so a missing run refuses without leaving rows or a manifest behind.

## Row store, manifests, and diff (v1.0.2)

By default `dataset-version-create` also stores the version's **row bodies** so it
can later be diffed (and, in a future slice, restored):

- **Content-addressed store** — `dataset_versions/row_store.jsonl`, one line per
  *unique* row (`{"row_id": <sha256>, "row": <canonical>}`). `row_id` is the
  SHA-256 of the same `exact_row_signature` used for identity, so identical rows
  across versions are stored once. It is line-inspectable and grows as versions are
  captured; **row-store GC** (`dataset-version-gc`, or the Versions tab's "Clean up
  row store" action — a dry-run + confirm) prunes rows no version manifest references,
  fail-closed (an unclassifiable/unreadable case keeps the rows rather than risk data loss).
- **Ordered manifest** — a per-version sidecar `dataset_versions/<version_id>.rows`
  (one `row_id` per line, in order). The record carries `rows_stored`,
  `stored_row_count`, and `row_manifest_algo` (`sha256-exact-v1`, versioned).
- **Single-pass capture** — the fingerprint, the ordered manifest, and the store
  writes all come from **one** read of `examples.jsonl`, so they can never desync.
- **Cost is surfaced, not silent** — the first capture duplicates the dataset into
  the store; capture prints the stored/new row counts, and `--no-store-rows` opts
  out (that version then can't be diffed, `rows_stored=false`).

**`dataset-version-diff`** compares two versions' manifests as **multisets** (so
duplicate rows count) and reports added / removed / common, with sample row
bodies pulled from the store. Because identity is the *canonical* signature, a
pure reordering or a key-order/whitespace-only change is **not** a diff.

> **Canonical caveat:** the store holds the canonical row (sorted keys, compact),
> so diff — and a future restore — reconstruct the same rows *in order with keys
> normalized*, not a byte-identical file. Semantic content is preserved.

A version captured before v1.0.2 (or with `--no-store-rows`) has no manifest;
`dataset-version-diff` refuses it with a clear "recapture with row storage"
message rather than guessing.

## Restore (v1.0.3)

`dataset-version-restore` reconstructs a version's exact rows (manifest order,
from the store) and writes them as JSONL to `--output`:

- **`examples.jsonl` has one sanctioned writer — the engine.** By default restore
  targets an explicit `--output` path (and refuses to resolve to `examples.jsonl`);
  `--in-place` overwrites the dataset through the engine's single-writer
  (`examples-append` / `storage/examples_writer.py`), capturing an undo version
  first. The WPF/Avalonia desktop that historically owned the write is retiring (#545).
- **Verified by default.** The reconstruction is re-fingerprinted and must equal
  the version's recorded `content_fingerprint` — because the row signature is
  idempotent on canonical rows, a faithful restore *must* reproduce it. On a
  mismatch, restore writes nothing. `--no-verify` skips the check.
- **All-or-nothing.** If any manifest row is missing from the store, restore
  refuses (reports the count + a few sample ids) and writes nothing — never a
  partial dataset.
- **Overwrite-safe & atomic.** Refuses an existing `--output` unless `--force`;
  writes a temp file beside `--output` then `os.replace` (no half-written file).
- **Canonical caveat, made concrete.** Restored rows are in canonical form (keys
  normalized), so the file is *not* byte-identical to the original — the
  fingerprint match is the proof the rows are semantically identical.

A version without stored rows (pre-v1.0.2 or `--no-store-rows`) can't be
restored and refuses with the same "recapture with row storage" message.

## Concurrency and crash safety (#859)

The row store and the per-version manifests are **one shared structure**: a
version is restorable only while every row its manifest names is in the store, and
GC deletes the store rows no manifest names. Two protocols mutate it:

- **capture / publication**: append the dataset's new rows to the store, publish
  the manifest, commit the record (`dataset-version-create`, `import-commit`, the
  undo capture of `examples-delete` / `examples-edit` / `examples-clean`, and
  `dataset-version-restore --in-place`);
- **GC**: read every manifest (the live set), read the store, atomically replace it.

Unserialized, a version published between GC's manifest scan and its replace would
lose its rows for good. Both protocols therefore run under one cross-process
**version-store lock**, `dataset_versions/.version_store.lock`
(`engine/corpus_studio/versions/store_lock.py`):

- Publication holds it from the first store append through the manifest and the
  record; GC holds it from the manifest scan through the replace (a dry run holds
  it too, so its report is consistent). Every other store append takes it as well.
- It is portable (`fcntl.flock` on POSIX, `msvcrt.locking` on Windows), waits a
  **bounded** time (60 s by default) and then refuses with a "version store is
  busy" error (nothing written, nothing pruned, safe to retry), is same-thread
  reentrant, and is released by the kernel when its holder exits or crashes, so a
  crash never leaves a stale lock. Never delete the lock file to "unstick" it.
- The version store is **single-owner**. The lock file is created `0600` and must
  be one regular file owned by the user running the engine (a planted link or
  another user's file is refused), so a project shared between accounts, or run
  once under `sudo` or as a container's root, refuses capture and GC with "lock
  file is owned by another user". To recover, with no capture or GC running, the
  file's owner or an administrator removes `dataset_versions/.version_store.lock`
  (or `chown`s it to that user); the next operation recreates it. Removing it is
  safe only then, because a running holder keeps its lock on the removed file.
- Readers stay lock-free: `dataset-version-diff`, `restore --output`, and
  version-pinned suites read published manifests, whose rows GC never prunes, and
  GC swaps the store with an atomic `os.replace`.
- The record, the manifest, and GC's rewritten store are all written through a
  uniquely named, fsynced temp file plus `os.replace`, so a crash leaves either the
  old file or the complete new one. That temp file is created owner-only (`0600`,
  as for `examples.jsonl`), so records, manifests, and a store GC has rewritten are
  readable only by their owner, consistent with the single-owner lock.

**Lock order.** The `examples.jsonl` single-writer lock (non-blocking) is always
taken **before** the version-store lock, never while holding it: `import-commit`
takes the writer lock, then the version-store lock (before appending, so a busy
store refuses the whole commit rather than committing rows without their undo
version), and the examples-mutation and in-place restore undo captures take the
version-store lock inside the writer lock. Capture, publication, and GC never
request the writer lock, so the order is acyclic and nesting cannot deadlock. A
busy version store makes those commands refuse with "nothing committed" /
"nothing changed".

**Interrupted states and how they recover** (no manual repair needed):

| Interrupted at | Left on disk | Recovery |
|---|---|---|
| Mid-append | Complete orphan rows, possibly a torn last line (the cut can fall inside a multi-byte UTF-8 character) | The next capture newline-terminates the torn line before appending, at the byte level, so it cannot swallow a row. Every store reader decodes line by line and skips an undecodable or torn line, so capture, diff, and restore keep working. GC prunes the orphan rows and keeps the unclassifiable fragment. |
| After the append, before the manifest | Orphan rows (and possibly a stray temp file nothing reads) | GC prunes the orphans. It can do so safely only because it holds the lock, which proves no publication is in flight. |
| After the manifest, before the record | A manifest with no record | GC keeps its rows (it never prunes on a guess); the version is invisible to list/restore. |
| During GC | The original store, intact (atomic replace) | Re-run GC. |
| Failed capture (unreadable dataset, store I/O error) | Nothing new | The store is rolled back to its pre-capture size, and only ever shrunk, never extended. |

**GC refuses an incomplete reference scan** (exit 1, nothing pruned) when a manifest
cannot be read, is not valid UTF-8, has a line that is not a sha256 row id (torn or
corrupt), or lists a different number of rows than its record's
`stored_row_count`; it also refuses when the store itself is not valid UTF-8, when
the store cannot be replaced (e.g. held open on Windows), or when the version store
stays busy. A manifest whose record is missing or unreadable stays live as read, so
it can only keep rows.

## Hard boundaries

The engine only **reads** `examples.jsonl` and **writes** only under
`dataset_versions/` (records, manifests, the row store, and the version-store lock
file). It never moves, copies, or deletes the dataset or any weight file; it runs
no ML and makes no network calls. Capture is explicit/opt-in, taken when the
dataset is quiescent (after an import/append commits), never as a side effect.

## Implemented vs deferred

**Implemented (v1.0.0, engine):** the version record + registry
(`versions/version_registry.py`), the streaming fingerprint, live integrity, the
live version card (`versions/version_card.py`), the `source_snapshot_id`
run back-link, and the three CLI commands.

**Implemented (v1.0.1, desktop):** a **Versions** tab surfacing the history —
a read-only list with a live integrity badge (✅ matches / ⚠ drifted /
⛔ unreadable), a one-line summary, a **Capture version** button (opt-in, with an
optional label), and **View card** (the rendered version card). Both capture and
list go **through the engine** (`dataset-version-create` / `-list`), so the
desktop never recomputes the fingerprint and integrity is verified, not guessed.

**Implemented (v1.0.2, engine):** stable per-row identity (`row_id`), a
content-addressed deduped row store (`versions/row_store.py`), a per-version
ordered manifest captured single-pass with the fingerprint
(`versions/version_registry.py`), and a read-only `dataset-version-diff`
(`versions/version_diff.py`, multiset added/removed/common + sample rows).

**Implemented (v1.0.3, engine):** `dataset-version-restore`
(`versions/version_restore.py`) — reconstruct a version's rows from the store to
an `--output` file, verified against the recorded fingerprint (all-or-nothing,
overwrite-safe, atomic), or in place via `--in-place` (below).

**Implemented (engine in-place restore, `--in-place`):** `dataset-version-restore
--in-place` captures the current dataset as an **undo** version first (and refuses
if that undo isn't a genuine recovery point — an unreadable dataset, or the row
store couldn't be written), verifies the reconstruction, then atomically swaps it
in through the single-writer lock (snapshot → verify → swap is one locked critical
section). Any failure before the swap leaves the dataset untouched. The undo version
stays in the history, so restoring it reverts.

**Implemented (desktop diff view):** the Versions tab can compare two versions —
"Set diff base" pins the selected version, then "Diff base → selected" renders the
engine's `dataset-version-diff` (added/removed/common + sample rows) in the detail
pane. Read-only; the engine owns the diff (it refuses a version without stored rows).

**Reorder detection** (#196): when two versions hold the *same* rows in a *different*
order (nothing added or removed), the diff flags it explicitly — `reordered` +
`moved_count`, and a "⚠ Reordered" note in the diff card — rather than showing a
misleading "no changes". A sequence-sensitive trainer would see a different dataset.

**Deferred:**
- Auto-capture after an import/append commit; restore straight to stdout.
- A `normalized` row identity (whitespace/key-order-insensitive) as an opt-in mode.
