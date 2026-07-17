# Product vs research boundary

CorpusStudio is a **local-first AI dataset and model-development application** (see
[`PRODUCT_SPEC.md`](PRODUCT_SPEC.md)). The **native-Linux 7B research paper** under
[`../research/ieee-linux-training/`](../research/ieee-linux-training/) and [`paper/`](paper/) is a
**separate project that uses** CorpusStudio to verify the training engine can train a 7B model at
sequence length 4096 on this host. That paper's machinery must never become mandatory product behavior.

This doc is the canonical statement of where the line sits. When another doc, a plan, or a chat frames
CorpusStudio itself as a "research platform," it is wrong - correct it against this file.

## The two modes

| | **Standard product mode** (default, every user) | **Sealed research mode** (opt-in, paper only) |
|---|---|---|
| Who uses it | Normal CorpusStudio users | The native-Linux 7B paper experiments |
| Environments | `backend-corpus-studio`, `backend-unsloth`, capability/control-plane recipes | `...readiness-*`, `...research-math/flash-v*` (`requires_worker_wheel=true`) |
| Worker install | Verified worker package, ordinary reproducibility (target: pinned + hash-verified) | Embedded canonical `BUILD_PROVENANCE.json`, reviewed `required_git_ancestor` floor, `source_commit` match |
| Reproducibility | Ordinary: pinned deps, environment lock, capability probe | Immutable environment lock + experiment lineage + evidence sealing |
| Identity | Ordinary run IDs and run-scoped output dirs | Reserved experiment identities, append-only amendment chain |
| Telemetry | Progress + metrics; paper-completeness flags simply read `false` | Paper telemetry completeness required |
| Lives in | `engine/corpus_studio/**`, `docs/**` | `research/ieee-linux-training/**`, `docs/paper/**` |

## Product mode must never require

- research amendments or effective experiment matrices;
- reserved experimental identities;
- paper-performance / paper-telemetry completeness;
- IEEE experiment-matrix membership;
- paper-specific lineage (per-lineage git-ancestor floors, sealed source-commit matching);
- scientific promotion / matched-trial rules.

A normal user should be able to build a dataset, pick a model and tokenizer, fine-tune locally,
evaluate, and export - without ever encountering an amendment, a paper cell, or a reserved research
identity.

## Sealed research mode may require

- embedded canonical build provenance and exact worker wheel hashes;
- immutable environment locks and reserved experiment identities;
- prospective, append-only amendments;
- paper telemetry, matched-trial requirements, and evidence sealing.

Isolate this workflow; do not weaken it. Everything above stays reachable **only** when a recipe or
execution explicitly declares sealed-research operation.

## How the boundary holds today

The separation already exists in code, mostly as an implicit toggle rather than a named mode:

- **The sealed-provenance admission gate is conditional.** `EnvironmentManager` runs the build-provenance
  admission (reviewed floor + embedded `BUILD_PROVENANCE.json` + optional `source_commit` match) only
  when `recipe.requires_worker_wheel` is true. The standard product training backend
  `backend-corpus-studio` has `requires_worker_wheel=false`, so it never reaches that gate.
- **Paper telemetry is descriptive, not a gate.** `scientific_resource_complete` /
  `scientific_throughput_complete` are booleans on the run summary; a missing-paper-field note is
  appended to a human-readable report. Nothing blocks or fails a normal run.
- **Reserved identities, amendments, and experiment matrices live in `research/ieee-linux-training/`**
  (`validate_protocol.py`). The engine package does not import them; the normal `platform-plan` ->
  `platform-run` path does not enforce them. The only engine touch-point is the `required_git_ancestor`
  floor plumbing, itself gated on `requires_worker_wheel`.

## Known gaps / planned work (not yet implemented)

These make the boundary explicit and give the product a proper middle tier. They are design intent, not
current behavior:

1. **Name the mode.** Introduce an explicit assurance mode (`standard` vs `sealed_research`) on the
   recipe/plan instead of overloading `requires_worker_wheel` as the de-facto toggle.
2. **Standard verified-worker tier.** A product training recipe that installs a **pinned, hash-verified**
   worker package - reproducible enough for ordinary use - with no reviewed git floor, embedded research
   provenance, amendment, or reserved identity. Today `backend-corpus-studio` uses loose version ranges;
   this fills the gap between "loose" and "ultra-sealed research."
3. **De-research the shared vocabulary.** Rename research-flavored names on the general Environment
   Manager (e.g. `validate_wheel_provenance_for_scientific_admission` ->
   `..._for_sealed_admission`); keep the concepts in `research/`.
4. **Plugin/skill overlay split.** A product-first skill plus an optional research-overlay skill scoped
   to `research/ieee-linux-training/` and `docs/paper/` that loads only for paper work.

## Rule of thumb

If a requirement only exists because of the paper, it belongs behind sealed research mode. If a normal
user building a dataset and training a model locally would hit it, it must not depend on anything in
`research/ieee-linux-training/`.
