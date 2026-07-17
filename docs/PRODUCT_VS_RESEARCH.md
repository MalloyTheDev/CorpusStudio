# Product vs research boundary

CorpusStudio is a **local-first AI dataset and model-development application** (see
[`PRODUCT_SPEC.md`](PRODUCT_SPEC.md)). The **native-Linux 7B research paper** under
[`../research/ieee-linux-training/`](../research/ieee-linux-training/) and [`paper/`](paper/) is a
**separate project that uses** CorpusStudio to verify the training engine can train a 7B model at
sequence length 4096 on this host.

**The IEEE 7B paper must not define CorpusStudio's product identity, defaults, navigation, or ordinary
user workflow.** CorpusStudio may still contain opt-in research and interpretability tools (for example a
future Behavior Lab) - the constraint is only that the paper's machinery is never mandatory for a normal
user. This doc is the canonical statement of where the line sits.

## The three assurance tiers

The intended model is three tiers, from lightest to strictest. Each higher tier adds to the one below.

### STANDARD

- normal local product workflow;
- no research protocol.

### VERIFIED

- pinned / hash-verified worker and dependencies;
- environment lock and capability evidence;
- generic source / build provenance;
- reproducible artifact verification;
- **no** amendments, reserved identities, matrix cells, or paper promotion.

### SEALED_RESEARCH

- all VERIFIED guarantees, plus:
- reviewed `required_git_ancestor`;
- immutable study lineage;
- amendments and reserved identities;
- matrix membership, matched trials, evidence sealing, paper completeness.

| | STANDARD | VERIFIED | SEALED_RESEARCH |
|---|---|---|---|
| Who | any user, quick local work | users who want reproducibility | the 7B paper only |
| Worker + deps | as-resolved | pinned + hash-verified | pinned + hash-verified |
| Environment | created | lock + capability evidence | immutable lock, study lineage |
| Provenance | none required | generic source/build provenance | reviewed `required_git_ancestor`, sealed `source_commit` |
| Identity | ordinary run IDs | ordinary run IDs | reserved experiment identities |
| Protocol | none | none | amendments, matrix membership, matched trials |
| Artifacts | verified | reproducibly verified | reproducibly verified + evidence-sealed |
| Telemetry | progress + metrics | progress + metrics | paper completeness required |

## This is the target design, not the current implementation

Today the code exposes these boundaries **imperfectly**. There is no explicit tier selector: the
sealed-research provenance gate is toggled by **overloading `requires_worker_wheel`** (true routes an
environment through the sealed admission gate; false takes the loose standard path). The **VERIFIED tier
does not yet exist as a distinct mode** - the standard product backend `backend-corpus-studio` installs
loose version ranges rather than a pinned, hash-verified package. So the tiers above are the intended
target, not a claim that all three are fully implemented. See "Known gaps" below.

## Product (STANDARD) must never require

- research amendments or effective experiment matrices;
- reserved experimental identities;
- paper-performance / paper-telemetry completeness;
- IEEE experiment-matrix membership;
- paper-specific lineage (per-lineage git-ancestor floors, sealed source-commit matching);
- scientific promotion / matched-trial rules.

A normal user should be able to build a dataset, pick a model and tokenizer, fine-tune locally, evaluate,
and export - without ever encountering an amendment, a paper cell, or a reserved research identity.

## SEALED_RESEARCH may require

- embedded canonical build provenance and exact worker wheel hashes;
- immutable environment locks and reserved experiment identities;
- prospective, append-only amendments;
- paper telemetry, matched-trial requirements, and evidence sealing.

Isolate this workflow; do not weaken it. Everything here stays reachable **only** when a recipe or
execution explicitly declares sealed-research operation.

## How the boundary holds today

The separation already exists in code, mostly as an implicit toggle rather than a named tier:

- **The sealed-provenance admission gate is conditional.** `EnvironmentManager` runs the build-provenance
  admission (reviewed floor + embedded `BUILD_PROVENANCE.json` + optional `source_commit` match) only
  when `recipe.requires_worker_wheel` is true. The standard product training backend
  `backend-corpus-studio` has `requires_worker_wheel=false`, so it never reaches that gate. A
  `requires_worker_wheel` (readiness) recipe is the sealed / verified worker-**packaging** mechanism; it
  is **not** itself a paper experiment - a paper experiment additionally binds a matrix cell, an
  amendment, and reserved identities.
- **Paper telemetry is descriptive, not a gate.** `scientific_resource_complete` /
  `scientific_throughput_complete` are booleans on the run summary; a missing-paper-field note is
  appended to a human-readable report. Nothing blocks or fails a normal run.
- **Reserved identities, amendments, and experiment matrices live in `research/ieee-linux-training/`**
  (`validate_protocol.py`). The engine package does not import them; the normal `platform-plan` ->
  `platform-run` path does not enforce them.
- **The engine does carry several paper integrations** - the `required_git_ancestor` floor is NOT the
  only one. They currently include: the `required_git_ancestor` provenance plumbing; `REQUIRED_PAPER_FIELDS`;
  the scientific / paper completeness calculation; shared telemetry-contract fields; and wheel / source
  identity injection into run summaries. These integrations are **descriptive or opt-in** - they populate
  or annotate summaries and gate only sealed-research (`requires_worker_wheel`) recipes. They do **not**
  make the normal product run path depend on amendments, matrix membership, reserved identities, or paper
  promotion.

## Known gaps / planned work (not yet implemented)

These make the three tiers explicit and give the product a real VERIFIED tier. They are design intent,
not current behavior:

1. **Name the tier.** Introduce an explicit assurance selector (`standard` / `verified` /
   `sealed_research`) on the recipe/plan instead of overloading `requires_worker_wheel`.
2. **Implement the VERIFIED tier.** A product training recipe that installs a **pinned, hash-verified**
   worker package - reproducible enough for ordinary use - with environment lock + capability evidence
   and generic build provenance, but no reviewed git floor, reserved identity, amendment, or matrix cell.
   Today `backend-corpus-studio` uses loose version ranges; this fills the gap between STANDARD and
   SEALED_RESEARCH.
3. **De-research the shared vocabulary.** Rename research-flavored names on the general Environment
   Manager (e.g. `validate_wheel_provenance_for_scientific_admission` -> `..._for_sealed_admission`);
   keep the concepts in `research/`.
4. **Plugin/skill overlay split.** A product-first skill plus an optional research-overlay skill scoped
   to `research/ieee-linux-training/` and `docs/paper/` that loads only for paper work.

## Rule of thumb

If a requirement only exists because of the paper, it belongs in SEALED_RESEARCH. If a normal user
building a dataset and training a model locally would hit it, it must not depend on anything in
`research/ieee-linux-training/`.
