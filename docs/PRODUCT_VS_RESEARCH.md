# Product vs research boundary

CorpusStudio is a **local-first, end-to-end AI development ecosystem and IDE** with seven co-equal product
areas (see [`PRODUCT_AREAS.md`](PRODUCT_AREAS.md) and [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md)). Research is the
**supporting Evidence & Experiments track** - one area among seven - that validates the product and
documents discoveries; it does not define it. The **native-Linux 7B research paper** under
[`../research/ieee-linux-training/`](../research/ieee-linux-training/) and [`paper/`](paper/) is a
**separate evidence program that uses** CorpusStudio to verify the training engine can train a 7B model at
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

## Implementation status

The explicit assurance-tier selector now exists (#492): `EnvironmentRecipe.assurance_tier`
(`standard` / `verified` / `sealed_research`) is chosen **directly** rather than inferred from
`requires_worker_wheel`, and a recipe validator keeps the tier and the pinned-wheel mechanism
consistent (STANDARD must not pin a wheel; VERIFIED / SEALED_RESEARCH must). `requires_worker_wheel`
remains the exact-wheel packaging / provenance-admission **mechanism** - now GOVERNED by the tier, not a
tier itself. Concretely:

- `backend-corpus-studio` (`assurance_tier=standard`, `requires_worker_wheel=false`) uses the loose
  **STANDARD** path.
- The worker-wheel readiness recipes are classified **SEALED_RESEARCH**: a pinned wheel AND the reviewed
  per-lineage git floor. The reviewed `required_git_ancestor` floor (and a reviewed `worker_source_commit`)
  are now required at plan time **iff** the tier is `sealed_research`, keyed on the tier rather than on
  the `requires_worker_wheel` mechanism.
- A run is **SEALED_RESEARCH** only when it is *additionally* bound to the paper's amendment, effective
  matrix, reserved identities, matrix cell, and study-evidence requirements (still only in `research/`).

The **VERIFIED tier now resolves as a distinct plan-time mode** (#492): a `verified` worker-wheel recipe
is pinned + provenance-admitted but resolves with **no** reviewed git floor, and a floor / source-commit
handed to it is refused. Two pieces remain **follow-ups**: no *builtin product* recipe ships at the
VERIFIED tier yet, and the env-CREATE admission gate still *always* requires an embedded floor in the
wheel, so building + admitting a verified wheel end to end is not yet wired. See "Known gaps" below.

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

The separation already exists in code, mostly as an implicit mechanism rather than a named tier:

- **The exact-wheel provenance admission gate is conditional.** `EnvironmentManager` runs the
  build-provenance admission (reviewed floor + embedded `BUILD_PROVENANCE.json` + optional `source_commit`
  match) only when `recipe.requires_worker_wheel` is true. The standard product training backend
  `backend-corpus-studio` has `requires_worker_wheel=false`, so it never reaches that gate. A
  `requires_worker_wheel` (readiness) recipe is the exact-wheel packaging / provenance-admission
  mechanism; it is **not** itself a paper experiment - a paper experiment additionally binds a matrix
  cell, an amendment, and reserved identities.
- **Paper telemetry is descriptive, not a gate.** `scientific_resource_complete` /
  `scientific_throughput_complete` are booleans on the run summary; a missing-paper-field note is
  appended to a human-readable report. Nothing blocks or fails a normal run.
- **Reserved identities, amendments, and experiment matrices live in `research/ieee-linux-training/`**
  (`validate_protocol.py`). The engine package does not import them; the normal `platform-plan` ->
  `platform-run` path does not enforce them.
- **The engine integrations split into three accurate categories** (the `required_git_ancestor` floor is
  only one part of the first):
  - *Shared exact-wheel integrations* - embedded build provenance, wheel / source identity, and the
    exact-wheel admission gate (including the `required_git_ancestor` plumbing). These belong to the
    packaging / provenance-admission mechanism (usable by VERIFIED or SEALED_RESEARCH), not to the paper
    alone.
  - *Shared descriptive paper fields* - `REQUIRED_PAPER_FIELDS` and the scientific / paper completeness
    summaries. These annotate run summaries and are descriptive or opt-in.
  - *Paper-governance bindings* - amendments, the effective matrix, reserved identities, matrix
    membership, matched trials, and promotion rules. These live only in `research/ieee-linux-training/`.
  **Neither the exact-wheel integrations nor the descriptive paper fields, on their own, make an
  environment or run a paper experiment** - only the paper-governance bindings do. None of these makes the
  normal product run path depend on amendments, matrix membership, reserved identities, or paper
  promotion.

## Status and remaining work

The three tiers are now explicit and the product has a plan-time VERIFIED tier. What has landed vs. what
remains:

1. **Name the tier - DONE (#492).** `EnvironmentRecipe.assurance_tier` (`standard` / `verified` /
   `sealed_research`) is chosen directly rather than inferred from `requires_worker_wheel`, with a
   validator enforcing tier <-> pinned-wheel consistency; the reviewed git floor keys on
   `sealed_research`, not the worker-wheel mechanism.
2. **VERIFIED tier - PLAN-PATH DONE (#492).** A `verified` worker-wheel recipe resolves floor-free
   (pinned + provenance-admitted, no reviewed floor). **Remaining:** no builtin *product* recipe ships
   at VERIFIED yet, and the env-CREATE admission gate still always requires an embedded floor in the
   wheel, so building + admitting a verified wheel end to end is not wired. Today `backend-corpus-studio`
   still uses loose version ranges.
3. **De-research the shared vocabulary - DONE (#493).** The general admission gate was renamed
   `validate_wheel_provenance_for_scientific_admission` -> `..._for_sealed_admission`, and "scientific"
   was scrubbed from the general Environment-Manager / build-provenance comments (sealed environment /
   host / provenance / admission). The descriptive PAPER-telemetry fields (`scientifically_complete`,
   `ScientificCompleteness`, `scientific_resource_complete` / `_throughput_complete`) are **intentionally
   kept** - they measure PAPER completeness, the concept that belongs with `research/`.
4. **Plugin/skill overlay split - remaining.** A product-first skill plus an optional research-overlay
   skill scoped to `research/ieee-linux-training/` and `docs/paper/` that loads only for paper work.

## Rule of thumb

If a requirement only exists because of the paper, it belongs in SEALED_RESEARCH. If a normal user
building a dataset and training a model locally would hit it, it must not depend on anything in
`research/ieee-linux-training/`.
