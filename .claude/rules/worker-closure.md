---
paths:
  - "engine/corpus_studio/platform/worker.py"
  - "engine/corpus_studio/platform/runners.py"
  - "engine/corpus_studio/platform/artifacts.py"
  - "engine/corpus_studio/platform/supervisor.py"
  - "engine/corpus_studio/platform/execution_config.py"
  - "engine/corpus_studio/platform/planner.py"
  - "engine/corpus_studio/training/trainer.py"
---

# Worker execution closure

These modules are (or are reachable from) the supervised worker child:
`worker.py::run_worker` -> `supervisor.py::execute_run` -> success admission (`artifacts.py`) ->
`runners.py` -> `training/trainer.py`. **`artifacts.py` and `runners.py` are worker code even though
they live under `platform/`.**

- A change to worker-execution bytes needs a fresh pinned worker package + new environment locks. In
  sealed research it additionally needs a new dated amendment, an effective-matrix bump, a superset
  reserved-identity set, a reproducibly rebuilt wheel, and new `-vN` environments.
- **Do NOT classify from a file list - TRACE the import path.** Lazy (function-local) imports make
  modules like `planner` runtime-reachable from the worker. Treat any worker-reachable module as
  `RUNTIME_REACHABLE_REVIEW_REQUIRED` until a symbol-level trace proves non-impact. Write the verdict
  down (WORKER_CHANGE_REQUIRED vs control-plane-only).
- Once identities are instantiated (wheel built, environments sealed, a run produced), a "reuse the
  lineage" rationale that depended on non-instantiation no longer holds - prove non-impact or bump the
  lineage.
