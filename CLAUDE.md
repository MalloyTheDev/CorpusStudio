# CLAUDE.md

Guidance for Claude Code (and other AI coding agents) in this repository. The full agent contract
lives in `AGENTS.md` and is imported here so it governs every session:

@AGENTS.md

## Context selection

Use the **corpus-studio** skill as the router and the deterministic assurance tooling
(`scripts/cs_assure.py`: `changeset`, `doclint`) for change-set and doc-trust facts. Load only what
the task activates - do not infer host, GPU, or research state unless the task is hardware- or
paper-adjacent.

- **Where you are / host facts:** [`docs/HOST_STATE.md`](docs/HOST_STATE.md) - the checkout path, GPU,
  the managed environment, and exactly what it does and does not prove. Resolve the checkout root with
  `git rev-parse --show-toplevel`; host paths are not hardcoded in guidance.
- **Feature state (authoritative):** [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md).
- **Session state + roadmap:** [`HANDOFF.md`](HANDOFF.md); forward plan
  [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).
- **Standard / verified / sealed-research boundary:** [`docs/PRODUCT_VS_RESEARCH.md`](docs/PRODUCT_VS_RESEARCH.md).

Where an older doc shows Windows `C:`/`F:` paths or an "until the Linux NVMe is installed" precondition,
`HOST_STATE.md` supersedes it for *where you are*; the Windows/WDDM material is preserved as history, not
current guidance.
