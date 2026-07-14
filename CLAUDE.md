# CLAUDE.md

Guidance for Claude Code (and other AI coding agents) working in this repository.
The full agent contract lives in `AGENTS.md` and is imported here so it governs every
Claude Code session:

@AGENTS.md

## Current host (read before anything hardware-adjacent)

This repository now runs on a **native-Linux RTX 5070 host** (Ubuntu 24.04), not the
historical Windows/WDDM machine. The active checkout is
`/mnt/training-nvme/repos/CorpusStudio`. For the verified host facts — paths, GPU, and the
`HARDWARE_VERIFIED` `backend-corpus-studio` managed environment (and exactly what that
does and does *not* prove) — read [`docs/HOST_STATE.md`](docs/HOST_STATE.md).

Session state + roadmap remain in [`HANDOFF.md`](HANDOFF.md); the authoritative *feature*
state is [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md); the forward plan is
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md). Where an older doc still shows
Windows `C:`/`F:` paths or an "until the Linux NVMe is installed" precondition, `HOST_STATE.md`
supersedes it for *where you are*; the Windows/WDDM evidence is preserved as history, not deleted.
