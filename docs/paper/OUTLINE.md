# Local LLM Fine-Tuning on Consumer Hardware: Silent Failures and a Platform to Prevent Them

**Working title.** Framing (c): the empirical *pitfalls* motivate the *system*. IEEE-style structure
(convertible to `IEEEtran` two-column LaTeX). Every number below is measured on real hardware — an
NVIDIA RTX 5070 (12 GB, Blackwell sm_120) — during the development of CorpusStudio; nothing is
estimated. Fill each `» claim` with the figure/table noted.

---

## Abstract
Parameter-efficient fine-tuning (QLoRA) has made local LLM adaptation *seem* accessible on a single
consumer GPU. In practice it is riddled with **silent, hardware- and OS-specific failures** that
produce a "successful" run and a quietly broken model. We report an empirical study on a 12 GB
Blackwell GPU that surfaces five such failures — a Windows-only fused-attention deadlock, divergent
GPU behavior across native Windows / WSL2 / Linux, hard sequence-length memory walls, *silent data
truncation* that teaches a model to emit incomplete outputs, and a GPU-virtualization "wedge" state
that masquerades as a config bug. We then present **CorpusStudio**, a platform whose contracts,
honesty invariants, and guardrails detect or prevent each failure, turning cryptic cascades into
actionable diagnoses. » one-sentence result summary.

## I. Introduction
- **Motivation.** Local fine-tuning is desirable (privacy, cost, data control) and *marketed* as
  turnkey, but the gap between "the run finished" and "the model is correct" is wide and undocumented.
- **The core problem.** Failures are *silent*: a run completes at a green status while (a) using a
  slow fallback kernel, (b) spilling to system RAM at 100× slowdown, or (c) truncating every training
  example. The operator has no signal.
- **Contributions.**
  1. An empirical catalog of five silent failures on real 12 GB Blackwell hardware, each reproducible
     and generalizable beyond the specific card.
  2. A measured characterization of the memory/sequence walls for 7B QLoRA on 12 GB across three OS
     memory models.
  3. CorpusStudio: a platform with language-neutral run contracts, an *honesty* invariant (predicted
     vs. measured fit; a completed step ≠ a proven fit), per-OS platform detection, and guardrails
     that catch each failure — with the failures as the design rationale.

## II. Background & Related Work
- QLoRA / 4-bit NF4 + LoRA; flash vs. memory-efficient vs. math attention (O(seq) vs. O(seq²)).
- Consumer-GPU constraints; WDDM vs. Linux dedicated memory; WSL2 GPU paravirtualization (GPU-PV).
- Existing trainers (Axolotl, Unsloth, TRL) and where they leave the operator unguarded. » position.

## III. Empirical Findings — Silent Failures
### A. A Windows-only fused-attention deadlock on Blackwell
The fused flash SDPA *backward* deadlocks on sm_120 under the native-Windows WDDM driver. The identical
kernel passed under WSL2; bare Linux is still unverified and must not be inferred from that result.
» Table: MATH ~10466 ms vs. FLASH ~9 ms vs. MEM_EFFICIENT ~7 ms (raw kernel, WSL2, seq-1536,
fwd+bwd). The guard is evidence-aware and WDDM-specific, while every other host still needs a probe.

### B. Divergent GPU behavior across native Windows / WSL2 / Linux
The same card is represented as three platforms. » Table (measured Windows/WSL attention and spill
behavior; unverified Linux cells; direct vs. WDDM vs. GPU-PV access). WSL2 is a measured hybrid:
Linux CUDA userspace with WDDM-backed residency. Its evidence is not a bare-Linux result.

### C. Memory & sequence-length walls for 7B QLoRA on 12 GB
Raw flash attention is O(seq) and trivially fits (» 0.35 GB @ seq-4096); the *full model* is the wall.
» Table (true full-length, WSL2): seq-2048 → 14.1 GB / ~670 s/step, seq-2560 → 18.8 GB, seq-3072 →
24.4 GB (usable ceiling, ~460 s/step — 5–11 min/step), seq-3584 → fails. The card *spills-but-trains*
at 100× slowdown up to a ceiling, then hard-fails.

### D. Silent data truncation → a model that emits incomplete outputs
The most insidious failure. To fit 12 GB we set `sequence_len = 1536`; the training corpus (522
world-building examples) has a token-length distribution of **min 1802, mean 2241, max 3445** — so
**100 % of examples were truncated**, cutting the end (the model's target output) off every one. The
fine-tuned model learned to emit **incomplete/unterminated JSON**, and an eval "explained" it as a
generation-length artifact — the true cause was training-time truncation, unwarned. » Fig: token-length
histogram vs. seq_len cutoffs. Generalizable: any memory-driven `seq_len` < data length silently
degrades quality.

### E. GPU-PV "wedge" under iterative development
On WSL2, a crashed CUDA process leaves the GPU-PV layer in a poisoned state where **every subsequent
run fails identically with `cudaErrorNotReady`**, regardless of config — indistinguishable, without
tooling, from "my config is too big." A VM reset (`wsl --terminate`) clears it. This *contaminated our
own measurements* until isolated — a cautionary tale for anyone benchmarking iteratively on WSL2.

## IV. CorpusStudio — A Platform That Catches Them
Each subsection maps a failure (III.A–E) to a mechanism.
### A. Language-neutral run contracts & lifecycle
Profile → plan (hash-sealed `RunPlan`) → predict-fit → run (supervised worker) → measure-fit →
artifacts. Torch-free core; opt-in `[train]` extra. » Fig: lifecycle.
### B. Honesty invariants & the watchdog
Predicted vs. **measured** fit; `NATIVE_SAFE` is only ever earned by a *completed* run (a partial peak
from a failed run is `NATIVE_UNPROVEN`); an observed spill classifies as `ACCIDENTAL_WDDM_SPILL` from a
non-zero shared-memory fingerprint. → III.C.
### C. Per-OS platform detection & safe defaults
`OperatingSystem.{windows,wsl,linux}` + memory-residency model; the fused-flash-disable fires only on
native-Windows+Blackwell; elsewhere the probe must PASS on the exact environment before it proves
flash. WSL evidence and the unverified bare-Linux lane remain explicitly separate. → III.A, III.B.
### D. Guardrails
Truncation guardrail (warn when `seq_len` cuts the data; a pre-run `dataset-tokens` gate) → III.D;
GPU-health probe that classifies a *wedged* GPU and emits the OS-specific reset → III.E; actionable
spill guidance → III.C.

## V. Case Study & Evaluation
The World Bible Generator (WBG-7B) end-to-end: model-fetch → plan (sealed sdpa/flash on WSL) →
supervised subprocess run → measured `NATIVE_SAFE` → adapter → held-out schema-conformance eval. »
report the seq-4096-on-real-data fit measurement; the corrected (untruncated) re-train; before/after
output-completeness.

## VI. Discussion & Limitations
Generalizability beyond one card/model; single-GPU scope (multi-GPU FSDP as the scale path); the
unverified native-Linux path; threats to validity (the GPU-PV wedge contaminating measurements —
itself a finding).

## VII. Conclusion
Local fine-tuning's real risk is not *can't run* but *runs wrong, silently*. A platform that treats
the host honestly and guards the operator turns that risk into an actionable signal.

---
### Raw material (already in-repo — drop straight into the sections)
- `docs/RUNNING_ON_LINUX.md` (III.B, III.C tables), `RTX5070_TRAINING_FINDINGS.md` (III.A, III.C),
  the safe-spilling + training-runtime findings (III, IV), the platform contracts + watchdog +
  gpu_health + truncation guardrail source (IV). Measured numbers are reproducible via the scripts in
  `scripts/` and the platform CLI.
