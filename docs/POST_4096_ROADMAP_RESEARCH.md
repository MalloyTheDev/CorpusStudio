# Post-4096 Roadmap Research (synthesis)

Generated 2026-07-19 by a multi-agent research workflow (7 tracks + synthesis) to ground the roadmap
after 7B QLoRA seq-4096 was achieved (flash SDPA + liger fused-CE + bnb paged-8bit-AdamW +
max_split_size_mb:128 on the 12GB RTX 5070). Clean-room note: the OBLITERATUS section derives ONLY from
public research, never that AGPL-3.0 codebase. Full per-track findings preserved in the workflow journal.

# CorpusStudio Post-4096 Roadmap — Decision-Ready Synthesis

Grounding note: sm_120 is proven WORKLOAD_VERIFIED for exactly one tuple — 7B QLoRA, seq-4096, flash SDPA + liger fused-CE + bnb paged-8bit-AdamW + `max_split_size_mb:128` on the 12GB RTX 5070. Everything below treats that as the only proven fact and every other capability as DECLARED until its own probe passes. "installed != proven" is applied literally: a passing tuple never unions into another.

---

## 1. Architecture + Dependency Recommendation

The governing pattern for all of this: **one worker-protocol adapter boundary** (the hash-sealed `ResolvedExecutionConfiguration` in → typed `RunEvent` channels + `ArtifactManifest` out). Every backend is a *peer* behind that seal. The torch-free control plane never imports any of these; the Rust core is the sole admitter. No backend's own launcher ever owns run identity or artifact admission.

### ADOPT NOW (harden, pin exact, conformance-test per env)

| Dep | Tier / role | Adapter boundary | One-line rationale | Risk |
|---|---|---|---|---|
| **torch cu128 / 2.11** | L3 backend-env, sole framework substrate | `FrameworkBackend` registry (DECLARED-only; do not add JAX/MLX) | The proven sm_120 substrate; flash SDPA is torch-native, zero added dep surface | Pin the **cu128 index source**, not just version — default pip pulls CPU build |
| **transformers** | worker model layer | behind OrchestratorAdapter seal | Arch-agnostic pure-Python model defs; already the win's basis | `trust_remote_code` + Hub-download = **security axis (G4)**: refusable manifest field, fail-closed by assurance tier, exact hash in EnvironmentLock |
| **peft** | worker adapter surface | behind sealed ResolvedExecutionConfiguration | Narrow, stable, does one thing — the reference QLoRA path | `adapter_task_type`/`export_format` Literal-locked to dense-QLoRA-SFT today (G8) |
| **trl** | worker orchestration | same seal | HF-maintained SFT orchestration; rode the win | Version-couples the transformers/trl/peft/bnb/liger **quintuple** — pin exact, wheel-seal, conformance-test |
| **bitsandbytes** | worker: NF4 + paged-8bit-AdamW | functional+hardware probe gate | Just worked at seq-4096 on sm_120 | **Highest single-dep ABI/supply-chain risk on new silicon** — history of lagging CUDA arches; WORKLOAD_VERIFIED for this tuple *only*, gate behind probe, keep a fallback optimizer path |
| **liger-kernel** | worker: fused-CE (+ fused RMSNorm/RoPE/SwiGLU) | probe-gated Triton | Triton compiles for sm_120; fused-CE is what broke the seq-4096 vocab-logits wall | Triton-compile-per-arch — probe, don't assume other fused ops inherit the CE pass |

**This TRL+PEFT quintuple is the reference backend.** Keep it, harden it, treat every other backend as an optional peer behind the same adapter. Do not chase alternatives before this one is hardened.

### BUILD FIRST-PARTY (libs are weak here)

1. **Structure-aware chunker + token-coverage ledger** (see §2) — no library does loss-free *structure-aware* document-level splitting with a supervised-token ledger. Pure control-plane analysis + thin worker materializer. Zero new heavy dep.
2. **Behavior Lab DirectionProbe + evidence bundle** (see §3) — clean-room from public research, never AGPL code.
3. **The adapter boundary + conformance harness itself** — the thing that lets you *add* backends without accumulation. Extend existing wheel-sealing to a per-backend conformance test.

### DEFER (DECLARED-only until an isolated backend reaches WORKLOAD_VERIFIED)

| Dep | Why defer | When to revisit |
|---|---|---|
| **Unsloth** | Strongest single-GPU QLoRA candidate for this exact host profile, BUT: sm_120 kernels are **per-model** (pypi build lacked them for some newer families; "Studio" distro had them → needs a probe, not a blanket claim); heavy transformers monkeypatching → tight version coupling. AGENTS.md refuses it on Windows/WDDM; native-Linux sm_120 makes it viable. | Stand up as a *peer* adapter once the reference backend is hardened; probe sm_120 per-model; keep Pro/multi-GPU gated paths out of scope |
| **PyTorch FSDP2 + FSDP-QLoRA** | Cleanest possible adapter (torch-native, no 3rd-party CUDA, no monkeypatch); officially-blessed FSDP1 successor. But single 12GB card gets ~nothing today | The right **DECLARED** multi-GPU/bigger-than-one-card backend to stand up first when multi-GPU lands — additive, contracts already dense-safe |
| **torch DCP (distributed checkpoint)** | Resharding-on-load (save N / load M) is the right answer for the sharded gap | Adopt *behind* the existing CheckpointManifest when FSDP/MoE/pretraining lands — manifest gains a shard/topology descriptor, DCP writes shards, torch-free verifier still admits |
| **Megatron-Core MoE** | De-facto MoE reference (dropless routing, aux + aux-loss-free balancing, z-loss, EP/TP/PP/CP, distributed ckpt reshard). But heavy (TE/apex/CUDA-arch native builds), multi-GPU-first, hostile to torch-free plane; sm_120 grouped-GEMM path **unproven** (depends on TE/CUTLASS) | Wrap as isolated `backend-moe-distributed` worker *only* for datacenter-Blackwell; never the consumer target |
| **HF native MoE (Mixtral/Qwen-MoE/DeepSeek/OLMoE via Trainer/PEFT/bnb)** | The single-device consumer MoE path; reuses the exact winning stack; `moe_inspector.py` already allowlists these config schemas | This is the natural MoE runtime continuation — but validate per §4 before any claim |

### Never
JAX/MLX/second framework, PyO3-primary coupling, any dep in the control plane, letting Trainer's `save_strategy` own resume truth (`resolve_checkpoint_execution_policy` already gates it).

---

## 2. Full Validated seq-4096 Run (Roadmap Item 1) — Design

### Long-context handling recommendation: **structure-aware CHUNKING as the honest default**, packing as an opt-in throughput mode, sliding-window rejected as default.

Rationale and the decision tree:

- **CHUNKING (build first-party, default)** — deterministically split an over-length record into ordered contiguous chunks, each a full training row, **nothing dropped**. This is the honest default under no-silent-truncation. **Correctness-critical: split must be structure-aware** — chunk at message/turn boundaries; never orphan a completion from its prompt. A naive token-stream split severs the assistant target from its context and silently corrupts supervision — that is a silent-loss bug wearing a "no truncation" badge.
- **PACKING via TRL `SFTConfig` — opt-in only, and only `bfd_split` / `wrapped`.** `packing=True, strategy='bfd'` **TRUNCATES overflow → token loss** — this must be refused under the invariant. `bfd_split` (splits long seqs into ≤max_length chunks, preserves all tokens, per Fewer-Truncations 2404.10830) is the acceptable packing strategy; `wrapped` preserves tokens across pack boundaries. Packing sm_120 status is **unknown** → probe before claiming, and packing requires correct position_ids/attention-masking to avoid cross-document attention contamination — verify in the conformance test.
- **The genuine hard case:** a single *supervised span* alone exceeds seq_len. That is a real **refuse-or-allow operator decision surfaced explicitly**, never a silent cut. Default = `refuse` (extends the existing `truncation_policy='refuse'` row-level gate).
- **SLIDING WINDOW — reject as default** for SFT: it duplicates/reweights tokens and muddies supervised-label accounting; keep it out unless a specific objective declares it.

### Token-coverage / truncation metrics to report (extend, don't replace)

The existing `analyze_truncation()` / `TruncationReport` / `truncation_policy` is the foundation — but it counts truncated **examples, not tokens**. Add a **token-coverage ledger** (pure fn, torch-free-analyzable):

- `input_tokens_total`, `retained_tokens`, `dropped_tokens`, `coverage_pct`
- **`supervised_label_coverage`** — retained vs dropped *supervised* (loss-bearing) tokens (the metric that actually matters for SFT)
- `chunks_per_record` distribution, `boundary_severances` (count of any prompt/completion orphaning — must be 0 for a clean chunking run)
- `records_refused` + reason, `single_span_over_seqlen` count
- keep existing: `n_truncated`, `pct`, `max`, `median`, `seq_len_for_zero_truncation`
- **Gate:** admission refuses if `dropped_supervised_tokens > 0` under `refuse` policy, or if `boundary_severances > 0` ever.

### End-to-end pipeline validation checklist (ingest → repro)

1. **Ingest** — dataset hash pinned; license fail-closed check *before* any read; provenance recorded.
2. **Preprocess/chunk** — structure-aware chunk; emit coverage ledger; `boundary_severances == 0`; refused rows surfaced.
3. **Tokenize** — tokenizer hash pinned; verify token-coverage ledger matches post-tokenization reality (no silent tokenizer-side truncation); `examples_over_sequence_len` preflight.
4. **Train** — sealed `ResolvedExecutionConfiguration`; the exact proven tuple (flash SDPA, liger fused-CE, bnb paged-8bit-AdamW, `max_split_size_mb:128`); telemetry completeness enforced.
5. **Checkpoint/resume** — CheckpointManifest sealed (complete-marker + per-file sha256 + traversal defense, fail-closed); worker verifies optimizer+scheduler+scaler+RNG+sampler **before** loading any tensor; prove resume across a fresh process. **Honesty: claim CPU bitwise-identical only; do NOT claim GPU bitwise** (non-deterministic reductions).
6. **Eval** — quality gate on held-out; regression vs baseline; no eval on training data (contamination check).
7. **Artifact integrity** — complete `ArtifactManifest` (base-model hash, dataset/tokenizer/objective/seed, coverage ledger embedded); Rust core admits only on completeness.
8. **Export** — adapter safetensors as durable unit; license terms re-checked; no silent partial export.
9. **Repro** — re-run from sealed config reproduces the manifest; wheel-seal proves byte-identical worker.

---

## 3. Obliteratus Clean-Room Map

**Licensing guardrail (load-bearing):** the reference implementation is **AGPL-3.0**. Derive **only from the public research papers and their described math**, never from that codebase. No copied code, no adapted snippets, no reading its source to "check." Every design below cites public research and is implemented first-party. Behavior Lab is a **GATED** product area — design/study only until gates clear.

| Concept | Original research (public) | CorpusStudio-native Behavior Lab design | Honesty / guardrail |
|---|---|---|---|
| **Behavior-direction discovery** (diff-in-means in residual stream separating behavior-present vs behavior-absent cohorts at a layer/token pos) | Arditi et al. 2024 (2406.11717); Turner 2023 ActAdd (2308.10248); Rimsky 2024 CAA (2312.06681); Alain & Bengio 2016 (probing) | **`DirectionProbe`**: isolated Python worker caches activations for two user-declared cohorts, returns **only** the direction + separability stats as an evidence bundle. Provenance binds direction to (model hash, tokenizer, layer/pos, cohort dataset hashes, seed). Dense-safe/MoE via (layer, position, optional expert-id) addressing. | A direction is **DECLARED, not proven causal** until an intervention eval measures effect. Control plane never imports torch; Rust is sole admitter. |
| **Inference-time steering vectors** (add/subtract scaled direction via forward hooks — reversible, weight-preserving) | Turner 2023 ActAdd; Rimsky 2024 CAA; representation-engineering framing | **`SteeringHook`** intervention in the worker: apply direction at chosen layers at inference; **reversible**, no weight change. Emits a measured effect delta as evidence. | Reversible + weight-preserving = lowest-risk intervention; this is what *proves* a direction causal → run it before any weight edit. |
| **Weight-orthogonalization / projection editing** (permanently project direction out of writing matrices: attn out-proj, MLP down-proj, embeddings — training-free) | Arditi et al. 2024 (orthogonalization form); ROME (2202.05262) / MEMIT (2210.07229) closed-form low-rank family; norm-preserving projection variants | **Model & Release Studio weight-surgery path, gated behind Behavior Lab authorization**: worker emits a candidate weight delta/adapter; Rust admits **only** with complete ArtifactManifest (base-model hash, direction provenance, projection method, layers touched, **norm-delta report per matrix**). | **License fail-closed on base model** before emitting. No-silent-truncation analogue: **record every matrix touched + its norm change; never a silent partial edit.** Weight surgery only after inference-time steering has measured the effect. |

**Sequencing within Behavior Lab:** DirectionProbe (declare) → SteeringHook (measure/prove causal, reversible) → weight surgery (permanent, gated, fully manifested). Never skip to weight surgery.

---

## 4. MoE Reliability Validation Checklist

Map to `docs/MOE_ARCHITECTURE.md` sec.3/10 and TRAINING_SYSTEMS_ARCHITECTURE G8. **Nothing here is assumed — each is VALIDATE.** Target families: Mixtral / Qwen-MoE / DeepSeek-MoE / OLMoE via HF native modeling (the consumer path); Megatron-Core only for datacenter-Blackwell.

**Router behavior**
- [ ] Router logits/top-k selection numerically stable at sm_120 fp precision
- [ ] Per-expert **load distribution** telemetry parsed into typed RunEvent channels (not framework stdout)
- [ ] Dropped/overflow token counts surfaced — **dropless (no-drop/no-pad) routing verified**, not assumed

**Load balancing / aux losses**
- [ ] Aux-loss balancing wired and logged; **aux-loss-free (bias-based) balancing** path validated separately
- [ ] Router **z-loss** present and logged
- [ ] Balancing actually improves expert utilization (measure, don't assume the loss term works on sm_120)

**Expert loading / placement**
- [ ] Grouped-GEMM expert path **sm_120-UNVERIFIED** (depends on TransformerEngine/CUTLASS) — flag explicitly, probe before any claim
- [ ] Expert placement deterministic and recorded in sealed config

**Distributed (EP/TP/PP/CP)** — defer to multi-GPU; DECLARED-only now
- [ ] Contracts confirmed dense-safe/MoE-compatible (already true) — no single-GPU assumption baked into manifest

**Expert checkpoint fidelity (the G8 gap)**
- [ ] Expert-shard save/load round-trips bitwise (CPU) through CheckpointManifest + shard/topology descriptor
- [ ] **EP-degree-rebind / reshard** fidelity (save N ranks, load M) — the flagged gap; validate via DCP *behind* the manifest when multi-GPU lands
- [ ] Distributed-optimizer resharding verified

**Expert-specific evaluation**
- [ ] Per-expert grad coverage (no dead experts silently untrained)
- [ ] Expert-utilization eval distinct from aggregate loss
- [ ] `moe_inspector.py` allowlist matches the actual runtime config schema per family

**Adapter boundary invariant:** control plane emits hash-sealed ResolvedExecutionConfiguration → worker translates to MCore/HF args → framework telemetry (per-expert load, dropped/overflow, aux/z-loss, grad coverage) parsed back into typed channels. **Never let the framework launcher own run identity or artifact admission.**

---

## 5. Prioritized, Sequenced Backlog

Effort: S = ~1-3 days, M = ~1 week, L = ~2-4 weeks. Each = one coherent CI-green PR slice (or a small epic). CI stays green, coverage ≥88, contracts regenerated when `platform/` changes.

| # | Item | Why here / dependency | Effort |
|---|---|---|---|
| **1** | **Harden the reference backend + per-env conformance test.** Pin the exact transformers/trl/peft/bnb/liger quintuple; extend wheel-sealing to a functional+hardware conformance probe that reproduces the seq-4096 tuple. | Protects the only proven asset; unblocks trusting any future claim. Highest-leverage, lowest-risk. | M |
| **2** | **Token-coverage ledger** (extend `analyze_truncation`/`TruncationReport`): tokens + supervised-label coverage, `boundary_severances`, refuse gate on dropped supervised tokens. Pure control-plane, torch-free. | Closes the "counts examples not tokens" gap; prerequisite for honest long-context. No new dep. | S-M |
| **3** | **Structure-aware chunker** (control-plane analysis + thin worker materializer): message/turn-boundary split, single-span-over-seqlen → refuse/allow surfaced. | The honest long-context default; depends on #2's ledger. Build-where-weak. | M |
| **4** | **Full validated seq-4096 pipeline run** (§2 checklist end-to-end): ingest→…→repro with chunking + ledger + CheckpointManifest resume + artifact integrity + export + repro-from-sealed-config. | Roadmap item 1 completion; consumes #1-3. Produces the first fully-validated long-context evidence bundle. | L |
| **5** | **TRL `bfd_split` packing as opt-in throughput mode** behind a probe (position_ids/masking correctness verified; reject `bfd`). | Additive throughput; only after chunking is the trusted default. sm_120 packing UNVERIFIED → probe. | M |
| **6** | **Behavior Lab `DirectionProbe`** (clean-room, evidence-bundle-only, provenance-bound). Gated product area → design + worker probe, no weight mutation. | First Behavior Lab slice; lowest-risk (read-only). AGPL guardrail enforced. | M |
| **7** | **Behavior Lab `SteeringHook`** (reversible inference-time intervention + measured-effect eval). | Proves a direction causal; prerequisite before any weight surgery. Depends on #6. | M |
| **8** | **Unsloth peer adapter** behind the worker seal, **per-model sm_120 probe** (no blanket claim), Pro/multi-GPU paths out of scope. | Strongest single-GPU speed/VRAM win for this host — but only after reference backend + conformance harness (#1) exist to compare against. | M-L |
| **9** | **HF-native single-device MoE validation** (Mixtral/Qwen-MoE/DeepSeek/OLMoE) against the §4 checklist; grouped-GEMM sm_120 flagged UNVERIFIED. | MoE reliability on the consumer path; reuses the winning stack; `moe_inspector` already allowlists these. | L |
| **10** | **Weight-surgery path** (Model & Release Studio, gated behind Behavior Lab auth): candidate delta/adapter, complete manifest with per-matrix norm-delta, license fail-closed. | Highest-risk Behavior Lab step; only after #7 measures causal effect. | M |
| **DEFER** | FSDP2/FSDP-QLoRA + DCP behind CheckpointManifest (multi-GPU); Megatron-Core `backend-moe-distributed` (datacenter-Blackwell only). Stand up as **DECLARED** backends; do not implement until multi-GPU hardware exists. | Contracts already dense-safe → additive later. Premature now. | — |

**Critical path:** 1 → 2 → 3 → 4 (validated long-context) runs in parallel with 6 → 7 (Behavior Lab read-only → causal), with 5 and 8 as throughput/backend peers once #1 lands, and 9/10 gated behind their prerequisites.

### Honesty flags (carry forward, do not launder)
- sm_120 proven for **one tuple only**; bnb + liger + packing + grouped-GEMM each need their **own** probe.
- Unsloth sm_120 is **per-model**, not blanket.
- GPU bitwise-resume is **not** claimed (only CPU bitwise).
- Megatron/grouped-GEMM MoE on sm_120 is **unverified** — datacenter-Blackwell only.
- A direction is **declared, not causal**, until intervention eval measures it.
- Contracts, fake workers, CI, and a passing env probe are **not** proof a workload trains.
