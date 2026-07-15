# Amendment 0001: corrected v3 smoke bindings

Amendment ID: `cs-ieee-linux-training-v1-amendment-0001`

Previous protocol version: `1.0.0`

Effective protocol version: `1.0.1`

Recorded: `2026-07-15T05:56:57Z`

Author: CorpusStudio research engineering

Status: prospective

Analysis class: production-path bring-up only; not a preregistered paper-matrix result

## Reason

The frozen `1.0.0` matrix named the initially proposed first-party environment IDs
`backend-corpus-studio-research-math-v1` and
`backend-corpus-studio-research-flash-v1`. Those identifiers are not an executable matched pair:

- the math-v1 creation is preserved as `BROKEN` without a lock;
- no research-flash-v1 environment was created (the similarly named
  `backend-corpus-studio-readiness-flash-v1` is historical readiness evidence, not the research
  execution path);
- the matched v2 pair used an earlier worker and both first production-path attempts failed before
  optimizer step 1 because the worker inspected a BF16 pre-accumulation autograd tensor instead of
  the sealed FP32 materialized leaf gradient;
- PR #438 corrected that verifier, changing worker bytes and therefore requiring new immutable
  environment locks and new RunPlans.

This amendment changes identity bindings only. It was not motivated by observed math-versus-flash
performance, and it does not alter a training hyperparameter, input, success criterion, failure
criterion, stopping rule, exclusion, or statistical method.

## Exact prospective bindings

Both corrected environments use the same worker artifact:

- worker source commit: `16ef6e95722ec3988ee8826b45333c9356ef76f9`;
- distribution/version: `corpus-studio-engine==1.3.0`;
- wheel SHA-256: `6ecc82595af761142b723017a31b980241fe6ef4afebf0a2223f90b8bcef724d`;
- wheel METADATA SHA-256:
  `c8eb3e03d457da4495545bc0bb355131a02d3d48f397bc4a9c07fe1cff9704fe`;
- manager version: `1.2.0`.

For the logical execution path `first-party-math`, replace the proposed v1 binding for the corrected
Phase 3 smoke with:

- environment ID: `backend-corpus-studio-research-math-v3`;
- lock ID: `lock-cd86808ce8e96533b6d6`;
- lock hash: `cd86808ce8e96533b6d6d3a0b4c0472e2e6e27ecf8d25bad916a9a08d4e6887d`;
- required probe: `cuda_qlora_math_execution`;
- repeat-probe evidence hash:
  `89b460d9c19a90a3b56078e48c7735d492c57b918a62f954adf3ff353a956338`;
- forced kernel: `torch_sdpa_math`;
- toggles: flash `false`, memory-efficient `false`, math `true`.

For the logical execution path `first-party-flash`, replace the proposed v1 binding for the corrected
Phase 3 smoke with:

- environment ID: `backend-corpus-studio-research-flash-v3`;
- lock ID: `lock-a2b839b160e4676d968c`;
- lock hash: `a2b839b160e4676d968cdd006040dde6cce756c30f51a2c92ef2b1442132aa2a`;
- required probe: `cuda_qlora_sdpa_flash_execution`;
- repeat-probe evidence hash:
  `ac5f19485ff508e0583542ba2ae02400718128c8d4af9b264f3970a154bddc8e`;
- forced kernel: `torch_sdpa_flash`;
- toggles: flash `true`, memory-efficient `false`, math `false`.

The matched-environment comparison remains a prerequisite. Its result is
`MATCHED_FOR_ATTENTION_KERNEL_STUDY`, and it establishes equal worker bytes, package versions,
package sources, artifact hashes, Python/PyTorch/CUDA identity, BF16 activation policy, NF4, and
double quantization. Environment-root-bound RECORD and installed-tree hashes remain distinct evidence,
not semantic package differences.

## Plan timing and identity rule

Executable corrected-smoke RunPlans must be minted after the final commit containing this amendment is
merged. They must record that final repository commit separately from the older worker source commit,
bind the v3 locks above, use fresh plan and execution identities, and pass a field-by-field normalized
pair audit before dispatch.

Any v3 plans minted before this amendment's final repository commit are preserved as unexecuted
candidate evidence. They are not dispatched, copied, or reused. Their existence does not authorize a
run. The historical failed v2 plans and runs remain reconstruction-only evidence under the same rule.

Any later worker-code change requires another wheel, new environment IDs and locks, another
prospective amendment, and completely fresh RunPlans.

## Affected work

This amendment applies only to the corrected Qwen2.5 0.5B, sequence-256, three-step Phase 3
production-path smoke pair and to the admission identity used to prove the first-party paths before
harness work begins.

The pair is bound to the immutable `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775` and the eight-row
`pipeline-smoke-fixture-v2` (`a322b1059709a30c4f927b087e0e655724d6e2a06873175b71d03073a17fa289`).
It does not use the user's in-progress 500-output training corpus. That corpus and every 7B workload
remain unavailable until the user explicitly marks them ready.

It does not turn that pair into either of the following:

- a `primary_matrix` cell (whose preregistered lengths begin at 512); or
- the matrix's three-step feasibility trial (which uses the separately frozen matrix seed policy).

Before primary-matrix data collection, the then-current worker and environment identities must receive
their own prospective binding if they differ from this amendment. Primary cells, repeat counts,
counterbalancing, warm-up handling, metrics, and statistical summaries remain exactly as frozen in
protocol `1.0.0`.

## Results visible before this amendment

The following operational evidence was visible:

- the math-v1 failed creation state;
- matched v2 environment-level tiny QLoRA probe passes;
- one v2 math and one v2 flash real-model attempt, each completing zero optimizer steps and writing no
  adapter or checkpoint;
- the common v2 gradient-verifier failure classification;
- the reviewed PR #438 correction and its unit tests;
- v3 environment creation, repeat probes, matched-stack comparison, and unexecuted normalized plan
  candidates.

No corrected v3 model load, adapter insertion, optimizer step, terminal workload success, performance
measurement, or sequence-4096 outcome was visible. Neither v2 failure is reclassified as a successful
smoke or paper result.

## Compatibility and claim impact

- Protocol `1.0.0` and all seven frozen base files remain byte-for-byte unchanged.
- All v1, v2, readiness, failed-run, and diagnostic evidence remains preserved under its original
  identity.
- Protocol `1.0.1` consumers apply this amendment as an overlay; they do not rewrite the base matrix.
- This amendment authorizes no execution by itself. The corrected pair retains its separate explicit
  approval gate and zero automatic retries.
- No claim is added for optimizer-step success, sequence 4096, Windows, WSL, macOS, external
  flash-attn, offload, MoE, model quality, or comparative performance.

## Evidence roots

- matched v3 environments:
  `/mnt/training-nvme/corpusstudio/evidence/backend-corpus-studio-research-matched-v3/`;
- preserved failed v2 attempts:
  `/mnt/training-nvme/corpusstudio/evidence/production-smoke-matched-v2/20260715T034634Z/`;
- pre-amendment unexecuted v3 plan candidates:
  `/mnt/training-nvme/corpusstudio/evidence/production-smoke-matched-v3/20260715T052743Z/`.

Large evidence remains outside the source repository. Paths are host-specific evidence references, not
portable product defaults or shared contract fields.
