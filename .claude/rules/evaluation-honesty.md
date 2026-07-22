---
paths:
  - "engine/corpus_studio/evaluation/**"
---

# Evaluation honesty: null-with-reason, never a fabricated zero

The evaluation path must not manufacture a metric it could not measure.

- An unavailable metric (backend error, scorer error, a missing field) is null with a typed reason, not
  a plausible-looking `0.0`. The model card upholds this ("null-with-reason, never a fabricated 0") -
  the scorers and the aggregate must uphold it too.
- **Do not fold infrastructure-failure rows into a quality mean.** A `backend_error` / `scorer_error`
  row recorded as `score=0.0` measures "we could not evaluate this row," not "the model answered
  maximally wrong." Averaging it into the reported quality fabricates a zero and drags the headline
  number down. Exclude unavailable rows from the quality denominator and report them separately (a
  measured count with a typed reason).
- A non-parseable schema-conformance answer scores a measured `0` with a typed reason (that IS a real
  measurement), never a fabricated pass.
