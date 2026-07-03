# Dataset Debt (v1.1)

**Dataset debt** is the inventory of outstanding quality problems in a dataset —
the things you should *pay down* before training. The debt ledger reuses the
existing quality report (it adds **no** new detection) and reframes it to answer
one question the raw numbers don't: **"is this dataset train-ready, and if not,
what do I fix first?"**

## Debt vs quality vs gates

| | Answers | Shape |
|---|---|---|
| **Quality report** | "what did we detect?" | a flat bag of raw counts |
| **Gates** | "may this move forward *now*?" | pass / warn / block at a threshold |
| **Debt** | "how bad is it, and what do I fix first?" | a **normalized, ranked, graded ledger** with remediation |

The value debt adds over the quality report is exactly three things:

1. **Normalization** — counts become *rates* per dataset size. "8 duplicates" is a
   crisis in 20 rows (40%) and noise in 100k (0.008%). Rates make the numbers
   interpretable.
2. **Prioritization** — items are ranked by severity, so there is a clear #1 fix.
3. **One honest grade** — a single A–F health signal, plus a concrete paydown
   action per item.

## The ledger

`build_debt_report(quality_report)` (pure) emits a `DebtReport`:

- `grade` — **A–F**, or **`N/A`** for an empty dataset (0 rows is "no rows to
  assess", *not* grade A).
- `items` — a list of `DebtItem{category, severity, count, rate, message,
  remediation}`, **highest severity first**.
- `.clean` — true only when there are rows and no debt.

### Severity rules (documented, per category)

Severity is **coarse and rule-based**, never a fake-precise score.

| Category | Rule |
|---|---|
| `empty_rows`, `low_information` | rate > 0.10 → high, > 0.02 → moderate, > 0 → low |
| `exact_duplicates`, `normalized_duplicates` | rate > 0.05 → high, > 0.01 → moderate, > 0 → low |
| `secrets` (high-severity PII: keys/tokens/JWTs) | **present → critical** |
| `personal_data` (medium PII: emails/SSNs) | **present → high** |
| `synthetic_patterns` | max issue severity: high→high, medium→moderate, low→low |
| `token_length_outliers` (advisory) | rate > 0.10 → moderate, > 0 → low (capped) |
| `category_imbalance` (worst field's dominant share) | share > 0.90 → high, > 0.75 → moderate, > 0.50 → low |

> **Secrets/PII are graded by PRESENCE, never by rate.** A single leaked API key
> is `critical` no matter how large the dataset — normalizing a credential away by
> rate would be exactly the wrong call. `rate` is `null` for these (and for
> imbalance, which uses share).

### Grade rule

**F** if any item is critical; else **D** if any high; else **C** if any moderate;
else **B** if any low; else **A** (rows present, no debt).

## Command

```
# Prioritized, graded debt ledger (Markdown, or --json for the DebtReport)
python -m corpus_studio.cli dataset-debt <examples.jsonl> [--json]
```

## Implemented vs deferred

**Implemented (v1.1, engine):** `reporting/debt_report.py` (`DebtItem`,
`DebtReport`, `build_debt_report`, `render_debt_report_markdown`) and the
`dataset-debt` CLI, reusing `build_basic_quality_report`.

**Deferred:** desktop surfacing (a Debt tab / dashboard); **trend over time**
(is debt growing or shrinking, via quality history); folding gate results into the
ledger; remediation *actions* (the ledger recommends fixes, it does not apply
them); and any opaque numeric score (a grade is deliberately used instead).
