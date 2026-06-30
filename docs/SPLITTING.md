# Dataset Splitting

Corpus Studio should split datasets into:

- train
- validation
- test

## v0.1

Basic random split:

```text
train: 90%
validation: 5%
test: 5%
```

## v0.2+

Better split strategies:

- tag-balanced split
- source-aware split
- group-aware split
- dedupe-aware split
- leakage-checked split

## Leakage warning

Near-identical examples should not appear in both train and test sets.
