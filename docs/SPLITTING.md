# Dataset Splitting

Corpus Studio should split datasets into:

- train
- validation
- test

## v0.1

Basic random split with configurable desktop ratios and seed:

```text
default train: 90%
default validation: 5%
default test: remaining 5%
default seed: 42
```

The desktop app exposes train percentage, validation percentage, and seed.
The test split is derived from the remaining percentage so the UI cannot create
a fourth hidden split. Successful split settings are saved in the project's
`project.json` and restored when the project is reopened.

The engine CLI accepts the same values:

```powershell
python -m corpus_studio.cli split input.jsonl exports\my_dataset\splits instruction --train-ratio 0.8 --validation-ratio 0.1 --seed 123
```

The split report includes warnings when validation or test output is empty or
only one row. These warnings do not block file generation; they tell the user
that evaluation or regression scores from the split will be weak until more
examples are added or ratios are adjusted.

## Planned Split Hardening

Better split strategies:

- tag-balanced split
- source-aware split
- group-aware split
- dedupe-aware split
- leakage-checked split

## Leakage warning

Near-identical examples should not appear in both train and test sets.
