# Contributing to Corpus Studio

Corpus Studio is MIT-licensed and open to contributions.

**New here?** Start with the [Developer Guide](docs/DEVELOPER_GUIDE.md) — a hands-on
walkthrough of the engine⇄desktop bridge, the seam-based view-model structure, the
recipe for adding a feature, and how the tests fake it all. See also
[ARCHITECTURE.md](docs/ARCHITECTURE.md), [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md),
and the [CLI reference](docs/CLI_REFERENCE.md).

## Contribution priorities

Early contributions should focus on:

1. schema correctness
2. dataset validation
3. clean import/export behavior
4. local-first reliability
5. documentation clarity
6. test coverage

## Development standards

- Keep the dataset engine deterministic.
- Do not silently mutate user data.
- Validation errors should be specific and actionable.
- Exports must be reproducible.
- Every new schema requires docs and examples.
- Every new cleaning rule should explain what it removes and why.

## Pull request checklist

- [ ] Code is formatted.
- [ ] Tests pass.
- [ ] New behavior is documented.
- [ ] Schema changes include examples.
- [ ] Export behavior is covered by tests.
- [ ] No private data or generated bulk datasets are committed.

## Dataset safety

Do not commit private, copyrighted, scraped, or user-sensitive datasets to the repository. Use tiny synthetic examples only.
