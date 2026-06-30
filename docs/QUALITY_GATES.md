# Quality Gates

Quality gates prevent bad examples from silently entering training exports.

## v0.1 gates

Required:

- valid JSON
- row is a JSON object
- required fields present
- required fields non-empty
- declared field types match schema definitions
- chat messages include valid role/content structure
- schema ID known
- export format supported

Warnings:

- very short output
- very long output
- duplicate example ID
- missing tags
- missing source metadata
- missing license metadata

## v0.2 gates

- duplicate content detection
- near-duplicate content detection
- train/test leakage detection
- token length outlier detection
- low-information text detection
- category imbalance warnings

## Code dataset gates

- language field present
- code block not empty
- optional syntax parse
- optional tests field validation

## Image-caption gates

- image file exists
- caption not empty
- image resolution recorded
- duplicate captions warning
- missing license warning

## Principle

A quality gate should explain:

1. what failed
2. why it matters
3. how the user can fix it
