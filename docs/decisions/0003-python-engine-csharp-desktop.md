# ADR 0003: Python Engine with C# Desktop Shell

## Status

Proposed

## Context

C# is strong for desktop UI. Python is strong for dataset processing.

## Decision

Use a C# desktop app for UX and a Python engine for dataset operations.

## Consequences

Pros:

- best tool for each layer
- Python ecosystem helps validation/import/export
- desktop app stays responsive

Cons:

- IPC boundary required
- packaging needs care
