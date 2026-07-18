# ADR 0003: Python Engine with C# Desktop Shell

## Status

Superseded in part (2026-07-18) — the **Python-engine boundary stands**, but the **C# desktop
shell is being decommissioned** (#545; target UI = Tauri 2 + React, Rust core #522; see PR #555).

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
