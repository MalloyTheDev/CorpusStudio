# ADR 0001: Local-First by Default

## Status

Accepted

## Context

Dataset files may contain private or proprietary data. Corpus Studio should not require cloud services for core functionality.

## Decision

Core dataset authoring, validation, cleaning, splitting, and export will run locally.

## Consequences

Pros:

- better privacy
- easier offline use
- simpler early architecture
- user retains ownership

Cons:

- collaboration features come later
- cloud publishing requires optional integration
