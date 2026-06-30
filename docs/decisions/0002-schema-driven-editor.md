# ADR 0002: Schema-Driven Editor

## Status

Accepted

## Context

Corpus Studio must support multiple dataset types without becoming a pile of hardcoded editors.

## Decision

Dataset schemas define fields, validation rules, editor hints, and export mappings.

## Consequences

Pros:

- easier to add new dataset types
- supports custom schemas later
- keeps UI and engine cleaner

Cons:

- more up-front design work
- schema migrations must be handled carefully
