# Security Policy

Corpus Studio is a local-first dataset application. Dataset files may contain private, proprietary, or sensitive material, so safety and traceability matter.

## Reporting issues

Report security concerns privately to the maintainers before public disclosure.

## Security priorities

- avoid accidental data leakage
- avoid unsafe default exports
- avoid destructive cleaning operations
- protect local project databases
- make provenance and license metadata visible
- detect likely private information where possible

## Sensitive data rule

Corpus Studio should never assume imported data is safe. Future versions should provide configurable PII detection and provenance warnings.
