# packages/shared

Shared contracts for Gomoku Platform V3.

The source of truth lives in `schemas/*.schema.json`.
Generated registries live in `generated/`.

Phase 1 keeps code generation intentionally lightweight:

- JSON Schema files are authored by hand
- A small script generates language-specific schema registries
- Type-safe clients and server bindings can be layered on top in later phases
