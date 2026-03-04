# Status — Current Snapshot

## Summary

narrow is in the **planning complete / pre-implementation** state. All architectural decisions
are locked. Documentation structure mirrors nabled's patterns. Implementation has not begun.

## Crate State

| Aspect                | Status                                        |
|-----------------------|-----------------------------------------------|
| Cargo.toml            | Skeleton only. No dependencies yet.           |
| src/lib.rs            | Doc comment only. No code.                    |
| Module tree           | Not created. Defined in architecture.md.      |
| Dependencies          | None. arrow + ndarray to be added in Phase 1. |
| Tests                 | None.                                         |
| CI                    | None.                                         |
| Coverage              | N/A                                           |

## Design State

| Aspect                     | Status    | Document                  |
|----------------------------|-----------|---------------------------|
| Type mappings              | Locked    | DECISIONS.md D-010..D-015 |
| Trait hierarchy            | Locked    | DECISIONS.md D-050..D-052 |
| Null handling              | Locked    | DECISIONS.md D-020..D-021 |
| Ownership model            | Locked    | DECISIONS.md D-030..D-032 |
| Extension types            | Locked    | DECISIONS.md D-040..D-041 |
| Module structure           | Locked    | architecture.md           |
| Error model                | Locked    | architecture.md           |
| Allocation contract        | Locked    | PERFORMANCE_CONTRACTS.md  |
| Implementation plan        | Locked    | ROADMAP.md                |

## Dependencies on External Crates

| Crate       | Expected Version | Purpose                        | Status      |
|-------------|-----------------|--------------------------------|-------------|
| arrow       | Latest stable   | Arrow array types, ext types   | Not added   |
| ndarray     | 0.17            | Array views and owned arrays   | Not added   |

## Dependencies on Upstream Changes

See `NABLED_CHANGES.md` for full details.

| Change                           | Crate   | Status    | Blocking? |
|----------------------------------|---------|-----------|-----------|
| f32 first-class API support      | nabled  | Not started | No — narrow uses ndarray directly |
| CsrMatrixView with Arrow-native index types | nabled | Not started | No — narrow defines its own CsrView |
| Generic float trait              | nabled  | Not started | No — narrow's NarrowElement is independent |

None of the nabled changes block narrow's implementation. narrow uses ndarray directly and
defines its own view types. nabled changes will improve end-to-end ergonomics when both crates
are used together, but narrow is independently functional.

## Constraints In Force

1. Zero-copy bridge — no allocations on the conversion path
2. Vendor-agnostic — no knowledge of Qdrant, DataFusion, etc.
3. ndarray-independent — no dependency on nabled
4. Algebraic, compositional, homomorphic, denotationally sound
5. Test coverage >= 90%
6. Clippy -D warnings
7. Documentation on every public item

## Next Milestone

**Phase 1: Foundation** — Create crate skeleton, core traits, error types, module tree.
See ROADMAP.md Phase 1 and EXECUTION_TRACKER.md N-001 through N-007.
