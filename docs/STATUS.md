# Status — Current Snapshot

Last updated: 2026-03-05

## Summary

ndarrow is in the **core bridge complete** state.
Core traits, error model, dense/sparse/tensor conversions, explicit null handling tiers, helper APIs,
CI, and release automation are implemented.
The current implementation milestone is complete, including post-core helper/property/benchmark
hardening items.

## Crate State

| Aspect                | Status |
|-----------------------|--------|
| Cargo.toml            | Workspace and crate dependencies configured (`arrow*`, `ndarray`, test/tooling deps). |
| src/lib.rs            | Public API module wiring and re-exports implemented. |
| Module tree           | `element`, `error`, `inbound`, `outbound`, `sparse`, `tensor`, `helpers` implemented. |
| Dependencies          | Added and pinned at workspace level. |
| Tests                 | Unit + integration tests for dense, sparse, tensor, null semantics, and zero-copy behavior. |
| CI                    | Implemented (`fmt`, `clippy`, feature checks, unit/integration tests, coverage, bench smoke). |
| Coverage              | Gate configured at 90% line coverage and currently passing (`90.33%` on latest `just checks`). |

## Implemented Capability Baseline

1. `NdarrowElement` trait with `f32`/`f64` support.
2. `NdarrowError` taxonomy.
3. `AsNdarray` for `PrimitiveArray<T>`.
4. `FixedSizeListArray -> ArrayView2<T>` conversion APIs (`validated`, `unchecked`, `masked`).
5. `IntoArrow` for `Array1<T>` and `Array2<T>`.
6. Sparse bridge APIs (`CsrMatrixExtension`, `CsrView`, inbound/outbound CSR paths).
7. Tensor bridge APIs for `arrow.fixed_shape_tensor` and `arrow.variable_shape_tensor`.
8. Explicit helpers (`cast_f32_to_f64`, `cast_f64_to_f32`, reshape helpers, layout normalization).
9. Explicit sparse densification helper (`densify_csr_view`) with CSR invariant validation.
10. Integration tests for round-trip correctness and zero-copy pointer guarantees (including sparse/tensor pointer identity checks).
11. Property-test coverage for dense/sparse/tensor round-trip invariants.
12. Benchmark harness with smoke-compatible public API conversion benchmarks plus baseline regression reporting/check gates.

## Dependencies on Upstream Changes

See `NABLED_CHANGES.md` for detail.

| Change | Crate | Status | Blocking? |
|--------|-------|--------|-----------|
| First-class `f32` support | nabled | Completed and released in nabled `0.0.4` | No |
| `CsrMatrixView` with Arrow-native index types | nabled | Completed and released in nabled `0.0.4` | No |
| View-accepting sparse ops | nabled | Completed and released in nabled `0.0.4` | No |
| Complex Arrow representation assessment | nabled | Out of nabled scope (tracked on ndarrow side) | No |

## Constraints In Force

1. Zero-copy bridge semantics for view/ownership-transfer paths.
2. Vendor agnosticism (no producer/consumer coupling).
3. ndarray independence (no hard dependency on nabled).
4. Explicit null handling at call sites.
5. Quality gates: `fmt`, `clippy -D warnings -W clippy::pedantic`, coverage >= 90%.

## Next Milestone

Current remaining design work is tracked as:

1. Complex-number Arrow representation decision and implementation.
