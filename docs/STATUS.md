# Status — Current Snapshot

Last updated: 2026-03-12

## Summary

ndarrow is now release-ready for the downstream-driven checkpoint 1 hardening round.
Core traits, error model, dense/sparse/tensor/complex conversions, explicit null handling tiers,
helper APIs, extension registry, prelude exports, CI, and release automation are implemented.
The numerical-object null contract has now been tightened for `FixedSizeList -> ArrayView2`
masked ingress, higher-rank complex matrix/fixed-shape tensor carriers are first-class, batched
sparse matrices now have an explicit `ndarrow.csr_matrix_batch` carrier, and ragged complex
tensors now compose through canonical `arrow.variable_shape_tensor<ndarrow.complex*>`.
The downstream checkpoint framing is now satisfied: checkpoint 1 is closed and `ndarrow` is ready
for the next release under the concept-first standalone / `rows-of-X` bridge model.

## Crate State

| Aspect                | Status |
|-----------------------|--------|
| Cargo.toml            | Workspace and crate dependencies configured (`arrow*`, `ndarray`, test/tooling deps). |
| src/lib.rs            | Public API module wiring and re-exports implemented (`complex`, `extensions`, `prelude` included). |
| Module tree           | `element`, `error`, `inbound`, `outbound`, `sparse`, `tensor`, `complex`, `extensions`, `helpers`, `prelude` implemented. |
| Dependencies          | Added and pinned at workspace level. |
| Tests                 | Unit + integration tests for dense, sparse, tensor, null semantics, and zero-copy behavior. |
| CI                    | Implemented (`fmt`, `clippy`, feature checks, unit/integration tests, coverage, bench smoke). |
| Coverage              | Gate configured at 90% line coverage and currently passing (`90.83%` on the latest full `just checks`). |

## Implemented Capability Baseline

1. `NdarrowElement` trait with `f32`/`f64` support.
2. `NdarrowError` taxonomy.
3. `AsNdarray` for `PrimitiveArray<T>`.
4. `FixedSizeListArray -> ArrayView2<T>` conversion APIs (`validated`, `unchecked`, `masked`) with masked ingress defined as outer-row validity only.
5. `IntoArrow` for `Array1<T>` and `Array2<T>`.
6. Sparse bridge APIs (`CsrMatrixExtension`, `CsrView`, inbound/outbound CSR paths).
7. Tensor bridge APIs for `arrow.fixed_shape_tensor` and `arrow.variable_shape_tensor`.
8. Explicit null helpers (`fill_nulls_with_zero`, `fill_nulls_with_mean`, `compact_non_null`).
9. Explicit helpers (`cast_f32_to_f64`, `cast_f64_to_f32`, reshape helpers, layout normalization).
10. Explicit sparse densification helper (`densify_csr_view`) with CSR invariant validation.
11. Extension registry APIs (`registered_extension_names`, `deserialize_registered_extension`) and prelude re-exports.
12. Complex scalar, matrix, and fixed-shape tensor bridge APIs built from `ndarrow.complex32` / `ndarrow.complex64`.
13. Integration tests for round-trip correctness and zero-copy pointer guarantees (including sparse/tensor pointer identity checks).
14. Property-test coverage for dense/sparse/tensor round-trip invariants.
15. Benchmark harness with smoke-compatible public API conversion benchmarks plus baseline regression reporting/check gates.
16. Batched sparse matrix bridge APIs (`ndarrow.csr_matrix_batch`, per-row iterator, outbound constructor).
17. Complex ragged tensor bridge APIs over canonical `arrow.variable_shape_tensor`.
18. Full quality gate currently passing (`just checks`, including `90.83%` line coverage).

## Dependencies on Upstream Changes

See `NABLED_CHANGES.md` for detail.

| Change | Crate | Status | Blocking? |
|--------|-------|--------|-----------|
| First-class `f32` support | nabled | Completed and released in nabled `0.0.4` | No |
| `CsrMatrixView` with Arrow-native index types | nabled | Completed and released in nabled `0.0.4` | No |
| View-accepting sparse ops | nabled | Completed and released in nabled `0.0.4` | No |
| Complex Arrow representation assessment | nabled | Out of nabled scope (completed in ndarrow) | No |

## Constraints In Force

1. Zero-copy bridge semantics for view/ownership-transfer paths.
2. Vendor agnosticism (no producer/consumer coupling).
3. ndarray independence (no hard dependency on nabled).
4. Explicit null handling at call sites.
5. Quality gates: `fmt`, `clippy -D warnings -W clippy::pedantic`, coverage >= 90%.

## Next Milestone

1. Publish the next `ndarrow` release from the now-closed checkpoint 1 state.
2. Move upstream to `nabled` checkpoint 2 adoption so downstream custom complex codecs can be removed.
