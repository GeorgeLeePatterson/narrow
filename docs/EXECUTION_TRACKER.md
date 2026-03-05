# Execution Tracker

Last updated: 2026-03-05

## Purpose

Operational state tracker for ndarrow implementation progress.

## Resume Protocol

1. Read `docs/README.md`.
2. Read `docs/DECISIONS.md` (locked constraints).
3. Read this file for current done/next state.
4. Read `docs/architecture.md` for structural work.

## Done

| ID    | Description | Phase |
|-------|-------------|-------|
| D-001 | Planning docs and architectural decisions completed | Pre |
| D-002 | AGENTS/quality-gate baseline established | Pre |
| I-001 | Workspace dependencies and lint config wired | 1 |
| I-002 | `NdarrowError` implemented | 1 |
| I-003 | `NdarrowElement` implemented (`f32`/`f64`) | 1 |
| I-004 | `AsNdarray` trait implemented for primitive inbound path | 2 |
| I-005 | `FixedSizeList -> ArrayView2` inbound conversions implemented | 2 |
| I-006 | `IntoArrow` trait implemented for `Array1`/`Array2` outbound path | 3 |
| I-007 | Explicit null handling tiers implemented (`validated`/`unchecked`/`masked`) | 4 |
| I-008 | Dense round-trip integration tests added | 9 |
| I-009 | Zero-copy pointer verification tests added | 9 |
| I-010 | Null handling integration tests added | 9 |
| I-011 | CI pipeline and coverage gate added | 10 |
| I-012 | Public API benchmark suites added for inbound/outbound conversions | 9 |
| I-013 | Benchmark smoke command aligned to quick smoke convention | 9 |
| I-014 | nabled interop tracker updated with verified NC-001/NC-002/NC-003 completion | 10 |
| I-015 | `ndarrow.csr_matrix` extension type + `CsrView` + sparse inbound/outbound APIs implemented | 5 |
| I-016 | Fixed-shape tensor inbound/outbound APIs implemented | 6 |
| I-017 | Variable-shape tensor inbound iterator + outbound packing implemented | 7 |
| I-018 | Explicit helper APIs implemented (`cast`, `reshape`, `to_standard_layout`) | 8 |
| I-019 | Coverage restored to passing threshold after sparse/tensor additions (>= 90%) | 10 |
| I-020 | Capability/tracker/status/nabled docs synchronized to implemented state | 10 |
| I-021 | `densify_csr_view` helper implemented with CSR invariant validation and helper tests | 8 |
| I-022 | Property-based integration tests added for dense/sparse/tensor round-trip invariants | 9 |
| I-023 | Benchmark baseline persistence/reporting refinements implemented (`bench_report.sh`, CI summary + threshold checks) | 9 |
| I-024 | Sparse/tensor allocation verification expanded in integration tests (pointer-identity zero-copy checks) | 9 |
| I-025 | `cast_f64_to_f32` finite-range failure semantics hardened with explicit overflow rejection tests | 8 |
| I-026 | Null helper APIs completed (`fill_nulls_with_zero`, `fill_nulls_with_mean`, `compact_non_null`) | 8 |
| I-027 | Extension registry and crate prelude completed (`extensions`, `prelude`) | 10 |
| I-028 | Complex extension bridge completed (`ndarrow.complex32`/`ndarrow.complex64` inbound/outbound + tests) | 10 |

## Next (Priority Order)

No open in-scope implementation items remain for the current milestone.

## Needed (Open Questions)

No open questions remain for the current milestone.
