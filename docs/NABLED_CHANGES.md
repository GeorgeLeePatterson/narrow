# Nabled Changes & Interop Tracker

Last updated: 2026-03-05

## Purpose

This document tracks:

1. Interop-relevant upstream work in nabled.
2. What is complete versus what remains for ndarrow.
3. Cohesion state between ndarrow and nabled.

Critical reminder: ndarrow does **not** depend on nabled. Upstream completion improves end-to-end
ergonomics when both crates are used together, but does not block ndarrow implementation.

## Required Nabled Changes

### NC-001: First-Class f32 Support

**Status**: Completed and released in nabled `0.0.4`  
**Priority**: High  
**Blocking ndarrow?**: No

**Verification snapshot**:

1. `NabledReal` trait includes both `f32` and `f64`.
2. Real API surface across core/linalg/ml is generic over `NabledReal`.
3. `f32` parity tests exist across major domains.

### NC-002: CsrMatrixView with Arrow-Native Index Types

**Status**: Completed and released in nabled `0.0.4`  
**Priority**: High  
**Blocking ndarrow?**: No

**Verification snapshot**:

1. `CsrIndex` supports `usize`, `u32`, and `i32`.
2. `CsrMatrixView<'a, R, T, C>` is first-class and validated.
3. Mixed `i32`/`u32` view parity tests exist.

### NC-003: View-Accepting Sparse Operations

**Status**: Completed and released in nabled `0.0.4`  
**Priority**: Medium  
**Blocking ndarrow?**: No

**Verification snapshot**:

1. Sparse APIs provide owned and `_view` variants.
2. Owned entrypoints delegate through view-native paths where appropriate.
3. Parity coverage exists for owned-vs-view behavior.

### NC-004: Complex Type Support Assessment

**Status**: Out of scope for nabled (tracked on ndarrow side)  
**Priority**: Low  
**Blocking ndarrow?**: No

This item concerns Arrow representation decisions for complex values. nabled does support complex
numerics, but Arrow-bridge representation choices belong to ndarrow architecture.

## Interim Artifacts in ndarrow

### IA-001: CsrView Type (implemented)

ndarrow keeps its own Arrow-native CSR view abstraction (`CsrView`). Even though nabled has
`CsrMatrixView`, ndarrow remains nabled-independent by design.

### IA-002: NdarrowElement Trait (implemented)

`NdarrowElement` remains ndarrow-owned. It bridges Arrow primitive typing and ndarray element
typing and is intentionally separate from nabled scalar traits.

## Cohesion State

| Aspect | ndarrow state | nabled state | Cohesion |
|--------|---------------|--------------|----------|
| `f32` support | Implemented (`NdarrowElement`, dense conversions/tests) | Released in `0.0.4` | Aligned |
| `f64` support | Implemented | Completed | Aligned |
| Dense views | Implemented (`AsNdarray`, `FixedSizeList -> ArrayView2`) | Accepts ndarray views | Aligned |
| Sparse views (`i32`/`u32`) | Implemented (`CsrView`, extension + column paths) | `CsrMatrixView` completed | Aligned |
| Sparse view ops | Implemented bridge-side view extraction | `_view` sparse APIs completed | Aligned |
| Complex Arrow bridge | Not yet defined | N/A (out of scope) | Open design item (ndarrow) |

Overall: upstream nabled interop tasks are complete and released (`0.0.4`); ndarrow can proceed
independently while interoperating with the published nabled baseline.
