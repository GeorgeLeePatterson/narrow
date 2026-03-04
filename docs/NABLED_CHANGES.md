# Nabled Changes & Interop Tracker

## Purpose

This document tracks:
1. Changes needed in nabled to improve interop with narrow
2. Temporary structures in narrow that exist as interim solutions
3. The cohesion state between narrow and nabled

**Critical**: None of these changes block narrow's development. narrow uses ndarray directly
and defines its own types. These changes improve end-to-end ergonomics when both crates are
used together.

---

## Required Nabled Changes

### NC-001: First-Class f32 Support

**Status**: Not started
**Priority**: High
**Blocking narrow?**: No

**Description**: nabled's core API is f64-only for real-valued operations. All functions take
`&Array2<f64>`, `&ArrayView2<f64>`, etc. To support Arrow data natively (which is often f32,
especially from embedding models), nabled needs f32 support.

**Options**:
1. **Generic float trait**: Introduce a trait (e.g., `NabledFloat`) that both f32 and f64
   implement. Make all public functions generic over this trait. This is the cleanest solution
   but requires touching every function signature.
2. **Duplicate API surface**: Add `_f32` variants of all functions. Doubles the API surface
   but avoids generics complexity.
3. **Generic wrappers**: Keep f64 implementations, add thin generic wrappers that cast on the
   way in and out. Simple but adds allocation at the boundary.

**Recommendation**: Option 1 (generic float trait) for new code. Migrate existing functions
incrementally.

**Impact on narrow**: Without this, callers must cast f32 -> f64 before calling nabled, then
cast f64 -> f32 to go back to Arrow. narrow provides the cast helper, but the round-trip
allocation is suboptimal.

### NC-002: CsrMatrixView with Arrow-Native Index Types

**Status**: Not started
**Priority**: High
**Blocking narrow?**: No

**Description**: nabled's `CsrMatrix` uses `Vec<usize>` for `row_ptrs` and `col_indices`.
Arrow uses `i32` for List offsets and `u32` for index values. On 64-bit systems, `usize` is
8 bytes while `i32`/`u32` are 4 bytes — the memory layouts are incompatible for zero-copy.

**Options**:
1. **CsrMatrixView type**: Add a new type `CsrMatrixView<'a, I, T>` that borrows `&[I]`
   offsets and index slices. Implement sparse operations for `I = i32, u32` in addition to
   `I = usize`. This preserves the existing CsrMatrix for owned data.
2. **Generic CsrMatrix**: Make `CsrMatrix<I, T>` generic over the index type. More invasive
   but unifies owned and borrowed representations.
3. **Accept usize conversion**: Accept that sparse interop requires an O(nnz) index conversion
   from u32 -> usize. This is simpler but violates the zero-copy principle.

**Recommendation**: Option 1 (CsrMatrixView). It's additive, doesn't break existing API, and
enables zero-copy from Arrow.

**Impact on narrow**: Without this, narrow's `CsrView` holds Arrow-native types (`&[i32]`,
`&[u32]`) but callers must convert to `usize` before passing to nabled. narrow provides the
view; the conversion cost falls on the caller or nabled.

### NC-003: View-Accepting Sparse Operations

**Status**: Not started
**Priority**: Medium
**Blocking narrow?**: No

**Description**: nabled's sparse operations take owned `CsrMatrix`. For zero-copy from Arrow,
operations should also accept borrowed `CsrMatrixView` (from NC-002).

**Options**:
1. Add `_view` variants of sparse operations (consistent with existing `matvec_view` pattern)
2. Make operations generic over `AsRef<CsrData>` or similar trait

**Recommendation**: Option 1 for consistency with nabled's existing patterns.

### NC-004: Complex Type Support Assessment

**Status**: Not started
**Priority**: Low
**Blocking narrow?**: No

**Description**: nabled supports Complex64 and Complex32. Arrow does not have native complex
types (they would need a custom extension type or struct representation). Assess whether
complex type bridging through narrow is needed and what the Arrow representation should be.

**Impact on narrow**: narrow's `NarrowElement` trait currently covers f32 and f64. Complex
types would need a different approach (possibly struct-based Arrow representation).

---

## Interim Artifacts in narrow

These structures exist in narrow as self-contained solutions. They may be simplified or
replaced when nabled changes land.

### IA-001: CsrView Type

**Location**: `src/inbound/sparse.rs` (planned)
**Purpose**: Holds borrowed CSR components with Arrow-native index types

```rust
pub struct CsrView<'a, T> {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptrs: &'a [i32],
    pub col_indices: &'a [u32],
    pub values: &'a [T],
}
```

**Interim because**: When nabled introduces `CsrMatrixView<'a, I, T>` (NC-002), narrow's
`CsrView` could potentially be replaced by or unified with nabled's type. However, narrow
should NOT depend on nabled, so this type may remain as narrow's own representation even after
NC-002 lands.

**Resolution**: Keep `CsrView` as narrow's own type. If nabled's `CsrMatrixView` has the same
layout, callers can transmute or construct one from the other with zero cost. narrow never
imports nabled.

### IA-002: NarrowElement Trait

**Location**: `src/element.rs` (planned)
**Purpose**: Bridges Arrow primitive types and ndarray element types

**Interim because**: If nabled introduces a generic float trait (NC-001), there may be overlap.
However, narrow's trait bridges Arrow and ndarray, not ndarray and linalg. The traits serve
different purposes and should remain separate.

**Resolution**: Keep as permanent. NarrowElement is narrow's own abstraction. It may share
implementations with nabled's float trait but is defined independently.

---

## Qdrant-DataFusion Changes

These changes are needed in qdrant-datafusion to leverage narrow effectively. They are tracked
here for completeness but are not narrow's responsibility.

### QD-001: Dense Vectors as FixedSizeList

**Description**: Change dense vector columns from `List<Float32>` to `FixedSizeList<Float32>(D)`
where D is the vector dimension from collection config.

**Rationale**: FixedSizeList enables zero-copy to ArrayView2. List does not.

### QD-002: Multi-Vectors as List<FixedSizeList>

**Description**: Change multi-vector columns from `List<List<Float32>>` to
`List<FixedSizeList<Float32>(D)>`.

**Rationale**: Inner FixedSizeList ensures each embedding has fixed dimension, enabling
zero-copy per-point ArrayView2.

### QD-003: Sparse Vectors as narrow.csr_matrix

**Description**: Consider using narrow's CSR extension type instead of two separate columns
for sparse vectors.

**Rationale**: Self-describing column with ncols metadata. Single column per sparse vector
field instead of two.

### QD-004: Evaluate FixedShapeTensor / VariableShapeTensor

**Description**: Consider using canonical tensor extension types for vector and multi-vector
columns, for maximum cross-language interop.

**Rationale**: pyarrow and other Arrow implementations recognize these types. Enables
interop with Python ML pipelines.

---

## Cohesion State

| Aspect                  | narrow State       | nabled State       | Cohesion |
|-------------------------|--------------------|--------------------|----------|
| f32 support             | Planned (NarrowElement for f32) | f64 only  | Gap — NC-001 |
| f64 support             | Planned (NarrowElement for f64) | Full      | Aligned  |
| Dense views             | Planned (ArrayView1/2)         | Accepts views | Aligned  |
| Sparse views            | Planned (CsrView<i32,u32>)     | CsrMatrix<usize> | Gap — NC-002 |
| Tensor views            | Planned (ArrayViewD)           | Accepts ArrayD | Aligned  |
| Owned array transfer    | Planned (IntoArrow)            | Returns owned  | Aligned  |
| Complex types           | Not planned yet                 | Complex64/32   | Gap — NC-004 |

**Overall**: narrow and nabled are architecturally compatible. The ndarray type system is the
common language. Gaps exist in f32 support and sparse index types, both addressable with
additive nabled changes. No narrow design decisions are blocked by nabled's current state.
