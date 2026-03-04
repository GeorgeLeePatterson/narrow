# Locked Architectural Decisions

Decisions here are final. They may only be revisited with explicit justification and a new
decision entry that supersedes the old one.

## Foundation

### D-001: Zero-Copy Bridge Contract

narrow never allocates on the bridge path. Arrow -> ndarray produces views. ndarray -> Arrow
transfers buffer ownership. The bridge is O(1) in time and O(0) in additional memory.

**Rationale**: The entire purpose of narrow is to avoid the cost of conversion. If the bridge
allocates, callers might as well copy manually.

### D-002: ndarray is the Numerical Substrate

narrow maps Arrow types to ndarray types. Not nalgebra, not faer, not raw slices. ndarray is the
single target because it is the substrate for nabled and provides the view/owned/stride model
needed for zero-copy.

**Rationale**: Consistency with nabled's ndarray-first architecture. One substrate, one set of
invariants to maintain.

### D-003: Vendor Agnosticism

narrow has zero knowledge of any Arrow producer or consumer. It does not import qdrant-client,
datafusion, polars, or any domain crate. It bridges Arrow and ndarray, nothing more.

**Rationale**: narrow is a foundational library. Domain assumptions reduce reusability and create
coupling that complicates maintenance.

### D-004: No nabled Dependency

narrow depends on `arrow` and `ndarray`. It does not depend on nabled. If nabled needs view types
or index generics that narrow would benefit from, those changes happen in nabled. narrow's public
API exposes only Arrow and ndarray types.

**Rationale**: narrow must be usable by any ndarray consumer, not just nabled. Coupling to nabled
would violate vendor agnosticism for the ndarray side.

### D-005: Algebraic, Compositional, Homomorphic, Denotationally Sound

All abstractions follow these four principles. Types form algebras. Small conversions compose into
larger ones. The bridge preserves structure (Arrow shapes map to ndarray shapes). Every function
has an unambiguous mathematical denotation.

**Rationale**: These principles are the author's universal design requirements. They ensure
correctness, composability, and long-term maintainability.

## Type Mappings

### D-010: Dense Vectors — FixedSizeList

A column of fixed-dimension dense vectors is represented as `FixedSizeList<T>(D)` in Arrow, where
D is the vector dimension. This maps to `ArrayView2<T>` of shape `(M, D)` where M is the number
of rows.

**Rationale**: FixedSizeList stores values as a single contiguous buffer without an offsets array.
The flat buffer IS the row-major M x D matrix. Zero-copy to ArrayView2.

**Supersedes**: Variable-length `List<T>` for fixed-dimension vectors. `List` carries unnecessary
offsets and does not guarantee the contiguous layout needed for zero-copy matrix interpretation.

### D-011: Scalar Arrays — PrimitiveArray

A column of scalar values is `PrimitiveArray<T>` in Arrow, mapping to `ArrayView1<T>` of shape
`(M,)`. This is the trivial case — Arrow's values buffer is directly borrowable as a slice.

**Rationale**: PrimitiveArray is Arrow's native representation for scalars. The values buffer is
a contiguous `&[T]`. ndarray's ArrayView1 wraps a slice.

### D-012: Tensors — Canonical arrow.fixed_shape_tensor

A column where each row is a fixed-shape tensor (matrix, cube, or higher) uses the canonical
Arrow extension type `arrow.fixed_shape_tensor`. Storage type is `FixedSizeList<T>(product_of_dims)`.
Metadata carries `shape`, optional `dim_names`, and optional `permutation`. Elements are stored
in row-major (C-contiguous) order.

This maps to `ArrayView<T, D>` where D is determined by the tensor rank plus one (for the batch
dimension). For arbitrary rank, `ArrayViewD<T>` is used.

**Rationale**: This is a canonical Arrow extension type, already implemented in arrow-rs and
recognized by pyarrow. Using it gives cross-language interop for free. The memory layout (flat
C-contiguous buffer + shape metadata) is exactly what ndarray needs.

### D-013: Variable-Shape Tensors — Canonical arrow.variable_shape_tensor

A column where each row is a tensor whose shape varies per row (e.g., ColBERT multi-vectors with
varying token counts) uses the canonical `arrow.variable_shape_tensor`. Storage is
`StructArray{data: List<T>, shape: FixedSizeList<int32>(ndim)}`. Metadata supports `uniform_shape`
for dimensions that are fixed (e.g., embedding dimension) while others vary (e.g., token count).

Per-row access yields `ArrayView<T, D>` for each element. Batch access is not possible as a
single contiguous ndarray (the data is ragged).

**Rationale**: Canonical Arrow extension type. Handles the ragged case (e.g., multi-vectors)
without inventing a custom type. Cross-language interop with pyarrow.

### D-014: Sparse Vectors — narrow.csr_matrix Extension Type

A column of sparse vectors uses a narrow-defined extension type `narrow.csr_matrix`. Storage is
`StructArray{indices: List<UInt32>, values: List<T>}`. Metadata carries `ncols` (the dimension
of the sparse vector space).

The List offsets buffer serves as CSR `row_ptrs`. The indices values buffer serves as CSR
`col_indices`. The values values buffer serves as CSR `values`. Both List columns must have
identical offsets (same sparsity pattern per row).

This maps to a `CsrMatrixView` or equivalent ndarray-compatible sparse view.

**Rationale**: No canonical Arrow sparse extension type exists. The two-column representation
(separate `indices` and `values` columns) is also supported as a convenience, but the extension
type packages the sparse representation as a single self-describing column. CSR is chosen because
Arrow's List offsets are structurally identical to CSR row pointers.

### D-015: Both f32 and f64 are First-Class

narrow supports both `f32` and `f64` as first-class element types. The trait bridge (`NarrowElement`
or equivalent) has concrete implementations for both. Neither is preferred or defaulted.

Additional types (complex, integer, f16) may be added. The trait system is extensible.

**Rationale**: Different producers use different types. Qdrant uses f32. Scientific computing
often uses f64. narrow is vendor-agnostic and must not assume a type.

## Null Handling

### D-020: Three Null Tiers

Null handling is explicit at the call site via three API tiers:

1. **Unchecked** (`_unchecked`): Caller guarantees no nulls. Zero cost. UB if nulls present
   (or assertion in debug mode).
2. **Validated** (default): Checks `null_count() == 0`, returns `Err` if nulls present. O(1).
3. **Masked** (`_masked`): Returns the view alongside the validity bitmap. Caller decides how
   to handle nulls. Zero allocation.

**Rationale**: Nulls in Arrow are orthogonal to the data buffer. The validity bitmap exists
separately. The view of the data buffer is always valid regardless of nulls — the question is
whether the caller wants to acknowledge them. Making this explicit avoids hidden allocation
(filtering) or hidden semantics (NaN substitution).

### D-021: Null Helpers are Opt-In Allocation Points

Functions like `fill_nulls(strategy)` or `compact_non_null()` are explicit helpers that allocate.
They are not part of the bridge path. They exist in a `helpers` or `nulls` module and their
allocation cost is visible at the call site.

**Rationale**: Null handling strategies (fill with zero, fill with mean, drop rows) are
domain-specific. narrow provides the tools but does not choose a strategy.

## Ownership and Lifetime

### D-030: Inbound is Borrowing

Arrow -> ndarray conversions borrow the Arrow buffer and produce ndarray views. The view's
lifetime is tied to the Arrow array's lifetime. No ownership transfer occurs.

**Rationale**: Arrow arrays own their buffers (via Arc). ndarray views borrow slices. This is
the natural zero-copy mapping.

### D-031: Outbound is Ownership Transfer

ndarray -> Arrow conversions consume the owned ndarray array and transfer the underlying buffer
to Arrow. The mechanism is: `Array::into_raw_vec()` -> `ScalarBuffer::from(vec)` ->
`PrimitiveArray::new(buffer, None)`. No copy occurs if the ndarray is in standard (C-contiguous)
layout.

**Rationale**: Owned ndarray arrays hold a `Vec<T>`. Arrow's `ScalarBuffer::from(Vec<T>)` takes
ownership without copying. The buffer nabled (or any computation) allocated IS the Arrow buffer.

### D-032: Non-Standard Layout May Allocate

If an owned ndarray is not in standard C-contiguous layout (e.g., it was transposed or sliced),
`into_arrow` must first call `as_standard_layout().into_owned()`, which allocates. This is
documented and the allocation is inherent — there is no way to represent non-contiguous memory
as a single Arrow buffer.

**Rationale**: Arrow buffers are contiguous by definition. Non-contiguous ndarray layouts cannot
be transferred without copying. The allocation is unavoidable and documented.

## Extension Types

### D-040: Canonical Types First

Where a canonical Arrow extension type exists that serves narrow's purpose, narrow uses it.
narrow does not define custom extension types that duplicate canonical ones.

Currently used canonical types:
- `arrow.fixed_shape_tensor` — fixed-shape tensors (matrices, cubes, higher-rank)
- `arrow.variable_shape_tensor` — variable-shape tensors (multi-vectors, ragged)

**Rationale**: Canonical types provide cross-language interop, spec compliance, and ecosystem
recognition without additional work.

### D-041: Custom Types for Gaps

Where no canonical type exists, narrow defines extension types under the `narrow.` namespace.
Currently defined:
- `narrow.csr_matrix` — CSR sparse matrix

Custom types implement the `ExtensionType` trait from `arrow_schema::extension` with proper
serialization, deserialization, and validation.

**Rationale**: Sparse matrices have no canonical Arrow representation. A proper extension type
makes sparse columns self-describing and enables cross-language handler registration.

## API Design

### D-050: Trait-Based Bridge

The bridge is defined by traits, not standalone functions. Primary traits:
- `AsNdarray` — Arrow array -> ndarray view (inbound, borrowing)
- `IntoArrow` — ndarray owned array -> Arrow array (outbound, ownership transfer)
- `NarrowElement` — bridges Arrow primitive types and ndarray element types

**Rationale**: Traits make the bridge extensible. New Arrow types or ndarray layouts can be
supported by implementing the traits. Traits also enable generic code bounded by the bridge
capability.

### D-051: Composability over Convenience

Column-level conversions are the primitives. RecordBatch-level or multi-column operations
compose from column-level primitives. Convenience wrappers may exist but never bypass the
column-level layer.

**Rationale**: Composability ensures that the library is useful in contexts we haven't
anticipated. Monolithic helpers would embed assumptions about schema structure.

### D-052: Helpers are Explicit Allocation Points

Functions that allocate (type casting, sparse densification, null filling, layout normalization)
are clearly separated from the zero-copy bridge. They live in a `helpers` module (or equivalent),
and their names or types make the allocation cost obvious.

**Rationale**: Callers must be able to distinguish zero-cost operations from allocating ones
at a glance. Mixing them would undermine narrow's performance guarantees.

## Quality

### D-060: Test Coverage >= 90%

Line coverage must be >= 90% as measured by `cargo llvm-cov`. No exceptions.

**Rationale**: Every code path — including error paths, edge cases, and unsafe blocks — must be
exercised. A bridge library's correctness is its only value.

### D-061: Clippy -D warnings

All clippy warnings are errors. No `#[allow]` without a justifying comment.

### D-062: Documentation on Every Public Item

Every public function, trait, type, and module has a doc comment. Doc comments state what the
function does, whether it allocates, and its safety invariants (if unsafe).
