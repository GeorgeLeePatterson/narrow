# Roadmap — Implementation Plan

## Phase Overview

| Phase | Name                        | Scope                                               |
|-------|-----------------------------|------------------------------------------------------|
| 1     | Foundation                  | Crate skeleton, traits, error types, element bridge  |
| 2     | Dense Inbound               | PrimitiveArray -> ArrayView1, FixedSizeList -> ArrayView2 |
| 3     | Dense Outbound              | Array1 -> PrimitiveArray, Array2 -> FixedSizeList    |
| 4     | Null Handling               | Three-tier null API (unchecked, validated, masked)   |
| 5     | Sparse                      | narrow.csr_matrix ext type, CsrView, two-column API |
| 6     | Tensor                      | FixedShapeTensor support, ArrayViewD                 |
| 7     | Variable Tensor             | VariableShapeTensor support, per-row views           |
| 8     | Helpers                     | Cast, densify, reshape, layout normalization         |
| 9     | Property Tests & Benchmarks | Round-trip correctness, conversion overhead measurement |
| 10    | Production Hardening        | Docs polish, CI, publish prep                        |

---

## Phase 1: Foundation

**Goal**: Establish the crate structure, core traits, and error taxonomy. After this phase,
the crate compiles, has its module tree, and the trait contracts are defined (even if not yet
implemented).

### Tasks

1. **Update Cargo.toml**: Add dependencies (`arrow`, `ndarray`), configure features, set
   edition and metadata.

2. **Create error.rs**: Define `NarrowError` enum with all variants (NullsPresent, TypeMismatch,
   ShapeMismatch, InvalidMetadata, InnerTypeMismatch, NonStandardLayout, SparseOffsetMismatch,
   Arrow). Implement `From<ArrowError>`, `Display`, `Error`.

3. **Create element.rs**: Define `NarrowElement` trait with `ArrowType` associated type.
   Implement for `f32` and `f64`. This is the compile-time bridge between Arrow's type system
   and ndarray's element requirements.

4. **Create trait stubs**: Define `AsNdarray` trait (with `as_ndarray`, `as_ndarray_unchecked`,
   `as_ndarray_masked` methods) and `IntoArrow` trait (with `into_arrow` method). No
   implementations yet.

5. **Create module tree**: Set up `inbound/`, `outbound/`, `ext/`, `helpers/`, `nulls.rs`
   modules with placeholder `mod.rs` files.

6. **Create lib.rs**: Wire modules, define prelude, set up re-exports.

7. **Write initial tests**: Verify crate compiles, trait definitions are coherent, element
   trait implementations exist for f32/f64.

### Deliverables

- Compiling crate with full module tree
- `NarrowElement` trait with f32, f64 impls
- `AsNdarray`, `IntoArrow` trait definitions
- `NarrowError` enum
- Tests for trait coherence

---

## Phase 2: Dense Inbound

**Goal**: Zero-copy conversion from Arrow dense arrays to ndarray views.

### Tasks

1. **PrimitiveArray -> ArrayView1**: Implement `AsNdarray` for `PrimitiveArray<T>` where
   `T::Native: NarrowElement`. The implementation borrows `self.values().as_ref()` and wraps
   it in `ArrayView1::from(slice)`. Null handling follows the three-tier API.

2. **FixedSizeList -> ArrayView2**: Implement `AsNdarray` for `FixedSizeListArray`. Validate
   that the inner array is a `PrimitiveArray<T>` where `T::Native: NarrowElement`. Extract
   the inner values slice, compute shape `(num_rows, value_length)`, and construct
   `ArrayView2::from_shape(shape, slice)`. Null handling for both outer (row-level) and inner
   (component-level) validity.

3. **Generic dispatch**: Since `FixedSizeListArray` is not generic over element type in arrow-rs,
   implement type-dispatched conversion (match on inner DataType, downcast, convert). Provide
   both type-dispatched API and generic API where the caller knows the type.

4. **Tests**:
   - Round-trip: create Arrow array, convert to view, verify values match
   - Empty arrays (zero rows)
   - Large arrays (verify no allocation via custom allocator or timing)
   - Null handling: arrays with nulls trigger Err on validated path
   - Shape verification for FixedSizeList (M rows x N values)

### Deliverables

- `AsNdarray` impl for `PrimitiveArray<Float32Type>`, `PrimitiveArray<Float64Type>`
- `AsNdarray` impl for `FixedSizeListArray` (f32 and f64 inner types)
- Three null tiers working for both
- Tests with >= 90% coverage of inbound module

### Key Implementation Detail

```rust
// PrimitiveArray<T> -> ArrayView1<T::Native>
impl<T: ArrowPrimitiveType> AsNdarray for PrimitiveArray<T>
where
    T::Native: NarrowElement,
{
    type View<'a> = ArrayView1<'a, T::Native>;

    fn as_ndarray(&self) -> Result<Self::View<'_>, NarrowError> {
        if self.null_count() > 0 {
            return Err(NarrowError::NullsPresent { null_count: self.null_count() });
        }
        // values() returns &[T::Native], which is the contiguous buffer.
        Ok(ArrayView1::from(self.values().as_ref()))
    }
}

// FixedSizeListArray -> ArrayView2<T>
// This requires knowing T at the call site or using type dispatch.
fn fixed_size_list_as_array2<T: ArrowPrimitiveType>(
    array: &FixedSizeListArray,
) -> Result<ArrayView2<'_, T::Native>, NarrowError>
where
    T::Native: NarrowElement,
{
    if array.null_count() > 0 {
        return Err(NarrowError::NullsPresent { null_count: array.null_count() });
    }
    let values = array.values().as_primitive::<T>();
    if values.null_count() > 0 {
        return Err(NarrowError::NullsPresent { null_count: values.null_count() });
    }
    let n = array.value_length() as usize;
    let m = array.len();
    let slice: &[T::Native] = values.values().as_ref();
    ArrayView2::from_shape((m, n), slice)
        .map_err(|e| NarrowError::ShapeMismatch {
            expected: vec![m, n],
            found: vec![slice.len()],
        })
}
```

---

## Phase 3: Dense Outbound

**Goal**: Zero-copy ownership transfer from owned ndarray arrays to Arrow arrays.

### Tasks

1. **Array1 -> PrimitiveArray**: Implement `IntoArrow` for `Array1<T>` where `T: NarrowElement`.
   The implementation calls `self.into_raw_vec()` to extract the owned `Vec<T>`, constructs
   `ScalarBuffer::from(vec)`, and creates `PrimitiveArray::new(buffer, None)`.

2. **Array2 -> FixedSizeList**: Implement `IntoArrow` for `Array2<T>`. Check if standard layout
   (`is_standard_layout()`). If yes, `into_raw_vec()` gives the flat row-major buffer. Construct
   `PrimitiveArray` from the buffer, then wrap in `FixedSizeListArray::new(field, N, values, None)`
   where N is the column count.

3. **Non-standard layout handling**: If `!is_standard_layout()`, call
   `as_standard_layout().into_owned()` first (this allocates). Document this clearly. Consider
   returning `Err(NonStandardLayout)` in a strict variant, with a separate `into_arrow_force`
   that accepts the allocation.

4. **Tests**:
   - Round-trip: create ndarray, convert to Arrow, convert back to view, verify values
   - Verify no allocation for standard-layout arrays (timing or allocator introspection)
   - Non-standard layout: transposed Array2 triggers allocation or error
   - Shape preservation: Array2 of (M, N) produces FixedSizeList of length M with value_length N

### Deliverables

- `IntoArrow` impl for `Array1<f32>`, `Array1<f64>`
- `IntoArrow` impl for `Array2<f32>`, `Array2<f64>`
- Layout detection and documented allocation behavior
- Tests with >= 90% coverage of outbound module

### Key Implementation Detail

```rust
impl<T: NarrowElement> IntoArrow for Array1<T>
where
    ScalarBuffer<T>: From<Vec<T>>,
{
    type ArrowArray = PrimitiveArray<T::ArrowType>;

    fn into_arrow(self) -> Result<Self::ArrowArray, NarrowError> {
        let vec = self.into_raw_vec();
        let buffer = ScalarBuffer::from(vec);
        Ok(PrimitiveArray::new(buffer, None))
    }
}

impl<T: NarrowElement> IntoArrow for Array2<T>
where
    ScalarBuffer<T>: From<Vec<T>>,
{
    type ArrowArray = FixedSizeListArray;

    fn into_arrow(self) -> Result<Self::ArrowArray, NarrowError> {
        let (m, n) = self.dim();
        let array = if self.is_standard_layout() {
            self
        } else {
            self.as_standard_layout().into_owned()
        };
        let vec = array.into_raw_vec();
        let buffer = ScalarBuffer::from(vec);
        let values = PrimitiveArray::<T::ArrowType>::new(buffer, None);
        let field = Arc::new(Field::new("item", T::ArrowType::DATA_TYPE, false));
        Ok(FixedSizeListArray::new(field, n as i32, Arc::new(values), None))
    }
}
```

---

## Phase 4: Null Handling

**Goal**: Implement the three-tier null API and null helper functions.

### Tasks

1. **Validated tier**: Already partially done in Phase 2/3 (null_count checks). Ensure consistent
   error messages and behavior across all `AsNdarray` impls.

2. **Unchecked tier**: Add `unsafe fn as_ndarray_unchecked` to all `AsNdarray` impls. These skip
   the null check entirely. Add debug_assert for null_count in debug builds.

3. **Masked tier**: Add `fn as_ndarray_masked` returning `(View, Option<&BooleanBuffer>)`. The
   view is always produced (data buffer is valid regardless of nulls). The bitmap is `None` if
   `null_count() == 0`, otherwise `Some(array.nulls().inner())`.

4. **Null helpers**: Implement in `nulls.rs` or `helpers/`:
   - `fill_nulls_with_zero<T>(array) -> PrimitiveArray<T>` — allocates new array
   - `fill_nulls_with_value<T>(array, value) -> PrimitiveArray<T>` — allocates
   - `compact_non_null<T>(array) -> PrimitiveArray<T>` — allocates, removes null positions

5. **Tests**: All three tiers for all array types. Null helpers correctness.

### Deliverables

- Three-tier null API on all AsNdarray impls
- Null helper functions (allocating, clearly documented)
- Tests

---

## Phase 5: Sparse

**Goal**: Zero-copy sparse representation via the narrow.csr_matrix extension type and two-column
convenience API.

### Tasks

1. **Define narrow.csr_matrix extension type**: Implement `ExtensionType` trait in `ext/csr_matrix.rs`.
   - `NAME = "narrow.csr_matrix"`
   - `Metadata` struct: `CsrMatrixMetadata { ncols: usize }`
   - Storage type: `StructArray{indices: List<UInt32>, values: List<T>}`
   - `serialize_metadata`: JSON `{"ncols": N}`
   - `deserialize_metadata`: Parse JSON
   - `supports_data_type`: Validate struct with two List fields

2. **CsrView type**: Define a view type that holds borrowed CSR components:
   ```rust
   pub struct CsrView<'a, T> {
       pub nrows: usize,
       pub ncols: usize,
       pub row_ptrs: &'a [i32],      // Arrow List offsets
       pub col_indices: &'a [u32],   // Arrow UInt32 values
       pub values: &'a [T],          // Arrow T values
   }
   ```
   This type is narrow's sparse view. It uses Arrow's native index types (i32 offsets, u32 indices)
   to avoid conversion. Consumers (like nabled) that need `usize` indices convert on their side.

3. **Inbound from extension type**: Given a `StructArray` tagged with `narrow.csr_matrix`,
   extract the two List fields, borrow their offsets and values buffers, construct `CsrView`.

4. **Inbound from two columns**: Convenience API that takes separate `ListArray<UInt32>` (indices)
   and `ListArray<T>` (values) plus `ncols: usize`, validates matching offsets, constructs
   `CsrView`.

5. **Outbound**: Given owned CSR data (`Vec<i32>` row_ptrs, `Vec<u32>` col_indices, `Vec<T>` values),
   construct the StructArray with narrow.csr_matrix extension type. Transfer ownership of all
   three vecs.

6. **Tests**:
   - Round-trip: create sparse data, construct extension type, extract CsrView, verify
   - Offset validation: mismatched offsets between indices and values columns
   - Empty sparse arrays
   - Metadata serialization/deserialization

### Deliverables

- `narrow.csr_matrix` extension type implementation
- `CsrView` type
- Inbound from extension type and two-column convenience
- Outbound ownership transfer
- Tests

---

## Phase 6: Tensor (FixedShapeTensor)

**Goal**: Support the canonical `arrow.fixed_shape_tensor` extension type for multi-dimensional
data.

### Tasks

1. **Inbound**: Given a `FixedSizeListArray` tagged with `arrow.fixed_shape_tensor`, extract
   shape from metadata, validate storage type, construct `ArrayViewD<T>` with shape
   `[batch_size, ...tensor_shape]`.

2. **Outbound**: Given an owned `ArrayD<T>`, extract shape (excluding batch dimension),
   construct `FixedSizeListArray` with tensor extension metadata. Transfer ownership.

3. **Convenience for known ranks**: Provide typed variants for common cases:
   - `as_array_view3` for rank-3 tensors (batch of matrices)
   - These avoid the dynamic dimensionality of `ArrayViewD`

4. **Tests**:
   - Round-trip: create tensor data, convert to ArrayViewD, verify shape and values
   - Various ranks (1D through 5D)
   - Dimension names and permutation metadata

### Deliverables

- FixedShapeTensor inbound (ArrayViewD)
- FixedShapeTensor outbound (ArrayD -> extension type)
- Convenience methods for common ranks
- Tests

---

## Phase 7: Variable Tensor (VariableShapeTensor)

**Goal**: Support the canonical `arrow.variable_shape_tensor` for ragged data like multi-vectors.

### Tasks

1. **Inbound**: Given a `StructArray` tagged with `arrow.variable_shape_tensor`, provide an
   iterator that yields `(usize, ArrayViewD<T>)` tuples (index + view) for each element.
   Each element's view is zero-copy into the data buffer at the appropriate offset.

2. **uniform_shape support**: When `uniform_shape` metadata specifies fixed dimensions (e.g.,
   `[null, 128]` for variable-count fixed-dim vectors), validate that all elements conform.

3. **Outbound**: Given a `Vec<ArrayD<T>>` or iterator of owned arrays, construct the
   VariableShapeTensor struct array. This necessarily allocates (packing ragged data into
   Arrow's layout).

4. **Tests**:
   - Multi-vector scenario: variable number of 128-dim vectors per row
   - Single-element access: verify zero-copy view
   - Shape validation with uniform_shape metadata

### Deliverables

- VariableShapeTensor inbound iterator
- VariableShapeTensor outbound construction
- uniform_shape validation
- Tests

---

## Phase 8: Helpers

**Goal**: Implement explicit allocation helpers for type casting, densification, reshaping,
and layout normalization.

### Tasks

1. **cast**: `PrimitiveArray<T>` -> `PrimitiveArray<U>` element-wise type conversion. Uses
   Arrow's compute kernels where available, falls back to manual conversion.

2. **densify**: `narrow.csr_matrix` (or CsrView) -> `FixedSizeListArray`. Allocates a dense
   buffer, fills from sparse representation. Explicit allocation.

3. **reshape**: `PrimitiveArray<T>` -> `ArrayView2<T>` or `ArrayViewD<T>` with caller-specified
   shape. Zero-copy (just pointer + shape). Validates that `product(shape) == len`.

4. **to_standard_layout**: `ArrayD<T>` -> `ArrayD<T>` in C-contiguous order. No-op if already
   standard layout. Allocates only if needed.

5. **Tests**: All helpers, including edge cases (empty arrays, single-element, type overflow
   on narrowing cast).

### Deliverables

- `cast`, `densify`, `reshape`, `to_standard_layout` helpers
- Clear documentation of allocation behavior
- Tests

---

## Phase 9: Property Tests & Benchmarks

**Goal**: Verify correctness via property-based testing and measure conversion overhead.

### Tasks

1. **Property tests** (using proptest or quickcheck):
   - Round-trip: Arrow -> ndarray -> Arrow preserves all values
   - Round-trip: ndarray -> Arrow -> ndarray preserves all values
   - Shape preservation across all conversions
   - Null count preservation in masked tier
   - Sparse round-trip preserves sparsity pattern

2. **Benchmarks** (using criterion):
   - Inbound conversion latency (should be ~nanoseconds, O(1))
   - Outbound conversion latency (should be ~nanoseconds for standard layout)
   - Compare against naive copy baseline to quantify savings
   - Varying sizes: 100, 10K, 1M, 100M elements
   - Sparse: varying sparsity ratios

3. **Allocation verification**: Use `dhat` or custom global allocator to verify zero allocations
   on the bridge path.

### Deliverables

- Property test suite
- Benchmark suite with criterion
- Allocation verification tests
- Documented results

---

## Phase 10: Production Hardening

**Goal**: Polish documentation, set up CI, prepare for publication.

### Tasks

1. **Documentation**: Ensure every public item has doc comments. Write crate-level documentation
   with examples. Write module-level documentation.

2. **CI**: GitHub Actions workflow: fmt, clippy, test, coverage threshold, benchmarks.

3. **Publish checklist**: Version, changelog, cargo package, test all features, publish.

4. **README**: Comprehensive README with usage examples, type mapping table, performance
   guarantees.

### Deliverables

- Complete documentation
- CI pipeline
- Published crate

---

## Dependency Chart

```
Phase 1 (Foundation)
  ├── Phase 2 (Dense Inbound)
  │     └── Phase 4 (Null Handling) ─── depends on Phase 2 impls
  ├── Phase 3 (Dense Outbound)
  ├── Phase 5 (Sparse) ─── depends on Phase 1 traits + ext/ module
  ├── Phase 6 (Tensor) ─── depends on Phase 2 inbound patterns
  └── Phase 7 (Variable Tensor) ─── depends on Phase 6

Phase 8 (Helpers) ─── depends on Phases 2, 3, 5

Phase 9 (Property Tests & Benchmarks) ─── depends on all implementation phases

Phase 10 (Production Hardening) ─── depends on everything
```

Phases 2, 3, 5, and 6 can be developed in parallel after Phase 1 is complete.
Phase 4 depends on Phase 2 implementations existing.
Phase 7 depends on Phase 6 patterns.
Phase 8 depends on the types from Phases 2, 3, and 5.
