# Architecture

## Overview

narrow is a single crate organized by conversion direction and data category. The trait hierarchy
defines the bridge contract. Concrete implementations wire Arrow types to ndarray types.

## Module Tree

```
narrow/
  src/
    lib.rs                      # Crate root, re-exports, prelude
    error.rs                    # NarrowError enum
    element.rs                  # NarrowElement trait (Arrow <-> ndarray type bridge)

    ext/                        # Arrow extension type definitions
      mod.rs
      csr_matrix.rs             # narrow.csr_matrix extension type

    inbound/                    # Arrow -> ndarray (zero-copy views)
      mod.rs
      primitive.rs              # PrimitiveArray<T> -> ArrayView1<T>
      fixed_size_list.rs        # FixedSizeList<T>(D) -> ArrayView2<T>
      tensor.rs                 # arrow.fixed_shape_tensor -> ArrayViewD<T>
      variable_tensor.rs        # arrow.variable_shape_tensor -> per-row ArrayViewD<T>
      sparse.rs                 # narrow.csr_matrix -> CsrView (or two-column convenience)

    outbound/                   # ndarray -> Arrow (ownership transfer)
      mod.rs
      array1.rs                 # Array1<T> -> PrimitiveArray<T>
      array2.rs                 # Array2<T> -> FixedSizeList<T>(N)
      arrayd.rs                 # ArrayD<T> -> arrow.fixed_shape_tensor
      sparse.rs                 # CsrMatrix-like -> narrow.csr_matrix

    nulls.rs                    # Null handling utilities (validate, mask, fill)

    helpers/                    # Explicit allocation points
      mod.rs
      cast.rs                   # Type widening/narrowing (e.g., f32 -> f64)
      densify.rs                # Sparse -> dense conversion
      reshape.rs                # PrimitiveArray -> 2D/3D/ND reinterpretation
      layout.rs                 # Standard layout normalization
```

## Trait Hierarchy

### NarrowElement

Bridges Arrow's type system with ndarray's element requirements.

```rust
pub trait NarrowElement: Clone + 'static {
    /// The corresponding Arrow primitive type.
    type ArrowType: ArrowPrimitiveType<Native = Self>;
}

impl NarrowElement for f32 {
    type ArrowType = Float32Type;
}

impl NarrowElement for f64 {
    type ArrowType = Float64Type;
}
```

This trait connects the worlds: any `T: NarrowElement` can be used in both Arrow arrays
and ndarray arrays. The associated type ensures compile-time type correspondence.

### AsNdarray (Inbound)

Converts Arrow arrays to ndarray views without allocation.

```rust
pub trait AsNdarray {
    type View<'a> where Self: 'a;

    /// Zero-copy view. Returns Err if nulls present.
    fn as_ndarray(&self) -> Result<Self::View<'_>, NarrowError>;

    /// Zero-copy view. Caller guarantees no nulls.
    /// # Safety
    /// Undefined behavior if the array contains nulls and the caller
    /// uses the view to access null positions.
    unsafe fn as_ndarray_unchecked(&self) -> Self::View<'_>;

    /// Zero-copy view with validity bitmap.
    fn as_ndarray_masked(&self) -> (Self::View<'_>, Option<&BooleanBuffer>);
}
```

Implemented for:
- `PrimitiveArray<T>` where `T::Native: NarrowElement` -> `ArrayView1<T::Native>`
- `FixedSizeListArray` (with inner `PrimitiveArray<T>`) -> `ArrayView2<T::Native>`
- `FixedShapeTensor` arrays -> `ArrayViewD<T::Native>`
- `VariableShapeTensor` arrays -> per-row `ArrayViewD<T::Native>` (via iterator)

### IntoArrow (Outbound)

Converts owned ndarray arrays to Arrow arrays via ownership transfer.

```rust
pub trait IntoArrow {
    type ArrowArray;

    /// Transfer ownership. Zero-copy if standard layout.
    fn into_arrow(self) -> Result<Self::ArrowArray, NarrowError>;
}
```

Implemented for:
- `Array1<T>` where `T: NarrowElement` -> `PrimitiveArray<T::ArrowType>`
- `Array2<T>` where `T: NarrowElement` -> `FixedSizeListArray`
- `ArrayD<T>` where `T: NarrowElement` -> `FixedShapeTensor` array (with shape metadata)

## Type Mapping Table

### Inbound (Arrow -> ndarray)

| Arrow Type                          | ndarray Type              | Copy? | Mechanism                    |
|-------------------------------------|---------------------------|-------|------------------------------|
| `PrimitiveArray<T>`                 | `ArrayView1<T::Native>`   | Zero  | Borrow values slice          |
| `FixedSizeList<T>(D)`              | `ArrayView2<T::Native>`   | Zero  | Borrow flat buffer + reshape |
| `arrow.fixed_shape_tensor`          | `ArrayViewD<T::Native>`   | Zero  | Borrow flat buffer + shape   |
| `arrow.variable_shape_tensor`       | Per-row `ArrayViewD`      | Zero  | Borrow slice per element     |
| `narrow.csr_matrix`                 | `CsrView` / equivalent   | Zero  | Borrow offsets + indices + values |
| Two-column sparse (indices+values)  | `CsrView` / equivalent   | Zero  | Borrow from both columns     |

### Outbound (ndarray -> Arrow)

| ndarray Type        | Arrow Type                          | Copy? | Mechanism                          |
|---------------------|-------------------------------------|-------|------------------------------------|
| `Array1<T>`         | `PrimitiveArray<T::ArrowType>`      | Zero  | `into_raw_vec()` -> `ScalarBuffer` |
| `Array2<T>` (M, N)  | `FixedSizeList<T>(N)`              | Zero* | `into_raw_vec()` -> buffer         |
| `ArrayD<T>`         | `arrow.fixed_shape_tensor`          | Zero* | `into_raw_vec()` + shape metadata  |
| Sparse owned        | `narrow.csr_matrix`                 | Zero* | Transfer row_ptrs, indices, values |

\* Zero-copy if standard layout. Allocates `as_standard_layout()` copy if not.

### Helpers (Explicit Allocation)

| Helper                | Input                 | Output                    | Allocates? |
|-----------------------|-----------------------|---------------------------|------------|
| `cast::<U>(array)`    | `PrimitiveArray<T>`   | `PrimitiveArray<U>`       | Yes        |
| `densify(sparse, D)`  | `narrow.csr_matrix`   | `FixedSizeList<T>(D)`     | Yes        |
| `fill_nulls(strategy)`| Any nullable array    | Same type, no nulls       | Yes        |
| `reshape(array, shape)`| `PrimitiveArray<T>`  | `ArrayView2/D<T::Native>` | No (view)  |
| `to_standard_layout()`| `ArrayD<T>`           | `ArrayD<T>` (C-order)     | Only if needed |

## Data Flow

### Typical Processing Pipeline

```
                    Arrow World                    ndarray World
                    ──────────                     ─────────────

RecordBatch ──> Extract column ──> AsNdarray ──> ArrayView ──> nabled op ──> Array (owned)
                                   (zero-copy)    (borrow)      (allocates)   (result)

                                                              IntoArrow <── Array (owned)
                                                              (zero-copy)
                                                                  │
                                                                  v
                                                          New Arrow Array
                                                                  │
                                                                  v
                                                          New RecordBatch
```

### Allocation Boundaries

```
┌─────────────────────────────────────────────────────────┐
│  Arrow (DataFusion, IPC, Flight, etc.)                  │  Owns input buffers
├─────────────────────────────────────────────────────────┤
│  narrow inbound                                         │  ZERO allocation
│  Arrow array -> ndarray view                            │  Just pointer + shape
├─────────────────────────────────────────────────────────┤
│  Computation (nabled, or any ndarray consumer)           │  ALLOCATES (expected)
│  View in, owned array out                               │  This is the real work
├─────────────────────────────────────────────────────────┤
│  narrow outbound                                        │  ZERO allocation
│  Owned array -> Arrow array                             │  Ownership transfer
├─────────────────────────────────────────────────────────┤
│  Arrow (DataFusion, IPC, Flight, etc.)                  │  Owns output buffers
└─────────────────────────────────────────────────────────┘
```

## Extension Type Architecture

### Canonical Types (from Arrow spec)

narrow registers handlers for canonical extension types that have ndarray mappings:

- `arrow.fixed_shape_tensor`: Extract shape from metadata, validate against storage type,
  provide `ArrayViewD` with shape `[batch_dim, ...tensor_shape]`.
- `arrow.variable_shape_tensor`: Extract uniform_shape from metadata, provide per-element
  `ArrayViewD` iterator.

### Custom Types (narrow-defined)

- `narrow.csr_matrix`:
  - Storage: `StructArray{indices: List<UInt32>, values: List<T>}`
  - Metadata: `{"ncols": N}` (required)
  - Validation: Both List fields must have identical offsets. Values type must be a
    supported NarrowElement type.
  - ndarray mapping: `CsrView` holding borrowed `&[i32]` offsets, `&[u32]` indices,
    `&[T]` values, plus `ncols: usize`.

## Sparse Representation

### CSR Memory Layout Correspondence

```
Arrow List<UInt32> indices column:
  offsets buffer:  [0, 3, 5, 9, ...]     ──> CSR row_ptrs
  values buffer:   [3, 7, 15, 1, 8, ...] ──> CSR col_indices

Arrow List<T> values column:
  offsets buffer:  [0, 3, 5, 9, ...]     ──> (must equal indices offsets)
  values buffer:   [0.5, 0.3, 0.8, ...]  ──> CSR values
```

The Arrow List representation IS CSR format. The offsets buffer is `row_ptrs`. The integer values
buffer is `col_indices`. The float values buffer is `values`. Zero-copy.

### Type Alignment Note

Arrow uses `i32` for List offsets and `u32` for index values. ndarray/nabled may use `usize`.
The view type exposes Arrow's native types (`&[i32]`, `&[u32]`). If a consumer needs `usize`,
that conversion is the consumer's responsibility, not narrow's.

See `docs/NABLED_CHANGES.md` for the corresponding nabled change to accept these types.

## Error Model

```rust
pub enum NarrowError {
    /// Arrow array contains null values where none were expected.
    NullsPresent { null_count: usize },

    /// The Arrow array's data type does not match the expected type.
    TypeMismatch { expected: DataType, found: DataType },

    /// The array shape is incompatible with the requested ndarray shape.
    ShapeMismatch { expected: Vec<usize>, found: Vec<usize> },

    /// Extension type metadata is missing or invalid.
    InvalidMetadata { message: String },

    /// The inner array of a composite type (FixedSizeList, Struct) is not
    /// the expected type.
    InnerTypeMismatch { message: String },

    /// The array is not in standard (C-contiguous) layout and cannot be
    /// transferred to Arrow without copying.
    NonStandardLayout,

    /// Sparse arrays have mismatched offsets between indices and values.
    SparseOffsetMismatch,

    /// An Arrow error propagated from the arrow crate.
    Arrow(ArrowError),
}
```
