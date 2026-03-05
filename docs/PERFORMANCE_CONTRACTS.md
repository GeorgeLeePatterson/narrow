# Performance Contracts

## Core Contract

ndarrow is a zero-cost bridge. The following invariants are absolute:

### Rule 1: Bridge Conversions Do Not Allocate

Any function that converts between Arrow and ndarray without the word "allocate", "copy", "cast",
"fill", "densify", or "compact" in its documentation MUST NOT allocate heap memory.

This includes:
- All `AsNdarray::as_ndarray` implementations (produces views, borrows buffers)
- All `AsNdarray::as_ndarray_unchecked` implementations
- All `AsNdarray::as_ndarray_masked` implementations
- All `IntoArrow::into_arrow` implementations on standard-layout arrays
- All `CsrView` construction from Arrow buffers
- All `reshape` operations (pointer + shape, no data movement)

### Rule 2: Allocating Functions Are Named and Documented

Functions that allocate are:
- Named to indicate the allocation (e.g., `cast`, `densify`, `fill_nulls`, `compact_non_null`)
- Documented with `/// # Allocation` section explaining what is allocated and why
- Located in the `helpers` module, not in the core `inbound`/`outbound` modules

### Rule 3: Unavoidable Allocations Are Documented

Some allocations are structurally unavoidable:
- `IntoArrow` on non-standard-layout arrays (requires `as_standard_layout().into_owned()`)
- `VariableShapeTensor` outbound (packing ragged data requires contiguous allocation)
- Any type-casting operation (different type = different memory)
- Sparse densification (dense representation is larger)

These are documented at the function level and at the type level.

## Allocation Classification

### Zero-Allocation Operations (Bridge Path)

| Operation                        | Mechanism                              | Complexity |
|----------------------------------|----------------------------------------|------------|
| PrimitiveArray -> ArrayView1     | Borrow values slice                    | O(1)       |
| FixedSizeList -> ArrayView2      | Borrow flat buffer, compute shape      | O(1)       |
| FixedShapeTensor -> ArrayViewD   | Borrow flat buffer, parse shape meta   | O(1)       |
| VarShapeTensor -> per-row view   | Borrow data slice at offset            | O(1)/row   |
| CSR ext type -> CsrView         | Borrow offsets + indices + values      | O(1)       |
| complex32 ext -> ArrayView1     | Borrow pair buffer, reinterpret as Complex32 | O(1) |
| complex64 ext -> ArrayView1     | Borrow pair buffer, reinterpret as Complex64 | O(1) |
| Two-column sparse -> CsrView    | Borrow from both columns               | O(1)       |
| Array1 -> PrimitiveArray         | into_raw_vec -> ScalarBuffer::from     | O(1)       |
| Array2 (std layout) -> FSL      | into_raw_vec -> ScalarBuffer::from     | O(1)       |
| ArrayD (std layout) -> Tensor   | into_raw_vec + metadata construction   | O(1)       |
| Array1<Complex32> -> complex32 ext | into_raw_vec -> pair reinterpret + ScalarBuffer | O(1) |
| Array1<Complex64> -> complex64 ext | into_raw_vec -> pair reinterpret + ScalarBuffer | O(1) |
| reshape(PrimitiveArray, shape)   | Pointer + shape, no copy               | O(1)       |
| null_count check                 | Read pre-computed count                 | O(1)       |

### Allocating Operations (Helper Path)

| Operation                        | Mechanism                              | Complexity |
|----------------------------------|----------------------------------------|------------|
| cast f32 -> f64                  | Allocate new buffer, widen each element| O(N)       |
| cast f64 -> f32                  | Allocate new buffer, ndarrow each elem  | O(N)       |
| densify(sparse, D)               | Allocate (M * D) buffer, fill          | O(M * D)   |
| fill_nulls_with_zero             | Allocate new buffer, copy + fill       | O(N)       |
| fill_nulls_with_mean             | Allocate new buffer, copy + fill       | O(N)       |
| compact_non_null(array)          | Allocate smaller buffer, copy valid    | O(N)       |
| Array2 (non-std) -> FSL         | as_standard_layout allocates copy      | O(M * N)   |
| VarShapeTensor outbound          | Pack ragged arrays into struct         | O(total)   |

### Computation Allocations (Expected, Not ndarrow's)

| Operation                        | Source    | Notes                             |
|----------------------------------|-----------|-----------------------------------|
| SVD of ArrayView2                | nabled    | Allocates u, s, vt                |
| cosine_similarity batch          | nabled    | Allocates result Array1           |
| Any nabled operation             | nabled    | Allocates owned result arrays     |

These are not ndarrow's allocations. ndarrow's job is to make the bridge free so that computation
allocations are the only ones that occur.

## Verification Strategy

### Compile-Time

- View-producing functions return `ArrayView*` types, which cannot own data
- `IntoArrow` consumes `self`, preventing double-use of the source buffer
- `NdarrowElement` trait bounds ensure type safety at compile time

### Runtime

- Debug-mode assertions on null checks in unchecked paths
- Shape validation before view construction (prevents UB)
- Standard-layout detection before ownership transfer

### Test-Time

- Allocation counting tests using a custom global allocator or `dhat`
- Round-trip property tests verifying value preservation
- Benchmark suite measuring conversion latency (should be ~nanoseconds)

## Performance Expectations

For a bridge operation (Arrow <-> ndarray):

| Array Size    | Expected Latency | Expected Allocations |
|---------------|------------------|----------------------|
| 100 elements  | < 100 ns         | 0                    |
| 10K elements  | < 100 ns         | 0                    |
| 1M elements   | < 100 ns         | 0                    |
| 100M elements | < 100 ns         | 0                    |

Latency should be **constant** regardless of array size, because the bridge is O(1). The
array size does not matter — we are creating a view (pointer + shape), not touching the data.

For helper operations (casting, densification), latency scales linearly with array size.
These are documented as allocating.
