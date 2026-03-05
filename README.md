# ndarrow

[![Crates.io](https://img.shields.io/crates/v/ndarrow.svg)](https://crates.io/crates/ndarrow)
[![Documentation](https://docs.rs/ndarrow/badge.svg)](https://docs.rs/ndarrow)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/georgeleepatterson/ndarrow/ci.yml?branch=main)](https://github.com/georgeleepatterson/ndarrow/actions)
[![Coverage](https://codecov.io/gh/georgeleepatterson/ndarrow/branch/main/graph/badge.svg)](https://codecov.io/gh/georgeleepatterson/ndarrow)

A zero-copy bridge between [Apache Arrow](https://arrow.apache.org/) and
[ndarray](https://docs.rs/ndarray). Convert Arrow arrays to ndarray views and back without
allocation overhead.

> ndarrow is under active development. Public APIs will change until v1. Pin your version.

## Install

```toml
[dependencies]
ndarrow = "0.0.2"
```

## What It Does

ndarrow lets you move data between Arrow and ndarray with zero allocations on the bridge path:

- **Arrow to ndarray**: borrow Arrow buffers as ndarray views (O(1), no copy)
- **ndarray to Arrow**: transfer owned ndarray buffers into Arrow arrays (O(1), ownership move)

This enables Arrow-native systems (DataFusion, Flight, IPC) to leverage ndarray-native
numerical libraries (like [nabled](https://crates.io/crates/nabled)) without paying for
serialization or conversion.

## Quick Example

```rust
use arrow_array::{Float64Array, FixedSizeListArray};
use ndarrow::{AsNdarray, IntoArrow};
use ndarray::Array1;

// Arrow -> ndarray (zero-copy view)
let arrow_array = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
let view = arrow_array.as_ndarray()?;  // ArrayView1<f64>, no allocation
assert_eq!(view[0], 1.0);

// ndarray -> Arrow (zero-copy ownership transfer)
let result = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
let arrow_result = result.into_arrow()?;  // PrimitiveArray<Float64>, no allocation
```

## Type Mappings

| Arrow Type | ndarray Type | Direction | Copy? |
|---|---|---|---|
| `PrimitiveArray<T>` | `ArrayView1<T>` | Arrow -> ndarray | Zero-copy |
| `FixedSizeList<T>(D)` | `ArrayView2<T>` (M, D) | Arrow -> ndarray | Zero-copy |
| `arrow.fixed_shape_tensor` | `ArrayViewD<T>` | Arrow -> ndarray | Zero-copy |
| `arrow.variable_shape_tensor` | Per-row `ArrayViewD<T>` | Arrow -> ndarray | Zero-copy |
| `ndarrow.csr_matrix` | `CsrView<T>` | Arrow -> ndarray | Zero-copy |
| `Array1<T>` | `PrimitiveArray<T>` | ndarray -> Arrow | Zero-copy |
| `Array2<T>` (M, N) | `FixedSizeList<T>(N)` | ndarray -> Arrow | Zero-copy* |
| `ArrayD<T>` | `arrow.fixed_shape_tensor` | ndarray -> Arrow | Zero-copy* |

\* Zero-copy if standard (C-contiguous) layout. Allocates a layout copy otherwise.

## Supported Element Types

- `f32`, `f64` (first-class)
- Additional types via the `NdarrowElement` trait

## Null Handling

Null semantics are explicit at the call site:

```rust
// Validated: returns Err if nulls present (O(1) check)
let view = array.as_ndarray()?;

// Unchecked: caller guarantees no nulls (zero cost)
let view = unsafe { array.as_ndarray_unchecked() };

// Masked: returns view + validity bitmap (zero allocation)
let (view, mask) = array.as_ndarray_masked();
```

## Extension Types

ndarrow uses canonical Arrow extension types where they exist:

- `arrow.fixed_shape_tensor` — fixed-shape multi-dimensional data
- `arrow.variable_shape_tensor` — variable-shape data (e.g., multi-vectors)

And defines its own for gaps:

- `ndarrow.csr_matrix` — CSR sparse matrix representation

## Performance Guarantee

Bridge conversions are O(1) regardless of array size. The bridge creates views (pointer +
shape) or transfers buffer ownership. It never touches the data. See
[docs/PERFORMANCE_CONTRACTS.md](docs/PERFORMANCE_CONTRACTS.md) for the full allocation contract.

## Quality Gates

```bash
just checks
```

## License

Licensed under the Apache License, Version 2.0.

See [LICENSE](LICENSE) for details.
