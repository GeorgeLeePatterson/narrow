# Capability Matrix

## Status Legend

| Symbol | Meaning       |
|--------|---------------|
| Done   | Implemented, tested, documented |
| WIP    | In progress   |
| --     | Not started   |

## Inbound: Arrow -> ndarray (Zero-Copy Views)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| PrimitiveArray<f32> -> ArrayView1   | --     | Trivial slice borrow                         |
| PrimitiveArray<f64> -> ArrayView1   | --     | Trivial slice borrow                         |
| FixedSizeList<f32> -> ArrayView2    | --     | Flat buffer + reshape                        |
| FixedSizeList<f64> -> ArrayView2    | --     | Flat buffer + reshape                        |
| FixedShapeTensor -> ArrayViewD      | --     | Shape from extension metadata                |
| VariableShapeTensor -> per-row view | --     | Iterator over per-element ArrayViewD         |
| narrow.csr_matrix -> CsrView       | --     | Borrow offsets + indices + values            |
| Two-column sparse -> CsrView       | --     | Convenience API for separate columns         |

## Outbound: ndarray -> Arrow (Ownership Transfer)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| Array1<f32> -> PrimitiveArray       | --     | into_raw_vec -> ScalarBuffer                 |
| Array1<f64> -> PrimitiveArray       | --     | into_raw_vec -> ScalarBuffer                 |
| Array2<f32> -> FixedSizeList        | --     | into_raw_vec + field construction            |
| Array2<f64> -> FixedSizeList        | --     | into_raw_vec + field construction            |
| ArrayD<T> -> FixedShapeTensor       | --     | into_raw_vec + shape metadata                |
| CsrMatrix-like -> narrow.csr_matrix | --     | Transfer row_ptrs, indices, values           |

## Null Handling

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| Validated (null_count check)        | --     | Default tier, returns Result                 |
| Unchecked (caller guarantees)       | --     | Zero-cost, unsafe                            |
| Masked (view + validity bitmap)     | --     | Returns tuple, zero allocation               |
| fill_nulls(zero)                    | --     | Allocating helper                            |
| fill_nulls(mean)                    | --     | Allocating helper, requires float type       |
| compact_non_null                    | --     | Allocating helper, removes null rows         |

## Helpers (Explicit Allocation Points)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| cast f32 -> f64                     | --     | Element-wise widening                        |
| cast f64 -> f32                     | --     | Element-wise narrowing                       |
| densify sparse -> dense             | --     | Sparse to FixedSizeList                      |
| reshape PrimitiveArray -> 2D view   | --     | Zero-copy if dimensions align                |
| reshape PrimitiveArray -> ND view   | --     | Zero-copy if dimensions align                |
| to_standard_layout                  | --     | No-op if already C-contiguous                |

## Extension Types

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| arrow.fixed_shape_tensor support    | --     | Read/write canonical tensor extension        |
| arrow.variable_shape_tensor support | --     | Read/write canonical variable tensor         |
| narrow.csr_matrix definition        | --     | Custom sparse extension type                 |
| Extension type registration         | --     | Register handlers for deserialization        |

## Element Type Support

| Type       | Inbound | Outbound | Notes                               |
|------------|---------|----------|-------------------------------------|
| f32        | --      | --       | First-class, primary                |
| f64        | --      | --       | First-class, primary                |
| Complex32  | --      | --       | Future, pending trait bounds        |
| Complex64  | --      | --       | Future, pending trait bounds        |
| i32        | --      | --       | Future, for index arrays            |
| i64        | --      | --       | Future, for index arrays            |
| u32        | --      | --       | Future, for sparse indices          |
| u64        | --      | --       | Future                              |

## Core Infrastructure

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| NarrowElement trait                 | --     | Type bridge: Arrow <-> ndarray               |
| AsNdarray trait                     | --     | Inbound conversion contract                  |
| IntoArrow trait                     | --     | Outbound conversion contract                 |
| NarrowError enum                    | --     | Error taxonomy                               |
| Prelude module                      | --     | Convenience re-exports                       |

## Quality Infrastructure

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| justfile                            | --     | Build/test/lint/coverage commands             |
| CI pipeline                         | --     | GitHub Actions                               |
| Coverage >= 90%                     | --     | cargo llvm-cov                               |
| Benchmarks                          | --     | Conversion overhead measurement              |
| Property tests                      | --     | Round-trip Arrow -> ndarray -> Arrow          |
