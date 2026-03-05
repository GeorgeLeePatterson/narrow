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
| PrimitiveArray<f32> -> ArrayView1   | Done   | Implemented via `AsNdarray`                  |
| PrimitiveArray<f64> -> ArrayView1   | Done   | Implemented via `AsNdarray`                  |
| FixedSizeList<f32> -> ArrayView2    | Done   | Implemented (`validated`/`unchecked`/`masked`) |
| FixedSizeList<f64> -> ArrayView2    | Done   | Implemented (`validated`/`unchecked`/`masked`) |
| FixedShapeTensor -> ArrayViewD      | Done   | `fixed_shape_tensor_as_array_viewd`          |
| VariableShapeTensor -> per-row view | Done   | `variable_shape_tensor_iter` iterator        |
| ndarrow.csr_matrix -> CsrView       | Done   | `csr_view_from_extension`                    |
| Two-column sparse -> CsrView        | Done   | `csr_view_from_columns` convenience path     |

## Outbound: ndarray -> Arrow (Ownership Transfer)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| Array1<f32> -> PrimitiveArray       | Done   | Implemented via `IntoArrow`                  |
| Array1<f64> -> PrimitiveArray       | Done   | Implemented via `IntoArrow`                  |
| Array2<f32> -> FixedSizeList        | Done   | Implemented via `IntoArrow`                  |
| Array2<f64> -> FixedSizeList        | Done   | Implemented via `IntoArrow`                  |
| ArrayD<T> -> FixedShapeTensor       | Done   | `arrayd_to_fixed_shape_tensor`               |
| ArrayD<T> rows -> VariableShapeTensor | Done | `arrays_to_variable_shape_tensor`            |
| CsrMatrix-like -> ndarrow.csr_matrix | Done  | `csr_to_extension_array`                     |

## Null Handling

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| Validated (null_count check)        | Done   | Default tier, returns Result                 |
| Unchecked (caller guarantees)       | Done   | Zero-cost, unsafe                            |
| Masked (view + validity bitmap)     | Done   | Returns tuple, zero allocation               |
| fill_nulls(zero)                    | Done   | `helpers::fill_nulls_with_zero`              |
| fill_nulls(mean)                    | Done   | `helpers::fill_nulls_with_mean` (float types) |
| compact_non_null                    | Done   | `helpers::compact_non_null`                  |

## Helpers (Explicit Allocation Points)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| cast f32 -> f64                     | Done   | `cast_f32_to_f64`                            |
| cast f64 -> f32                     | Done   | `cast_f64_to_f32` (fallible)                 |
| densify sparse -> dense             | Done   | `helpers::densify_csr_view`                  |
| reshape PrimitiveArray -> 2D view   | Done   | `reshape_primitive_to_array2`                |
| reshape PrimitiveArray -> ND view   | Done   | `reshape_primitive_to_arrayd`                |
| to_standard_layout                  | Done   | No-op if already C-contiguous                |

## Extension Types

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| arrow.fixed_shape_tensor support    | Done   | Inbound + outbound implemented               |
| arrow.variable_shape_tensor support | Done   | Inbound iterator + outbound implemented      |
| ndarrow.csr_matrix definition       | Done   | `CsrMatrixExtension` implemented             |
| ndarrow.complex32 definition        | Done   | `Complex32Extension` implemented             |
| ndarrow.complex64 definition        | Done   | `Complex64Extension` implemented             |
| Extension type registration         | Done   | `deserialize_registered_extension` registry  |

## Element Type Support

| Type       | Inbound | Outbound | Notes                               |
|------------|---------|----------|-------------------------------------|
| f32        | Done    | Done     | First-class, primary                |
| f64        | Done    | Done     | First-class, primary                |
| Complex32  | Done    | Done     | Via `ndarrow.complex32` extension APIs |
| Complex64  | Done    | Done     | Via `ndarrow.complex64` extension APIs |
| i32        | --      | --       | Future, for index arrays            |
| i64        | --      | --       | Future, for index arrays            |
| u32        | --      | --       | Future, for sparse indices          |
| u64        | --      | --       | Future                              |

## Core Infrastructure

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| NdarrowElement trait                 | Done   | Type bridge: Arrow <-> ndarray               |
| AsNdarray trait                     | Done   | Inbound conversion contract                  |
| IntoArrow trait                     | Done   | Outbound conversion contract                 |
| NdarrowError enum                    | Done   | Error taxonomy                               |
| Prelude module                      | Done   | `prelude` convenience re-exports             |

## Quality Infrastructure

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| justfile                            | Done   | Build/test/lint/coverage/release commands    |
| CI pipeline                         | Done   | GitHub Actions                               |
| Coverage >= 90%                     | Done   | Gate configured in just/CI                   |
| Benchmarks                          | Done   | Public API conversion benchmark suites       |
| Benchmark baseline/reporting gate   | Done   | Criterion baseline cache + regression summary/check |
| Sparse/tensor allocation verification | Done | Pointer-identity tests for CSR/tensor view paths |
| Property tests                      | Done   | Dense/sparse/tensor round-trip properties via `proptest` |
