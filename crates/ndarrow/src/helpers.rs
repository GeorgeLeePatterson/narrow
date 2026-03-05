//! Explicit helper APIs that may allocate.
//!
//! These helpers are intentionally separated from the zero-copy bridge path.

use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, PrimitiveArray,
    types::{ArrowPrimitiveType, Float32Type, Float64Type},
};
use arrow_buffer::ScalarBuffer;
use arrow_schema::Field;
use ndarray::{ArrayD, ArrayView2, ArrayViewD, IxDyn};

use crate::{element::NdarrowElement, error::NdarrowError, sparse::CsrView};

/// Casts `PrimitiveArray<f32>` to `PrimitiveArray<f64>`.
///
/// # Allocation
///
/// Allocates a new values buffer because the output element type differs.
#[must_use]
pub fn cast_f32_to_f64(array: &PrimitiveArray<Float32Type>) -> PrimitiveArray<Float64Type> {
    let values = array.values().iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
    PrimitiveArray::new(ScalarBuffer::from(values), array.nulls().cloned())
}

/// Casts `PrimitiveArray<f64>` to `PrimitiveArray<f32>`.
///
/// # Allocation
///
/// Allocates a new values buffer because the output element type differs.
///
/// # Errors
///
/// Returns an error when a finite `f64` value is outside the finite `f32`
/// range.
pub fn cast_f64_to_f32(
    array: &PrimitiveArray<Float64Type>,
) -> Result<PrimitiveArray<Float32Type>, NdarrowError> {
    let mut values = Vec::with_capacity(array.len());
    for value in array.values().iter().copied() {
        if value.is_finite() && !(f64::from(f32::MIN)..=f64::from(f32::MAX)).contains(&value) {
            return Err(NdarrowError::TypeMismatch {
                message: format!("cannot represent f64 value {value} as finite f32"),
            });
        }
        let casted =
            num_traits::cast::<f64, f32>(value).ok_or_else(|| NdarrowError::TypeMismatch {
                message: format!("cannot represent f64 value {value} as f32"),
            })?;
        values.push(casted);
    }
    Ok(PrimitiveArray::new(ScalarBuffer::from(values), array.nulls().cloned()))
}

/// Reinterprets a primitive Arrow array as a 2D ndarray view.
///
/// # Does not allocate
///
/// Returns an `ArrayView2` borrowing Arrow's values buffer.
///
/// # Errors
///
/// Returns an error if nulls are present or the requested shape is incompatible.
pub fn reshape_primitive_to_array2<T>(
    array: &PrimitiveArray<T>,
    rows: usize,
    cols: usize,
) -> Result<ArrayView2<'_, T::Native>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }
    let expected = rows.checked_mul(cols).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!("reshape dimensions overflow usize: rows={rows}, cols={cols}"),
    })?;
    if expected != array.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "reshape dimensions ({rows}, {cols}) expect length {expected}, found {}",
                array.len()
            ),
        });
    }

    ArrayView2::from_shape((rows, cols), array.values().as_ref()).map_err(NdarrowError::from)
}

/// Reinterprets a primitive Arrow array as an ND ndarray view.
///
/// # Does not allocate
///
/// Returns an `ArrayViewD` borrowing Arrow's values buffer.
///
/// # Errors
///
/// Returns an error if nulls are present or the requested shape is incompatible.
pub fn reshape_primitive_to_arrayd<'a, T>(
    array: &'a PrimitiveArray<T>,
    shape: &[usize],
) -> Result<ArrayViewD<'a, T::Native>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    let expected =
        shape.iter().try_fold(1_usize, |acc, dim| acc.checked_mul(*dim)).ok_or_else(|| {
            NdarrowError::ShapeMismatch {
                message: format!("reshape shape product overflows usize: {shape:?}"),
            }
        })?;

    if expected != array.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "reshape shape {:?} expects length {}, found {}",
                shape,
                expected,
                array.len()
            ),
        });
    }

    ArrayViewD::from_shape(IxDyn(shape), array.values().as_ref()).map_err(NdarrowError::from)
}

/// Converts an owned ndarray array to standard (C-contiguous) layout.
///
/// # Allocation
///
/// Allocates only when the source layout is non-standard.
#[must_use]
pub fn to_standard_layout<T>(array: ArrayD<T>) -> ArrayD<T>
where
    T: Clone,
{
    if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() }
}

/// Densifies a CSR view into an Arrow `FixedSizeListArray`.
///
/// # Allocation
///
/// Allocates a dense `(nrows * ncols)` values buffer and fills it from sparse
/// indices/values.
///
/// # Errors
///
/// Returns an error when CSR invariants are violated (invalid row pointers,
/// mismatched lengths, or out-of-bounds column indices).
pub fn densify_csr_view<T>(view: &CsrView<'_, T>) -> Result<FixedSizeListArray, NdarrowError>
where
    T: NdarrowElement,
{
    if view.row_ptrs.len() != view.nrows + 1 {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "row_ptrs length must be nrows + 1: expected {}, found {}",
                view.nrows + 1,
                view.row_ptrs.len()
            ),
        });
    }
    if view.col_indices.len() != view.values.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "col_indices and values lengths must match: {} vs {}",
                view.col_indices.len(),
                view.values.len()
            ),
        });
    }
    if view.row_ptrs[0] != 0 {
        return Err(NdarrowError::InvalidMetadata {
            message: format!("row_ptrs must start at 0, found {}", view.row_ptrs[0]),
        });
    }

    let last_raw = view.row_ptrs.last().copied().ok_or_else(|| NdarrowError::InvalidMetadata {
        message: "row_ptrs must not be empty".to_owned(),
    })?;
    let last_offset = usize::try_from(last_raw).map_err(|_| NdarrowError::InvalidMetadata {
        message: "row_ptrs contain negative offset".to_owned(),
    })?;
    if last_offset != view.values.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "last row_ptr offset ({last_offset}) must equal nnz ({})",
                view.values.len()
            ),
        });
    }

    let dense_len =
        view.nrows.checked_mul(view.ncols).ok_or_else(|| NdarrowError::ShapeMismatch {
            message: format!(
                "dense output size overflow: nrows={} ncols={}",
                view.nrows, view.ncols
            ),
        })?;
    let mut dense = vec![T::zero(); dense_len];

    for row in 0..view.nrows {
        let start =
            usize::try_from(view.row_ptrs[row]).map_err(|_| NdarrowError::InvalidMetadata {
                message: format!("row_ptrs contain negative offset at row {row}"),
            })?;
        let end =
            usize::try_from(view.row_ptrs[row + 1]).map_err(|_| NdarrowError::InvalidMetadata {
                message: format!("row_ptrs contain negative offset at row {}", row + 1),
            })?;
        if end < start {
            return Err(NdarrowError::InvalidMetadata {
                message: format!("row_ptrs must be non-decreasing at row {row}: {start} -> {end}"),
            });
        }

        for idx in start..end {
            let col = usize::try_from(view.col_indices[idx]).map_err(|_| {
                NdarrowError::ShapeMismatch {
                    message: format!("column index out of usize range at row {row}, index {idx}"),
                }
            })?;
            if col >= view.ncols {
                return Err(NdarrowError::ShapeMismatch {
                    message: format!(
                        "column index out of bounds at row {row}, index {idx}: {col} >= {}",
                        view.ncols
                    ),
                });
            }
            dense[row * view.ncols + col] = view.values[idx];
        }
    }

    let value_length = i32::try_from(view.ncols).map_err(|_| NdarrowError::ShapeMismatch {
        message: format!("ncols exceeds Arrow i32 limits: {}", view.ncols),
    })?;
    let values = PrimitiveArray::<T::ArrowType>::new(ScalarBuffer::from(dense), None);
    let item_field = Arc::new(Field::new("item", T::data_type(), false));
    Ok(FixedSizeListArray::new(item_field, value_length, Arc::new(values), None))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use arrow_array::{Array, Float32Array, Float64Array};
    use ndarray::{ArrayD, IxDyn};

    use super::*;

    #[test]
    fn cast_f32_to_f64_preserves_values_and_nulls() {
        let input = Float32Array::from(vec![Some(1.0_f32), None, Some(3.5)]);
        let output = cast_f32_to_f64(&input);
        assert_eq!(output.len(), input.len());
        assert_eq!(output.null_count(), input.null_count());
        assert_abs_diff_eq!(output.value(0), 1.0_f64);
        assert_abs_diff_eq!(output.value(2), 3.5_f64);
    }

    #[test]
    fn cast_f64_to_f32_preserves_values_and_nulls() {
        let input = Float64Array::from(vec![Some(2.0_f64), None, Some(4.25)]);
        let output = cast_f64_to_f32(&input).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(output.null_count(), input.null_count());
        assert_abs_diff_eq!(output.value(0), 2.0_f32);
        assert_abs_diff_eq!(output.value(2), 4.25_f32);
    }

    #[test]
    fn cast_f64_to_f32_rejects_non_representable_values() {
        let input = Float64Array::from(vec![f64::MAX]);
        let err = cast_f64_to_f32(&input).unwrap_err();
        assert!(matches!(err, NdarrowError::TypeMismatch { .. }));
    }

    #[test]
    fn cast_f64_to_f32_allows_nan_and_infinity() {
        let input = Float64Array::from(vec![f64::NEG_INFINITY, f64::NAN, f64::INFINITY]);
        let output = cast_f64_to_f32(&input).unwrap();
        assert!(output.value(0).is_infinite() && output.value(0).is_sign_negative());
        assert!(output.value(1).is_nan());
        assert!(output.value(2).is_infinite() && output.value(2).is_sign_positive());
    }

    #[test]
    fn reshape_primitive_to_array2_success() {
        let input = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let view = reshape_primitive_to_array2(&input, 2, 2).unwrap();
        assert_eq!(view.dim(), (2, 2));
        assert_abs_diff_eq!(view[[0, 1]], 2.0);
        assert_abs_diff_eq!(view[[1, 0]], 3.0);
    }

    #[test]
    fn reshape_primitive_to_array2_rejects_nulls() {
        let input = Float64Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
        let err = reshape_primitive_to_array2(&input, 2, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn reshape_primitive_to_array2_rejects_bad_shape() {
        let input = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let err = reshape_primitive_to_array2(&input, 2, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn reshape_primitive_to_arrayd_success() {
        let input = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let view = reshape_primitive_to_arrayd(&input, &[1, 2, 3]).unwrap();
        assert_eq!(view.shape(), &[1, 2, 3]);
        assert_abs_diff_eq!(view[[0, 1, 2]], 6.0_f32);
    }

    #[test]
    fn reshape_primitive_to_arrayd_rejects_nulls() {
        let input = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let err = reshape_primitive_to_arrayd(&input, &[3]).unwrap_err();
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn reshape_primitive_to_arrayd_rejects_overflow_shape_product() {
        let input = Float64Array::from(vec![1.0_f64]);
        let err = reshape_primitive_to_arrayd(&input, &[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn to_standard_layout_returns_standard() {
        let input = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let ptr = input.as_ptr();
        let output = to_standard_layout(input);
        assert!(output.is_standard_layout());
        assert_eq!(output.as_ptr(), ptr);
    }

    #[test]
    fn densify_csr_view_success() {
        let row_ptrs = vec![0_i32, 2, 3, 5];
        let col_indices = vec![0_u32, 2, 1, 0, 3];
        let values = vec![1.0_f64, 5.0, 2.0, 3.0, 4.0];
        let view = CsrView {
            nrows:       3,
            ncols:       4,
            row_ptrs:    &row_ptrs,
            col_indices: &col_indices,
            values:      &values,
        };

        let dense = densify_csr_view(&view).unwrap();
        assert_eq!(dense.len(), 3);
        assert_eq!(dense.value_length(), 4);

        let inner = dense.values().as_any().downcast_ref::<Float64Array>().unwrap();
        let actual: Vec<f64> = inner.values().iter().copied().collect();
        let expected = vec![
            1.0_f64, 0.0, 5.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 4.0,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *b);
        }
    }

    #[test]
    fn densify_csr_view_rejects_bad_row_ptr_length() {
        let row_ptrs = vec![0_i32, 1];
        let col_indices = vec![0_u32];
        let values = vec![1.0_f32];
        let view = CsrView {
            nrows:       2,
            ncols:       1,
            row_ptrs:    &row_ptrs,
            col_indices: &col_indices,
            values:      &values,
        };
        let err = densify_csr_view(&view).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn densify_csr_view_rejects_out_of_bounds_column() {
        let row_ptrs = vec![0_i32, 1];
        let col_indices = vec![2_u32];
        let values = vec![1.0_f64];
        let view = CsrView {
            nrows:       1,
            ncols:       2,
            row_ptrs:    &row_ptrs,
            col_indices: &col_indices,
            values:      &values,
        };
        let err = densify_csr_view(&view).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }
}
