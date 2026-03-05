//! Explicit helper APIs that may allocate.
//!
//! These helpers are intentionally separated from the zero-copy bridge path.

use arrow_array::{
    Array, PrimitiveArray,
    types::{ArrowPrimitiveType, Float32Type, Float64Type},
};
use arrow_buffer::ScalarBuffer;
use ndarray::{ArrayD, ArrayView2, ArrayViewD, IxDyn};

use crate::{element::NdarrowElement, error::NdarrowError};

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
/// Returns an error when a value cannot be represented as `f32`.
pub fn cast_f64_to_f32(
    array: &PrimitiveArray<Float64Type>,
) -> Result<PrimitiveArray<Float32Type>, NdarrowError> {
    let mut values = Vec::with_capacity(array.len());
    for value in array.values().iter().copied() {
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
    fn reshape_primitive_to_array2_success() {
        let input = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let view = reshape_primitive_to_array2(&input, 2, 2).unwrap();
        assert_eq!(view.dim(), (2, 2));
        assert_abs_diff_eq!(view[[0, 1]], 2.0);
        assert_abs_diff_eq!(view[[1, 0]], 3.0);
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
    fn to_standard_layout_returns_standard() {
        let input = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let ptr = input.as_ptr();
        let output = to_standard_layout(input);
        assert!(output.is_standard_layout());
        assert_eq!(output.as_ptr(), ptr);
    }
}
