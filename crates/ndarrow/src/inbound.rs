//! Arrow to ndarray conversions (zero-copy views).
//!
//! This module provides zero-copy conversions from Arrow arrays to ndarray views.
//! All conversions borrow the Arrow buffer and produce ndarray views — no allocation occurs.
//!
//! # Null handling
//!
//! Three tiers of null handling are provided:
//!
//! - [`AsNdarray::as_ndarray`] — validates no nulls, returns `Result`. O(1).
//! - [`AsNdarray::as_ndarray_unchecked`] — caller guarantees no nulls. Zero cost.
//! - [`AsNdarray::as_ndarray_masked`] — returns view + optional validity bitmap. Zero allocation.

use arrow_array::{Array, FixedSizeListArray, PrimitiveArray, types::ArrowPrimitiveType};
use arrow_buffer::NullBuffer;
use ndarray::{ArrayView1, ArrayView2};

use crate::{element::NdarrowElement, error::NdarrowError};

/// Zero-copy conversion from an Arrow array to an ndarray view.
///
/// Implementations borrow the Arrow buffer and produce ndarray views.
/// No allocation occurs on the bridge path.
///
/// # Does not allocate
///
/// All methods produce views (borrowed data). The view's lifetime is tied
/// to the Arrow array's lifetime.
pub trait AsNdarray {
    /// The ndarray view type produced by this conversion.
    type View<'a>
    where
        Self: 'a;

    /// Zero-copy conversion to an ndarray view.
    ///
    /// Returns `Err(NdarrowError::NullsPresent)` if the array contains any null values.
    /// The null check is O(1) — it reads the pre-computed null count, not the data.
    ///
    /// # Does not allocate
    ///
    /// # Errors
    ///
    /// Returns [`NdarrowError::NullsPresent`] when the array contains nulls.
    fn as_ndarray(&self) -> Result<Self::View<'_>, NdarrowError>;

    /// Zero-copy conversion to an ndarray view without null checking.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the array contains no null values.
    /// Accessing data at null positions through the returned view is undefined behavior
    /// (the underlying buffer may contain arbitrary bytes at those positions).
    ///
    /// In debug builds, this asserts `null_count() == 0`.
    ///
    /// # Does not allocate
    unsafe fn as_ndarray_unchecked(&self) -> Self::View<'_>;

    /// Zero-copy conversion to an ndarray view with a validity bitmap.
    ///
    /// Always succeeds. Returns the view alongside the optional null buffer.
    /// If `null_count() == 0`, the null buffer is `None`.
    ///
    /// The caller decides how to handle nulls using the bitmap.
    ///
    /// # Does not allocate
    fn as_ndarray_masked(&self) -> (Self::View<'_>, Option<&NullBuffer>);
}

// ─── PrimitiveArray<T> -> ArrayView1<T::Native> ───

impl<T> AsNdarray for PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    type View<'a> = ArrayView1<'a, T::Native>;

    fn as_ndarray(&self) -> Result<Self::View<'_>, NdarrowError> {
        if self.null_count() > 0 {
            return Err(NdarrowError::NullsPresent { null_count: self.null_count() });
        }
        // SAFETY: values() returns a contiguous &[T::Native] slice of length self.len().
        // With no nulls, all positions are valid.
        Ok(ArrayView1::from(self.values().as_ref()))
    }

    unsafe fn as_ndarray_unchecked(&self) -> Self::View<'_> {
        debug_assert_eq!(self.null_count(), 0, "as_ndarray_unchecked called on array with nulls");
        ArrayView1::from(self.values().as_ref())
    }

    fn as_ndarray_masked(&self) -> (Self::View<'_>, Option<&NullBuffer>) {
        let view = ArrayView1::from(self.values().as_ref());
        (view, self.nulls())
    }
}

// ─── FixedSizeListArray -> ArrayView2<T::Native> ───

fn fixed_size_list_value_length(array: &FixedSizeListArray) -> Result<usize, NdarrowError> {
    usize::try_from(array.value_length()).map_err(|_| NdarrowError::InvalidMetadata {
        message: format!(
            "FixedSizeList value_length must be non-negative; got {}",
            array.value_length()
        ),
    })
}

/// Zero-copy conversion from a [`FixedSizeListArray`] to an [`ArrayView2`].
///
/// The array is interpreted as an M x N matrix where:
/// - M = number of rows (`array.len()`)
/// - N = fixed list size (`array.value_length()`)
///
/// The inner values buffer must be a `PrimitiveArray<T>` where `T::Native: NdarrowElement`.
///
/// # Does not allocate
///
/// The returned view borrows the inner values buffer directly.
///
/// # Errors
///
/// Returns:
/// - [`NdarrowError::NullsPresent`] if either outer rows or inner values contain nulls.
/// - [`NdarrowError::InnerTypeMismatch`] if the inner values are not `PrimitiveArray<T>`.
/// - [`NdarrowError::InvalidMetadata`] if `value_length` is negative.
/// - [`NdarrowError::Shape`] if the inner buffer length does not match `(rows, value_length)`.
pub fn fixed_size_list_as_array2<T>(
    array: &FixedSizeListArray,
) -> Result<ArrayView2<'_, T::Native>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    // Validate no outer nulls (entire rows missing).
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    // Downcast inner values to the expected primitive type.
    let values = array.values().as_any().downcast_ref::<PrimitiveArray<T>>().ok_or_else(|| {
        NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected inner array of type {:?}, got {:?}",
                T::DATA_TYPE,
                array.values().data_type()
            ),
        }
    })?;

    // Validate no inner nulls (individual components missing).
    if values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: values.null_count() });
    }

    let n = fixed_size_list_value_length(array)?;
    let m = array.len();

    // The inner values buffer is a contiguous [T::Native] of length m * n.
    let slice: &[T::Native] = values.values().as_ref();

    ArrayView2::from_shape((m, n), slice).map_err(NdarrowError::from)
}

/// Zero-copy conversion from a [`FixedSizeListArray`] to an [`ArrayView2`], without null checks.
///
/// # Safety
///
/// The caller must guarantee that neither the outer array nor the inner values array
/// contains null values.
///
/// # Does not allocate
///
/// # Panics
///
/// Panics if:
/// - the inner values are not a `PrimitiveArray<T>`;
/// - `value_length` is negative;
/// - or Arrow's `(rows, value_length)` invariant is violated.
#[must_use]
pub unsafe fn fixed_size_list_as_array2_unchecked<T>(
    array: &FixedSizeListArray,
) -> ArrayView2<'_, T::Native>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    debug_assert_eq!(array.null_count(), 0, "unchecked called on array with outer nulls");

    let values = array
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("inner array type mismatch in unchecked path");

    debug_assert_eq!(values.null_count(), 0, "unchecked called on array with inner nulls");

    let n = usize::try_from(array.value_length())
        .expect("FixedSizeList value_length must be non-negative");
    let m = array.len();
    let slice: &[T::Native] = values.values().as_ref();

    // SAFETY: caller guarantees no nulls, and m * n == slice.len() by FixedSizeList invariant.
    ArrayView2::from_shape((m, n), slice).expect("shape must match FixedSizeList invariant")
}

/// Zero-copy conversion from a [`FixedSizeListArray`] to an [`ArrayView2`] with an outer-row
/// validity bitmap.
///
/// Returns the view alongside the optional outer null buffer.
/// Inner primitive nulls are rejected because this API only exposes whole-row validity; it does
/// not provide a component-level null mask.
///
/// # Does not allocate
///
/// # Errors
///
/// Returns:
/// - [`NdarrowError::NullsPresent`] if the inner values contain nulls.
/// - [`NdarrowError::InnerTypeMismatch`] if the inner values are not `PrimitiveArray<T>`.
/// - [`NdarrowError::InvalidMetadata`] if `value_length` is negative.
/// - [`NdarrowError::Shape`] if the inner buffer length does not match `(rows, value_length)`.
pub fn fixed_size_list_as_array2_masked<T>(
    array: &FixedSizeListArray,
) -> Result<(ArrayView2<'_, T::Native>, Option<&NullBuffer>), NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let values = array.values().as_any().downcast_ref::<PrimitiveArray<T>>().ok_or_else(|| {
        NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected inner array of type {:?}, got {:?}",
                T::DATA_TYPE,
                array.values().data_type()
            ),
        }
    })?;
    if values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: values.null_count() });
    }

    let n = fixed_size_list_value_length(array)?;
    let m = array.len();
    let slice: &[T::Native] = values.values().as_ref();

    let view = ArrayView2::from_shape((m, n), slice)?;
    Ok((view, array.nulls()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_abs_diff_eq;
    use arrow_array::{
        Array, Float32Array, Float64Array,
        types::{Float32Type, Float64Type},
    };
    use arrow_schema::Field;

    use super::*;

    // ─── PrimitiveArray tests ───

    #[test]
    fn primitive_f64_to_view1() {
        let arr = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let view = arr.as_ndarray().unwrap();
        assert_eq!(view.len(), 4);
        assert_abs_diff_eq!(view[0], 1.0);
        assert_abs_diff_eq!(view[3], 4.0);
    }

    #[test]
    fn primitive_f32_to_view1() {
        let arr = Float32Array::from(vec![1.0_f32, 2.0, 3.0]);
        let view = arr.as_ndarray().unwrap();
        assert_eq!(view.len(), 3);
        assert_abs_diff_eq!(view[1], 2.0_f32);
    }

    #[test]
    fn primitive_empty() {
        let arr = Float64Array::from(Vec::<f64>::new());
        let view = arr.as_ndarray().unwrap();
        assert_eq!(view.len(), 0);
    }

    #[test]
    fn primitive_with_nulls_errors() {
        let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let result = arr.as_ndarray();
        assert!(result.is_err());
        match result.unwrap_err() {
            NdarrowError::NullsPresent { null_count } => assert_eq!(null_count, 1),
            other => panic!("expected NullsPresent, got {other:?}"),
        }
    }

    #[test]
    fn primitive_unchecked() {
        let arr = Float64Array::from(vec![10.0, 20.0]);
        let view = unsafe { arr.as_ndarray_unchecked() };
        assert_abs_diff_eq!(view[0], 10.0);
        assert_abs_diff_eq!(view[1], 20.0);
    }

    #[test]
    fn primitive_masked_no_nulls() {
        let arr = Float64Array::from(vec![1.0, 2.0]);
        let (view, mask) = arr.as_ndarray_masked();
        assert_eq!(view.len(), 2);
        assert!(mask.is_none());
    }

    #[test]
    fn primitive_masked_with_nulls() {
        let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let (view, mask) = arr.as_ndarray_masked();
        // View covers all positions (including null ones — buffer is valid, semantics are
        // caller's).
        assert_eq!(view.len(), 3);
        assert!(mask.is_some());
        let nulls = mask.unwrap();
        assert!(nulls.is_valid(0));
        assert!(!nulls.is_valid(1));
        assert!(nulls.is_valid(2));
    }

    // ─── FixedSizeListArray tests ───

    fn make_fsl_f64(data: &[f64], dim: i32) -> FixedSizeListArray {
        let values = Float64Array::from(data.to_vec());
        let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
        FixedSizeListArray::new(field, dim, Arc::new(values), None)
    }

    fn make_fsl_f32(data: &[f32], dim: i32) -> FixedSizeListArray {
        let values = Float32Array::from(data.to_vec());
        let field = Arc::new(Field::new("item", arrow_schema::DataType::Float32, false));
        FixedSizeListArray::new(field, dim, Arc::new(values), None)
    }

    #[test]
    fn fsl_f64_to_view2() {
        // 3 rows of 2-dimensional vectors
        let fsl = make_fsl_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
        let view = fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();
        assert_eq!(view.dim(), (3, 2));
        assert_abs_diff_eq!(view[[0, 0]], 1.0);
        assert_abs_diff_eq!(view[[0, 1]], 2.0);
        assert_abs_diff_eq!(view[[1, 0]], 3.0);
        assert_abs_diff_eq!(view[[2, 1]], 6.0);
    }

    #[test]
    fn fsl_f32_to_view2() {
        let fsl = make_fsl_f32(&[1.0_f32, 2.0, 3.0, 4.0], 2);
        let view = fixed_size_list_as_array2::<Float32Type>(&fsl).unwrap();
        assert_eq!(view.dim(), (2, 2));
        assert_abs_diff_eq!(view[[1, 1]], 4.0_f32);
    }

    #[test]
    fn fsl_empty() {
        let fsl = make_fsl_f64(&[], 3);
        let view = fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();
        assert_eq!(view.dim(), (0, 3));
    }

    #[test]
    fn fsl_single_row() {
        let fsl = make_fsl_f64(&[10.0, 20.0, 30.0], 3);
        let view = fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();
        assert_eq!(view.dim(), (1, 3));
        assert_abs_diff_eq!(view[[0, 2]], 30.0);
    }

    #[test]
    fn fsl_type_mismatch() {
        // Try to interpret f64 data as f32
        let fsl = make_fsl_f64(&[1.0, 2.0], 2);
        let result = fixed_size_list_as_array2::<Float32Type>(&fsl);
        assert!(matches!(result, Err(NdarrowError::InnerTypeMismatch { .. })));
    }

    #[test]
    fn fsl_with_outer_nulls() {
        use arrow_buffer::NullBuffer;
        let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
        let nulls = NullBuffer::from(vec![true, false]); // second row is null
        let fsl = FixedSizeListArray::new(field, 2, Arc::new(values), Some(nulls));
        let result = fixed_size_list_as_array2::<Float64Type>(&fsl);
        assert!(matches!(result, Err(NdarrowError::NullsPresent { .. })));
    }

    #[test]
    fn fsl_unchecked() {
        let fsl = make_fsl_f64(&[1.0, 2.0, 3.0, 4.0], 2);
        let view = unsafe { fixed_size_list_as_array2_unchecked::<Float64Type>(&fsl) };
        assert_eq!(view.dim(), (2, 2));
        assert_abs_diff_eq!(view[[1, 0]], 3.0);
    }

    #[test]
    fn fsl_masked_no_nulls() {
        let fsl = make_fsl_f64(&[1.0, 2.0, 3.0, 4.0], 2);
        let (view, mask) = fixed_size_list_as_array2_masked::<Float64Type>(&fsl).unwrap();
        assert_eq!(view.dim(), (2, 2));
        assert!(mask.is_none());
    }

    #[test]
    fn fsl_masked_with_nulls() {
        use arrow_buffer::NullBuffer;
        let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
        let nulls = NullBuffer::from(vec![true, false]);
        let fsl = FixedSizeListArray::new(field, 2, Arc::new(values), Some(nulls));
        let (view, mask) = fixed_size_list_as_array2_masked::<Float64Type>(&fsl).unwrap();
        assert_eq!(view.dim(), (2, 2));
        assert!(mask.is_some());
    }

    #[test]
    fn fsl_masked_rejects_inner_nulls() {
        let values = Float64Array::from(vec![Some(1.0), None, Some(3.0), Some(4.0)]);
        let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, true));
        let fsl = FixedSizeListArray::new(field, 2, Arc::new(values), None);
        let result = fixed_size_list_as_array2_masked::<Float64Type>(&fsl);
        assert!(matches!(result, Err(NdarrowError::NullsPresent { null_count: 1 })));
    }

    // ─── Zero-copy verification ───

    #[test]
    fn primitive_view_shares_buffer() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Float64Array::from(data);
        let view = arr.as_ndarray().unwrap();
        // The view's data pointer should be within the Arrow buffer's address range.
        let arrow_ptr = arr.values().as_ref().as_ptr();
        let view_ptr = view.as_ptr();
        assert_eq!(arrow_ptr, view_ptr, "view must point to Arrow's buffer");
    }

    #[test]
    fn fsl_view_shares_buffer() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let fsl = make_fsl_f64(&data, 2);
        let view = fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();
        let inner = fsl.values().as_any().downcast_ref::<Float64Array>().unwrap();
        let arrow_ptr = inner.values().as_ref().as_ptr();
        let view_ptr = view.as_ptr();
        assert_eq!(arrow_ptr, view_ptr, "view must point to Arrow's buffer");
    }
}
