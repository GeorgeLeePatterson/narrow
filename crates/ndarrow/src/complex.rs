//! Complex-valued Arrow/ndarray bridge utilities.
//!
//! This module defines custom complex extension types:
//! - `ndarrow.complex32` with storage `FixedSizeList<Float32>(2)`
//! - `ndarrow.complex64` with storage `FixedSizeList<Float64>(2)`
//!
//! Each list element stores `[real, imag]` in row-major order.

use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, PrimitiveArray,
    types::{Float32Type, Float64Type},
};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{ArrowError, DataType, Field, extension::ExtensionType};
use ndarray::{Array1, ArrayView1};
use num_complex::{Complex32, Complex64};

use crate::error::NdarrowError;

fn validate_complex_storage(
    data_type: &DataType,
    expected_inner: &DataType,
    extension_name: &str,
) -> Result<(), ArrowError> {
    match data_type {
        DataType::FixedSizeList(item, size) => {
            if *size != 2 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{extension_name} data type mismatch, expected fixed-size list length 2, found {size}"
                )));
            }
            if !item.data_type().equals_datatype(expected_inner) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{extension_name} data type mismatch, expected inner {expected_inner}, found {}",
                    item.data_type()
                )));
            }
            Ok(())
        }
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "{extension_name} data type mismatch, expected FixedSizeList<{expected_inner}>(2), found {data_type}"
        ))),
    }
}

/// Extension type descriptor for `ndarrow.complex32`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Complex32Extension;

impl ExtensionType for Complex32Extension {
    type Metadata = ();

    const NAME: &'static str = "ndarrow.complex32";

    fn metadata(&self) -> &Self::Metadata {
        &()
    }

    fn serialize_metadata(&self) -> Option<String> {
        None
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        if metadata.is_some() {
            return Err(ArrowError::InvalidArgumentError(
                "ndarrow.complex32 expects no metadata".to_owned(),
            ));
        }
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        validate_complex_storage(data_type, &DataType::Float32, Self::NAME)
    }

    fn try_new(data_type: &DataType, _metadata: Self::Metadata) -> Result<Self, ArrowError> {
        let extension = Self;
        extension.supports_data_type(data_type)?;
        Ok(extension)
    }
}

/// Extension type descriptor for `ndarrow.complex64`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Complex64Extension;

impl ExtensionType for Complex64Extension {
    type Metadata = ();

    const NAME: &'static str = "ndarrow.complex64";

    fn metadata(&self) -> &Self::Metadata {
        &()
    }

    fn serialize_metadata(&self) -> Option<String> {
        None
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        if metadata.is_some() {
            return Err(ArrowError::InvalidArgumentError(
                "ndarrow.complex64 expects no metadata".to_owned(),
            ));
        }
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        validate_complex_storage(data_type, &DataType::Float64, Self::NAME)
    }

    fn try_new(data_type: &DataType, _metadata: Self::Metadata) -> Result<Self, ArrowError> {
        let extension = Self;
        extension.supports_data_type(data_type)?;
        Ok(extension)
    }
}

fn check_field_matches_array(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<(), NdarrowError> {
    if !field.data_type().equals_datatype(array.data_type()) {
        return Err(NdarrowError::TypeMismatch {
            message: format!(
                "field data type ({}) does not match array data type ({})",
                field.data_type(),
                array.data_type()
            ),
        });
    }
    Ok(())
}

fn view_from_complex32_values(
    values: &[f32],
    rows: usize,
) -> Result<ArrayView1<'_, Complex32>, NdarrowError> {
    if values.len() != rows * 2 {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "complex32 storage length mismatch: expected {}, found {}",
                rows * 2,
                values.len()
            ),
        });
    }

    // SAFETY:
    // - `num_complex::Complex32` stores exactly two contiguous `f32` values.
    // - `values` length is validated as `rows * 2`.
    // - We reinterpret immutable bytes; no aliasing violation is introduced.
    let complex_values =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Complex32>(), rows) };
    Ok(ArrayView1::from(complex_values))
}

fn view_from_complex64_values(
    values: &[f64],
    rows: usize,
) -> Result<ArrayView1<'_, Complex64>, NdarrowError> {
    if values.len() != rows * 2 {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "complex64 storage length mismatch: expected {}, found {}",
                rows * 2,
                values.len()
            ),
        });
    }

    // SAFETY:
    // - `num_complex::Complex64` stores exactly two contiguous `f64` values.
    // - `values` length is validated as `rows * 2`.
    // - We reinterpret immutable bytes; no aliasing violation is introduced.
    let complex_values =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Complex64>(), rows) };
    Ok(ArrayView1::from(complex_values))
}

/// Converts `ndarrow.complex32` storage into an `ArrayView1<Complex32>`.
///
/// # Does not allocate
///
/// The returned ndarray view borrows Arrow's primitive values buffer.
///
/// # Errors
///
/// Returns an error on extension/type mismatch, nulls, or storage-shape mismatch.
pub fn complex32_as_array_view1<'a>(
    field: &Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayView1<'a, Complex32>, NdarrowError> {
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    check_field_matches_array(field, array)?;
    let extension = field.try_extension_type::<Complex32Extension>().map_err(NdarrowError::from)?;
    extension.supports_data_type(array.data_type()).map_err(NdarrowError::from)?;

    let values =
        array.values().as_any().downcast_ref::<PrimitiveArray<Float32Type>>().ok_or_else(|| {
            NdarrowError::InnerTypeMismatch {
                message: format!(
                    "expected complex32 inner values as Float32, found {}",
                    array.values().data_type()
                ),
            }
        })?;
    if values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: values.null_count() });
    }

    view_from_complex32_values(values.values().as_ref(), array.len())
}

/// Converts `ndarrow.complex64` storage into an `ArrayView1<Complex64>`.
///
/// # Does not allocate
///
/// The returned ndarray view borrows Arrow's primitive values buffer.
///
/// # Errors
///
/// Returns an error on extension/type mismatch, nulls, or storage-shape mismatch.
pub fn complex64_as_array_view1<'a>(
    field: &Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayView1<'a, Complex64>, NdarrowError> {
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    check_field_matches_array(field, array)?;
    let extension = field.try_extension_type::<Complex64Extension>().map_err(NdarrowError::from)?;
    extension.supports_data_type(array.data_type()).map_err(NdarrowError::from)?;

    let values =
        array.values().as_any().downcast_ref::<PrimitiveArray<Float64Type>>().ok_or_else(|| {
            NdarrowError::InnerTypeMismatch {
                message: format!(
                    "expected complex64 inner values as Float64, found {}",
                    array.values().data_type()
                ),
            }
        })?;
    if values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: values.null_count() });
    }

    view_from_complex64_values(values.values().as_ref(), array.len())
}

fn normalize_array1<T>(array: Array1<T>) -> Result<Vec<T>, NdarrowError>
where
    T: Clone,
{
    let len = array.len();
    let standard =
        if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() };

    let (mut raw_vec, offset) = standard.into_raw_vec_and_offset();
    let start = offset.unwrap_or(0);
    let end = start.checked_add(len).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!(
            "offset + length overflow while normalizing Array1 (offset={start}, len={len})"
        ),
    })?;
    if end > raw_vec.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "offset/length range out of bounds while normalizing Array1 (offset={start}, len={len}, vec_len={})",
                raw_vec.len()
            ),
        });
    }

    if start == 0 {
        raw_vec.truncate(len);
        Ok(raw_vec)
    } else {
        Ok(raw_vec[start..end].to_vec())
    }
}

fn complex32_vec_to_primitive(mut values: Vec<Complex32>) -> Result<Vec<f32>, NdarrowError> {
    let len = values.len();
    let cap = values.capacity();
    let primitive_len = len.checked_mul(2).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!("complex32 length overflow while packing values: len={len}"),
    })?;
    let primitive_cap = cap.checked_mul(2).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!("complex32 capacity overflow while packing values: cap={cap}"),
    })?;
    let ptr = values.as_mut_ptr().cast::<f32>();
    std::mem::forget(values);

    // SAFETY:
    // - `Complex32` is represented by two adjacent `f32` values.
    // - `primitive_len` and `primitive_cap` are exactly 2x the complex counts.
    // - `ptr` comes from a live `Vec<Complex32>` allocation that we intentionally forgot.
    Ok(unsafe { Vec::from_raw_parts(ptr, primitive_len, primitive_cap) })
}

fn complex64_vec_to_primitive(mut values: Vec<Complex64>) -> Result<Vec<f64>, NdarrowError> {
    let len = values.len();
    let cap = values.capacity();
    let primitive_len = len.checked_mul(2).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!("complex64 length overflow while packing values: len={len}"),
    })?;
    let primitive_cap = cap.checked_mul(2).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!("complex64 capacity overflow while packing values: cap={cap}"),
    })?;
    let ptr = values.as_mut_ptr().cast::<f64>();
    std::mem::forget(values);

    // SAFETY:
    // - `Complex64` is represented by two adjacent `f64` values.
    // - `primitive_len` and `primitive_cap` are exactly 2x the complex counts.
    // - `ptr` comes from a live `Vec<Complex64>` allocation that we intentionally forgot.
    Ok(unsafe { Vec::from_raw_parts(ptr, primitive_len, primitive_cap) })
}

/// Converts an owned `Array1<Complex32>` into `ndarrow.complex32` storage.
///
/// # Allocation
///
/// Zero-copy for standard layout + zero offset arrays. Non-standard layout
/// or sliced offsets require one normalization allocation.
///
/// # Errors
///
/// Returns an error when shape metadata cannot be represented.
pub fn array1_complex32_to_extension(
    field_name: &str,
    array: Array1<Complex32>,
) -> Result<(Field, FixedSizeListArray), NdarrowError> {
    let values = complex32_vec_to_primitive(normalize_array1(array)?)?;
    let values_array = PrimitiveArray::<Float32Type>::new(ScalarBuffer::from(values), None);
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let fsl = FixedSizeListArray::new(item_field, 2, Arc::new(values_array), None);

    let mut field = Field::new(field_name, fsl.data_type().clone(), false);
    field.try_with_extension_type(Complex32Extension).map_err(NdarrowError::from)?;

    Ok((field, fsl))
}

/// Converts an owned `Array1<Complex64>` into `ndarrow.complex64` storage.
///
/// # Allocation
///
/// Zero-copy for standard layout + zero offset arrays. Non-standard layout
/// or sliced offsets require one normalization allocation.
///
/// # Errors
///
/// Returns an error when shape metadata cannot be represented.
pub fn array1_complex64_to_extension(
    field_name: &str,
    array: Array1<Complex64>,
) -> Result<(Field, FixedSizeListArray), NdarrowError> {
    let values = complex64_vec_to_primitive(normalize_array1(array)?)?;
    let values_array = PrimitiveArray::<Float64Type>::new(ScalarBuffer::from(values), None);
    let item_field = Arc::new(Field::new("item", DataType::Float64, false));
    let fsl = FixedSizeListArray::new(item_field, 2, Arc::new(values_array), None);

    let mut field = Field::new(field_name, fsl.data_type().clone(), false);
    field.try_with_extension_type(Complex64Extension).map_err(NdarrowError::from)?;

    Ok((field, fsl))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use approx::assert_abs_diff_eq;
    use arrow_buffer::NullBuffer;
    use ndarray::{Array1, s};

    use super::*;

    fn field_with_extension_name(name: &str, data_type: DataType) -> Field {
        let mut metadata = HashMap::new();
        metadata.insert("ARROW:extension:name".to_owned(), name.to_owned());
        Field::new("manual", data_type, false).with_metadata(metadata)
    }

    #[test]
    fn complex_extensions_support_expected_storage() {
        let complex32_storage = DataType::new_fixed_size_list(DataType::Float32, 2, false);
        let complex64_storage = DataType::new_fixed_size_list(DataType::Float64, 2, false);

        assert!(Complex32Extension.supports_data_type(&complex32_storage).is_ok());
        assert!(Complex64Extension.supports_data_type(&complex64_storage).is_ok());
    }

    #[test]
    fn complex_extensions_reject_invalid_storage_shapes_and_types() {
        let bad_len = DataType::new_fixed_size_list(DataType::Float32, 3, false);
        let bad_inner = DataType::new_fixed_size_list(DataType::Float64, 2, false);
        let bad_top_level = DataType::Float32;

        assert!(Complex32Extension.supports_data_type(&bad_len).is_err());
        assert!(Complex32Extension.supports_data_type(&bad_inner).is_err());
        assert!(Complex64Extension.supports_data_type(&bad_top_level).is_err());
    }

    #[test]
    fn complex_extensions_reject_metadata_payload() {
        assert!(Complex32Extension::deserialize_metadata(Some("unexpected")).is_err());
        assert!(Complex64Extension::deserialize_metadata(Some("unexpected")).is_err());
    }

    #[test]
    fn complex32_roundtrip_zero_copy() {
        let values = vec![
            Complex32::new(1.0_f32, -2.0),
            Complex32::new(0.5, 4.25),
            Complex32::new(-1.25, 0.0),
        ];
        let array = Array1::from_vec(values.clone());
        let original_ptr = array.as_ptr();

        let (field, storage) =
            array1_complex32_to_extension("c32", array).expect("complex32 outbound should succeed");
        let view =
            complex32_as_array_view1(&field, &storage).expect("complex32 inbound should succeed");

        assert_eq!(view.len(), values.len());
        assert_eq!(view.as_ptr(), original_ptr);
        for (actual, expected) in view.iter().zip(values.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re);
            assert_abs_diff_eq!(actual.im, expected.im);
        }
    }

    #[test]
    fn complex64_roundtrip_zero_copy() {
        let values = vec![
            Complex64::new(1.0_f64, -2.0),
            Complex64::new(0.5, 4.25),
            Complex64::new(-1.25, 0.0),
        ];
        let array = Array1::from_vec(values.clone());
        let original_ptr = array.as_ptr();

        let (field, storage) =
            array1_complex64_to_extension("c64", array).expect("complex64 outbound should succeed");
        let view =
            complex64_as_array_view1(&field, &storage).expect("complex64 inbound should succeed");

        assert_eq!(view.len(), values.len());
        assert_eq!(view.as_ptr(), original_ptr);
        for (actual, expected) in view.iter().zip(values.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re);
            assert_abs_diff_eq!(actual.im, expected.im);
        }
    }

    #[test]
    fn complex64_roundtrip_from_offset_array() {
        let base = Array1::from_vec(vec![
            Complex64::new(10.0_f64, 1.0),
            Complex64::new(20.0, 2.0),
            Complex64::new(30.0, 3.0),
            Complex64::new(40.0, 4.0),
        ]);
        let sliced = base.slice_move(s![1..3]);
        let expected = sliced.iter().copied().collect::<Vec<_>>();

        let (field, storage) = array1_complex64_to_extension("c64", sliced)
            .expect("complex64 outbound should succeed for sliced arrays");
        let view =
            complex64_as_array_view1(&field, &storage).expect("complex64 inbound should succeed");

        assert_eq!(view.len(), expected.len());
        for (actual, expected) in view.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re);
            assert_abs_diff_eq!(actual.im, expected.im);
        }
    }

    #[test]
    fn complex32_rejects_outer_nulls() {
        let values = PrimitiveArray::<Float32Type>::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let storage = FixedSizeListArray::new(
            item,
            2,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, false])),
        );

        let mut field = Field::new("c32", storage.data_type().clone(), false);
        field
            .try_with_extension_type(Complex32Extension)
            .expect("field extension attachment should succeed");

        let err = complex32_as_array_view1(&field, &storage).expect_err("outer nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn complex64_rejects_inner_nulls() {
        let values =
            PrimitiveArray::<Float64Type>::from(vec![Some(1.0_f64), None, Some(3.0), Some(4.0)]);
        let item = Arc::new(Field::new("item", DataType::Float64, true));
        let storage = FixedSizeListArray::new(item, 2, Arc::new(values), None);

        let mut field = Field::new("c64", storage.data_type().clone(), false);
        field
            .try_with_extension_type(Complex64Extension)
            .expect("field extension attachment should succeed");

        let err = complex64_as_array_view1(&field, &storage).expect_err("inner nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn complex32_rejects_missing_extension_metadata() {
        let values = PrimitiveArray::<Float32Type>::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let storage = FixedSizeListArray::new(item, 2, Arc::new(values), None);
        let field = Field::new("c32", storage.data_type().clone(), false);

        let err = complex32_as_array_view1(&field, &storage)
            .expect_err("missing extension metadata should fail");
        assert!(matches!(err, NdarrowError::Arrow(_)));
    }

    #[test]
    fn complex32_rejects_field_array_type_mismatch() {
        let values = PrimitiveArray::<Float32Type>::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let storage = FixedSizeListArray::new(item, 2, Arc::new(values), None);
        let field =
            Field::new("c32", DataType::new_fixed_size_list(DataType::Float64, 2, false), false);

        let err = complex32_as_array_view1(&field, &storage).expect_err("type mismatch must fail");
        assert!(matches!(err, NdarrowError::TypeMismatch { .. }));
    }

    #[test]
    fn complex32_rejects_invalid_extension_storage() {
        let values = PrimitiveArray::<Float32Type>::from(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let storage = FixedSizeListArray::new(item, 3, Arc::new(values), None);
        let field =
            field_with_extension_name(Complex32Extension::NAME, storage.data_type().clone());

        let err = complex32_as_array_view1(&field, &storage)
            .expect_err("invalid complex storage should fail validation");
        assert!(matches!(err, NdarrowError::Arrow(_)));
    }

    #[test]
    fn complex_view_helpers_reject_bad_storage_lengths() {
        let err32 = view_from_complex32_values(&[1.0_f32, 2.0, 3.0], 2)
            .expect_err("length mismatch must fail");
        let err64 = view_from_complex64_values(&[1.0_f64, 2.0, 3.0], 2)
            .expect_err("length mismatch must fail");
        assert!(matches!(err32, NdarrowError::ShapeMismatch { .. }));
        assert!(matches!(err64, NdarrowError::ShapeMismatch { .. }));
    }
}
