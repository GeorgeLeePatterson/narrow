//! Complex-valued Arrow/ndarray bridge utilities.
//!
//! This module defines custom complex extension types:
//! - `ndarrow.complex32` with storage `FixedSizeList<Float32>(2)`
//! - `ndarrow.complex64` with storage `FixedSizeList<Float64>(2)`
//!
//! Each list element stores `[real, imag]` in row-major order.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int32Array, ListArray, PrimitiveArray, StructArray,
    types::{Float32Type, Float64Type},
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{
    ArrowError, DataType, Field,
    extension::{
        EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY, ExtensionType, FixedShapeTensor,
        VariableShapeTensor,
    },
};
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, IxDyn};
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

fn slice_from_complex32_values(values: &[f32], rows: usize) -> Result<&[Complex32], NdarrowError> {
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
    Ok(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Complex32>(), rows) })
}

fn view_from_complex32_values(
    values: &[f32],
    rows: usize,
) -> Result<ArrayView1<'_, Complex32>, NdarrowError> {
    let complex_values = slice_from_complex32_values(values, rows)?;
    Ok(ArrayView1::from(complex_values))
}

fn slice_from_complex64_values(values: &[f64], rows: usize) -> Result<&[Complex64], NdarrowError> {
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
    Ok(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Complex64>(), rows) })
}

fn view_from_complex64_values(
    values: &[f64],
    rows: usize,
) -> Result<ArrayView1<'_, Complex64>, NdarrowError> {
    let complex_values = slice_from_complex64_values(values, rows)?;
    Ok(ArrayView1::from(complex_values))
}

fn nested_complex_storage<'a>(
    array: &'a FixedSizeListArray,
    context: &str,
) -> Result<(&'a Field, &'a FixedSizeListArray, usize), NdarrowError> {
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    let cols = usize::try_from(array.value_length()).map_err(|_| NdarrowError::ShapeMismatch {
        message: format!(
            "negative Arrow fixed-size list value length for {context}: {}",
            array.value_length()
        ),
    })?;

    let inner_field = match array.data_type() {
        DataType::FixedSizeList(item, _) => item.as_ref(),
        data_type => {
            return Err(NdarrowError::TypeMismatch {
                message: format!(
                    "expected nested complex {context} storage as FixedSizeList, found {data_type}"
                ),
            });
        }
    };
    let inner_array = array.values().as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(
        || NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected nested complex {context} inner storage as FixedSizeListArray, found {}",
                array.values().data_type()
            ),
        },
    )?;

    Ok((inner_field, inner_array, cols))
}

fn normalize_array2<T>(array: Array2<T>) -> Result<(usize, Vec<T>), NdarrowError>
where
    T: Clone,
{
    let (rows, cols) = array.dim();
    let total = rows.checked_mul(cols).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!("matrix element count overflows usize: ({rows}, {cols})"),
    })?;
    let standard =
        if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() };
    let (mut raw_vec, offset) = standard.into_raw_vec_and_offset();
    let start = offset.unwrap_or(0);
    let end = start.checked_add(total).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!(
            "offset + matrix element count overflow while normalizing Array2 (offset={start}, total={total})"
        ),
    })?;
    if end > raw_vec.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "offset/length range out of bounds while normalizing Array2 (offset={start}, total={total}, vec_len={})",
                raw_vec.len()
            ),
        });
    }

    let vec = if start == 0 {
        raw_vec.truncate(total);
        raw_vec
    } else {
        raw_vec[start..end].to_vec()
    };

    Ok((cols, vec))
}

fn fixed_shape_tensor_field(
    field_name: &str,
    storage_type: DataType,
    value_type: DataType,
    tensor_shape: &[usize],
) -> Result<Field, NdarrowError> {
    let extension = FixedShapeTensor::try_new(value_type, tensor_shape.to_vec(), None, None)
        .map_err(NdarrowError::from)?;
    extension.supports_data_type(&storage_type).map_err(NdarrowError::from)?;

    let metadata_json = serde_json::json!({ "shape": tensor_shape }).to_string();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(EXTENSION_TYPE_NAME_KEY.to_owned(), FixedShapeTensor::NAME.to_owned());
    metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), metadata_json);
    Ok(Field::new(field_name, storage_type, false).with_metadata(metadata))
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

/// Converts nested `FixedSizeList<ndarrow.complex32>(D)` storage into an
/// `ArrayView2<Complex32>`.
///
/// The outer fixed-size list represents the vector dimension `D`, while each
/// inner element is one `ndarrow.complex32` scalar.
///
/// # Does not allocate
///
/// The returned ndarray view borrows Arrow's primitive values buffer through
/// the existing `ndarrow.complex32` scalar extension storage.
///
/// # Errors
///
/// Returns an error on type mismatch, nulls, or storage-shape mismatch.
pub fn complex32_as_array_view2(
    array: &FixedSizeListArray,
) -> Result<ArrayView2<'_, Complex32>, NdarrowError> {
    let (inner_field, inner_array, cols) = nested_complex_storage(array, "matrix")?;
    let flat = complex32_as_array_view1(inner_field, inner_array)?;
    flat.into_shape_with_order((array.len(), cols)).map_err(NdarrowError::from)
}

/// Converts nested `FixedSizeList<ndarrow.complex64>(D)` storage into an
/// `ArrayView2<Complex64>`.
///
/// The outer fixed-size list represents the vector dimension `D`, while each
/// inner element is one `ndarrow.complex64` scalar.
///
/// # Does not allocate
///
/// The returned ndarray view borrows Arrow's primitive values buffer through
/// the existing `ndarrow.complex64` scalar extension storage.
///
/// # Errors
///
/// Returns an error on type mismatch, nulls, or storage-shape mismatch.
pub fn complex64_as_array_view2(
    array: &FixedSizeListArray,
) -> Result<ArrayView2<'_, Complex64>, NdarrowError> {
    let (inner_field, inner_array, cols) = nested_complex_storage(array, "matrix")?;
    let flat = complex64_as_array_view1(inner_field, inner_array)?;
    flat.into_shape_with_order((array.len(), cols)).map_err(NdarrowError::from)
}

/// Converts `arrow.fixed_shape_tensor` storage with `ndarrow.complex32` element
/// type into an `ArrayViewD<Complex32>`.
///
/// # Does not allocate
///
/// The returned ndarray view borrows Arrow's underlying complex scalar storage.
///
/// # Errors
///
/// Returns an error on nulls, type mismatches, invalid extension metadata, or
/// shape incompatibility.
pub fn complex32_fixed_shape_tensor_as_array_viewd<'a>(
    field: &Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayViewD<'a, Complex32>, NdarrowError> {
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }
    if !field.data_type().equals_datatype(array.data_type()) {
        return Err(NdarrowError::TypeMismatch {
            message: format!(
                "field data type ({}) does not match array data type ({})",
                field.data_type(),
                array.data_type()
            ),
        });
    }
    let tensor_shape = crate::tensor::parse_fixed_shape_metadata(field)?;
    let element_count = tensor_shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| NdarrowError::ShapeMismatch {
            message: format!("tensor shape product overflows usize: {tensor_shape:?}"),
        })?;
    let stored =
        usize::try_from(array.value_length()).map_err(|_| NdarrowError::ShapeMismatch {
            message: format!(
                "negative Arrow fixed-size list value length: {}",
                array.value_length()
            ),
        })?;
    if stored != element_count {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "fixed-shape complex tensor element count mismatch: metadata shape {tensor_shape:?} implies {element_count}, array stores {stored}",
            ),
        });
    }

    let (inner_field, inner_array, _) = nested_complex_storage(array, "fixed-shape tensor")?;
    let flat = complex32_as_array_view1(inner_field, inner_array)?;
    let mut full_shape = Vec::with_capacity(tensor_shape.len() + 1);
    full_shape.push(array.len());
    full_shape.extend_from_slice(&tensor_shape);
    flat.into_shape_with_order(IxDyn(&full_shape)).map_err(NdarrowError::from)
}

/// Converts `arrow.fixed_shape_tensor` storage with `ndarrow.complex64` element
/// type into an `ArrayViewD<Complex64>`.
///
/// # Does not allocate
///
/// The returned ndarray view borrows Arrow's underlying complex scalar storage.
///
/// # Errors
///
/// Returns an error on nulls, type mismatches, invalid extension metadata, or
/// shape incompatibility.
pub fn complex64_fixed_shape_tensor_as_array_viewd<'a>(
    field: &Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayViewD<'a, Complex64>, NdarrowError> {
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }
    if !field.data_type().equals_datatype(array.data_type()) {
        return Err(NdarrowError::TypeMismatch {
            message: format!(
                "field data type ({}) does not match array data type ({})",
                field.data_type(),
                array.data_type()
            ),
        });
    }
    let tensor_shape = crate::tensor::parse_fixed_shape_metadata(field)?;
    let element_count = tensor_shape
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| NdarrowError::ShapeMismatch {
            message: format!("tensor shape product overflows usize: {tensor_shape:?}"),
        })?;
    let stored =
        usize::try_from(array.value_length()).map_err(|_| NdarrowError::ShapeMismatch {
            message: format!(
                "negative Arrow fixed-size list value length: {}",
                array.value_length()
            ),
        })?;
    if stored != element_count {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "fixed-shape complex tensor element count mismatch: metadata shape {tensor_shape:?} implies {element_count}, array stores {stored}",
            ),
        });
    }

    let (inner_field, inner_array, _) = nested_complex_storage(array, "fixed-shape tensor")?;
    let flat = complex64_as_array_view1(inner_field, inner_array)?;
    let mut full_shape = Vec::with_capacity(tensor_shape.len() + 1);
    full_shape.push(array.len());
    full_shape.extend_from_slice(&tensor_shape);
    flat.into_shape_with_order(IxDyn(&full_shape)).map_err(NdarrowError::from)
}

/// Iterator over per-row complex tensor views for
/// `arrow.variable_shape_tensor<ndarrow.complex32>`.
pub struct Complex32VariableShapeTensorIter<'a> {
    rows:        crate::tensor::VariableShapeTensorRowCursor<'a>,
    flat_values: &'a [Complex32],
}

impl<'a> Iterator for Complex32VariableShapeTensorIter<'a> {
    type Item = Result<(usize, ArrayViewD<'a, Complex32>), NdarrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let row = match self.rows.next_row() {
            Some(Ok(row)) => row,
            Some(Err(err)) => return Some(Err(err)),
            None => return None,
        };

        let slice = &self.flat_values[row.start..row.end];
        let view = ArrayViewD::from_shape(IxDyn(&row.shape), slice).map_err(NdarrowError::from);
        Some(view.map(|view| (row.row, view)))
    }
}

/// Iterator over per-row complex tensor views for
/// `arrow.variable_shape_tensor<ndarrow.complex64>`.
pub struct Complex64VariableShapeTensorIter<'a> {
    rows:        crate::tensor::VariableShapeTensorRowCursor<'a>,
    flat_values: &'a [Complex64],
}

impl<'a> Iterator for Complex64VariableShapeTensorIter<'a> {
    type Item = Result<(usize, ArrayViewD<'a, Complex64>), NdarrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let row = match self.rows.next_row() {
            Some(Ok(row)) => row,
            Some(Err(err)) => return Some(Err(err)),
            None => return None,
        };

        let slice = &self.flat_values[row.start..row.end];
        let view = ArrayViewD::from_shape(IxDyn(&row.shape), slice).map_err(NdarrowError::from);
        Some(view.map(|view| (row.row, view)))
    }
}

/// Creates an iterator over per-row zero-copy complex tensor views for
/// `arrow.variable_shape_tensor<ndarrow.complex32>`.
///
/// # Does not allocate
///
/// This borrows Arrow values buffers directly through the complex scalar
/// carrier.
///
/// # Errors
///
/// Returns an error when extension/type/null invariants are violated.
pub fn complex32_variable_shape_tensor_iter<'a>(
    field: &Field,
    array: &'a StructArray,
) -> Result<Complex32VariableShapeTensorIter<'a>, NdarrowError> {
    let storage = crate::tensor::variable_shape_tensor_storage(field, array)?;
    let data_values = storage
        .data
        .values()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected variable complex32 tensor data values as FixedSizeListArray, found {}",
                storage.data.values().data_type()
            ),
        })?;
    let _validated = complex32_as_array_view1(storage.data_item_field, data_values)?;
    let primitive_values =
        data_values.values().as_any().downcast_ref::<PrimitiveArray<Float32Type>>().ok_or_else(
            || NdarrowError::InnerTypeMismatch {
                message: format!(
                    "expected complex32 variable tensor child values as Float32, found {}",
                    data_values.values().data_type()
                ),
            },
        )?;
    let flat_values =
        slice_from_complex32_values(primitive_values.values().as_ref(), data_values.len())?;

    Ok(Complex32VariableShapeTensorIter { rows: storage.row_cursor(), flat_values })
}

/// Creates an iterator over per-row zero-copy complex tensor views for
/// `arrow.variable_shape_tensor<ndarrow.complex64>`.
///
/// # Does not allocate
///
/// This borrows Arrow values buffers directly through the complex scalar
/// carrier.
///
/// # Errors
///
/// Returns an error when extension/type/null invariants are violated.
pub fn complex64_variable_shape_tensor_iter<'a>(
    field: &Field,
    array: &'a StructArray,
) -> Result<Complex64VariableShapeTensorIter<'a>, NdarrowError> {
    let storage = crate::tensor::variable_shape_tensor_storage(field, array)?;
    let data_values = storage
        .data
        .values()
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected variable complex64 tensor data values as FixedSizeListArray, found {}",
                storage.data.values().data_type()
            ),
        })?;
    let _validated = complex64_as_array_view1(storage.data_item_field, data_values)?;
    let primitive_values =
        data_values.values().as_any().downcast_ref::<PrimitiveArray<Float64Type>>().ok_or_else(
            || NdarrowError::InnerTypeMismatch {
                message: format!(
                    "expected complex64 variable tensor child values as Float64, found {}",
                    data_values.values().data_type()
                ),
            },
        )?;
    let flat_values =
        slice_from_complex64_values(primitive_values.values().as_ref(), data_values.len())?;

    Ok(Complex64VariableShapeTensorIter { rows: storage.row_cursor(), flat_values })
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

fn append_complex_row_values<T>(
    array: ArrayD<T>,
    row: usize,
    packed_values: &mut Vec<T>,
) -> Result<i32, NdarrowError>
where
    T: Clone,
{
    let standard =
        if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() };
    let element_count = standard.len();
    let (raw_vec, offset) = standard.into_raw_vec_and_offset();
    let start = offset.unwrap_or(0);
    let end = start.checked_add(element_count).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!(
            "offset + element count overflow while packing complex tensor row {row} (offset={start}, len={element_count})"
        ),
    })?;
    if end > raw_vec.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "offset/length range out of bounds while packing complex tensor row {row} (offset={start}, len={element_count}, vec_len={})",
                raw_vec.len()
            ),
        });
    }

    packed_values.extend_from_slice(&raw_vec[start..end]);

    i32::try_from(element_count).map_err(|_| NdarrowError::ShapeMismatch {
        message: format!(
            "complex tensor row element count exceeds i32 limits at row {row}: {element_count}"
        ),
    })
}

type ComplexScalarBuilder<T> =
    fn(&str, Array1<T>) -> Result<(Field, FixedSizeListArray), NdarrowError>;

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

/// Converts an owned `Array2<Complex32>` into nested
/// `FixedSizeList<ndarrow.complex32>(N)` storage.
///
/// # Allocation
///
/// Zero-copy for standard-layout arrays with zero offset. Non-standard layout
/// or sliced offsets require one normalization allocation.
///
/// # Errors
///
/// Returns an error when shape metadata cannot be represented.
pub fn array2_complex32_to_fixed_size_list(
    array: Array2<Complex32>,
) -> Result<FixedSizeListArray, NdarrowError> {
    let (cols, values) = normalize_array2(array)?;
    let (item_field, item_array) = array1_complex32_to_extension("item", Array1::from_vec(values))?;
    let value_length = i32::try_from(cols).map_err(|_| NdarrowError::ShapeMismatch {
        message: format!("matrix column count {cols} exceeds Arrow i32 value_length limits"),
    })?;
    Ok(FixedSizeListArray::new(Arc::new(item_field), value_length, Arc::new(item_array), None))
}

/// Converts an owned `Array2<Complex64>` into nested
/// `FixedSizeList<ndarrow.complex64>(N)` storage.
///
/// # Allocation
///
/// Zero-copy for standard-layout arrays with zero offset. Non-standard layout
/// or sliced offsets require one normalization allocation.
///
/// # Errors
///
/// Returns an error when shape metadata cannot be represented.
pub fn array2_complex64_to_fixed_size_list(
    array: Array2<Complex64>,
) -> Result<FixedSizeListArray, NdarrowError> {
    let (cols, values) = normalize_array2(array)?;
    let (item_field, item_array) = array1_complex64_to_extension("item", Array1::from_vec(values))?;
    let value_length = i32::try_from(cols).map_err(|_| NdarrowError::ShapeMismatch {
        message: format!("matrix column count {cols} exceeds Arrow i32 value_length limits"),
    })?;
    Ok(FixedSizeListArray::new(Arc::new(item_field), value_length, Arc::new(item_array), None))
}

/// Converts an owned complex tensor batch into `arrow.fixed_shape_tensor` with
/// `ndarrow.complex32` element storage.
///
/// The first ndarray axis is interpreted as batch dimension. Remaining axes are
/// stored in the fixed-shape tensor metadata.
///
/// # Allocation
///
/// Zero-copy for standard-layout arrays with zero offset. Non-standard layout
/// or sliced offsets require one normalization allocation.
///
/// # Errors
///
/// Returns an error on invalid shape or Arrow metadata constraints.
pub fn arrayd_complex32_to_fixed_shape_tensor(
    field_name: &str,
    array: ArrayD<Complex32>,
) -> Result<(Field, FixedSizeListArray), NdarrowError> {
    let shape = array.shape().to_vec();
    if shape.is_empty() {
        return Err(NdarrowError::ShapeMismatch {
            message: "ArrayD must have at least one dimension (batch axis)".to_owned(),
        });
    }

    let batch = shape[0];
    let tensor_shape = shape[1..].to_vec();
    let element_count = if tensor_shape.is_empty() {
        1
    } else {
        tensor_shape.iter().try_fold(1_usize, |acc, dim| acc.checked_mul(*dim)).ok_or_else(
            || NdarrowError::ShapeMismatch {
                message: format!("tensor shape product overflows usize: {tensor_shape:?}"),
            },
        )?
    };
    let expected_len =
        batch.checked_mul(element_count).ok_or_else(|| NdarrowError::ShapeMismatch {
            message: format!("batch × tensor_shape product overflows usize: {shape:?}"),
        })?;
    if expected_len != array.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "array length ({}) does not match batch × tensor element count ({expected_len})",
                array.len()
            ),
        });
    }

    let reshaped =
        array.into_shape_with_order((batch, element_count)).map_err(NdarrowError::from)?;
    let fsl = array2_complex32_to_fixed_size_list(reshaped)?;
    let field = fixed_shape_tensor_field(
        field_name,
        fsl.data_type().clone(),
        fsl.value_type().clone(),
        &tensor_shape,
    )?;
    Ok((field, fsl))
}

/// Converts an owned complex tensor batch into `arrow.fixed_shape_tensor` with
/// `ndarrow.complex64` element storage.
///
/// The first ndarray axis is interpreted as batch dimension. Remaining axes are
/// stored in the fixed-shape tensor metadata.
///
/// # Allocation
///
/// Zero-copy for standard-layout arrays with zero offset. Non-standard layout
/// or sliced offsets require one normalization allocation.
///
/// # Errors
///
/// Returns an error on invalid shape or Arrow metadata constraints.
pub fn arrayd_complex64_to_fixed_shape_tensor(
    field_name: &str,
    array: ArrayD<Complex64>,
) -> Result<(Field, FixedSizeListArray), NdarrowError> {
    let shape = array.shape().to_vec();
    if shape.is_empty() {
        return Err(NdarrowError::ShapeMismatch {
            message: "ArrayD must have at least one dimension (batch axis)".to_owned(),
        });
    }

    let batch = shape[0];
    let tensor_shape = shape[1..].to_vec();
    let element_count = if tensor_shape.is_empty() {
        1
    } else {
        tensor_shape.iter().try_fold(1_usize, |acc, dim| acc.checked_mul(*dim)).ok_or_else(
            || NdarrowError::ShapeMismatch {
                message: format!("tensor shape product overflows usize: {tensor_shape:?}"),
            },
        )?
    };
    let expected_len =
        batch.checked_mul(element_count).ok_or_else(|| NdarrowError::ShapeMismatch {
            message: format!("batch × tensor_shape product overflows usize: {shape:?}"),
        })?;
    if expected_len != array.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "array length ({}) does not match batch × tensor element count ({expected_len})",
                array.len()
            ),
        });
    }

    let reshaped =
        array.into_shape_with_order((batch, element_count)).map_err(NdarrowError::from)?;
    let fsl = array2_complex64_to_fixed_size_list(reshaped)?;
    let field = fixed_shape_tensor_field(
        field_name,
        fsl.data_type().clone(),
        fsl.value_type().clone(),
        &tensor_shape,
    )?;
    Ok((field, fsl))
}

fn arrays_complex_to_variable_shape_tensor<T>(
    field_name: &str,
    arrays: Vec<ArrayD<T>>,
    uniform_shape: Option<Vec<Option<i32>>>,
    scalar_builder: ComplexScalarBuilder<T>,
) -> Result<(Field, StructArray), NdarrowError>
where
    T: Clone,
{
    if arrays.is_empty() {
        return Err(NdarrowError::InvalidMetadata {
            message: "arrays_to_variable_shape_tensor requires at least one tensor".to_owned(),
        });
    }

    let dimensions = arrays[0].ndim();
    if let Some(uniform) = &uniform_shape {
        if uniform.len() != dimensions {
            return Err(NdarrowError::InvalidMetadata {
                message: format!(
                    "uniform_shape length mismatch: expected {dimensions}, found {}",
                    uniform.len()
                ),
            });
        }
    }

    let mut offsets = Vec::with_capacity(arrays.len() + 1);
    offsets.push(0_i32);
    let mut running_offset = 0_i32;
    let mut packed_values = Vec::new();
    let mut packed_shapes = Vec::with_capacity(arrays.len() * dimensions);

    for (row, array) in arrays.into_iter().enumerate() {
        if array.ndim() != dimensions {
            return Err(NdarrowError::ShapeMismatch {
                message: format!(
                    "all tensors must share rank {dimensions}; row {row} has rank {}",
                    array.ndim()
                ),
            });
        }

        crate::tensor::push_tensor_shape(
            array.shape(),
            row,
            uniform_shape.as_deref(),
            &mut packed_shapes,
        )?;
        let element_count_i32 = append_complex_row_values(array, row, &mut packed_values)?;
        running_offset = running_offset.checked_add(element_count_i32).ok_or_else(|| {
            NdarrowError::ShapeMismatch {
                message: "packed variable tensor offsets exceed i32 limits".to_owned(),
            }
        })?;
        offsets.push(running_offset);
    }

    let (data_item_field, data_item_array) =
        scalar_builder("item", Array1::from_vec(packed_values))?;
    let data_list: ArrayRef = Arc::new(ListArray::new(
        Arc::new(data_item_field.clone()),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(data_item_array),
        None,
    ));

    let shape_values = Int32Array::new(ScalarBuffer::from(packed_shapes), None);
    let shape_fsl: ArrayRef = Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Int32, false)),
        i32::try_from(dimensions).map_err(|_| NdarrowError::ShapeMismatch {
            message: format!("tensor rank exceeds i32 limits: {dimensions}"),
        })?,
        Arc::new(shape_values),
        None,
    ));

    let struct_fields = vec![
        Field::new("data", data_list.data_type().clone(), false),
        Field::new("shape", shape_fsl.data_type().clone(), false),
    ];
    let struct_array =
        StructArray::new(struct_fields.clone().into(), vec![data_list, shape_fsl], None);

    let extension = VariableShapeTensor::try_new(
        data_item_field.data_type().clone(),
        dimensions,
        None,
        None,
        uniform_shape,
    )
    .map_err(NdarrowError::from)?;
    extension.supports_data_type(struct_array.data_type()).map_err(NdarrowError::from)?;
    let mut field = Field::new(field_name, struct_array.data_type().clone(), false);
    field.try_with_extension_type(extension).map_err(NdarrowError::from)?;

    Ok((field, struct_array))
}

/// Converts owned complex tensors into `arrow.variable_shape_tensor` storage
/// with `ndarrow.complex32` element storage.
///
/// # Allocation
///
/// This packs ragged tensor rows into Arrow list/shape buffers and therefore
/// allocates.
///
/// # Errors
///
/// Returns an error when tensor ranks differ or metadata constraints fail.
pub fn arrays_complex32_to_variable_shape_tensor(
    field_name: &str,
    arrays: Vec<ArrayD<Complex32>>,
    uniform_shape: Option<Vec<Option<i32>>>,
) -> Result<(Field, StructArray), NdarrowError> {
    arrays_complex_to_variable_shape_tensor(
        field_name,
        arrays,
        uniform_shape,
        array1_complex32_to_extension,
    )
}

/// Converts owned complex tensors into `arrow.variable_shape_tensor` storage
/// with `ndarrow.complex64` element storage.
///
/// # Allocation
///
/// This packs ragged tensor rows into Arrow list/shape buffers and therefore
/// allocates.
///
/// # Errors
///
/// Returns an error when tensor ranks differ or metadata constraints fail.
pub fn arrays_complex64_to_variable_shape_tensor(
    field_name: &str,
    arrays: Vec<ArrayD<Complex64>>,
    uniform_shape: Option<Vec<Option<i32>>>,
) -> Result<(Field, StructArray), NdarrowError> {
    arrays_complex_to_variable_shape_tensor(
        field_name,
        arrays,
        uniform_shape,
        array1_complex64_to_extension,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use approx::assert_abs_diff_eq;
    use arrow_buffer::NullBuffer;
    use ndarray::{Array1, ArrayD, IxDyn, array, s};

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
    fn complex_extensions_expose_no_serialized_metadata() {
        assert_eq!(Complex32Extension.metadata(), &());
        assert_eq!(Complex64Extension.metadata(), &());
        assert_eq!(Complex32Extension.serialize_metadata(), None);
        assert_eq!(Complex64Extension.serialize_metadata(), None);
    }

    #[test]
    fn complex_extensions_try_new_accept_matching_storage() {
        let complex32_storage = DataType::new_fixed_size_list(DataType::Float32, 2, false);
        let complex64_storage = DataType::new_fixed_size_list(DataType::Float64, 2, false);

        assert!(Complex32Extension::try_new(&complex32_storage, ()).is_ok());
        assert!(Complex64Extension::try_new(&complex64_storage, ()).is_ok());
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
    fn complex32_matrix_roundtrip_zero_copy() {
        let matrix = array![
            [Complex32::new(1.0_f32, -1.0), Complex32::new(2.5, 0.25)],
            [Complex32::new(-3.0, 0.5), Complex32::new(4.0, -2.0)],
        ];
        let original_ptr = matrix.as_ptr();

        let storage = array2_complex32_to_fixed_size_list(matrix)
            .expect("complex32 matrix outbound should succeed");
        let view =
            complex32_as_array_view2(&storage).expect("complex32 matrix inbound should succeed");

        assert_eq!(view.dim(), (2, 2));
        assert_eq!(view.as_ptr(), original_ptr);
        assert_abs_diff_eq!(view[(0, 0)].re, 1.0_f32);
        assert_abs_diff_eq!(view[(0, 0)].im, -1.0_f32);
        assert_abs_diff_eq!(view[(1, 1)].re, 4.0_f32);
        assert_abs_diff_eq!(view[(1, 1)].im, -2.0_f32);
    }

    #[test]
    fn complex64_matrix_roundtrip_zero_copy() {
        let matrix = array![
            [Complex64::new(1.0_f64, -1.0), Complex64::new(2.5, 0.25)],
            [Complex64::new(-3.0, 0.5), Complex64::new(4.0, -2.0)],
        ];
        let original_ptr = matrix.as_ptr();

        let storage = array2_complex64_to_fixed_size_list(matrix)
            .expect("complex64 matrix outbound should succeed");
        let view =
            complex64_as_array_view2(&storage).expect("complex64 matrix inbound should succeed");

        assert_eq!(view.dim(), (2, 2));
        assert_eq!(view.as_ptr(), original_ptr);
        assert_abs_diff_eq!(view[(0, 1)].re, 2.5_f64);
        assert_abs_diff_eq!(view[(0, 1)].im, 0.25_f64);
        assert_abs_diff_eq!(view[(1, 0)].re, -3.0_f64);
        assert_abs_diff_eq!(view[(1, 0)].im, 0.5_f64);
    }

    #[test]
    fn complex64_matrix_rejects_non_nested_storage() {
        let values = PrimitiveArray::<Float64Type>::from(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let item = Arc::new(Field::new("item", DataType::Float64, false));
        let storage = FixedSizeListArray::new(item, 2, Arc::new(values), None);

        let err = complex64_as_array_view2(&storage).expect_err("non-nested storage must fail");
        assert!(matches!(err, NdarrowError::InnerTypeMismatch { .. }));
    }

    #[test]
    fn complex32_matrix_rejects_non_nested_storage() {
        let values = PrimitiveArray::<Float32Type>::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let storage = FixedSizeListArray::new(item, 2, Arc::new(values), None);

        let err = complex32_as_array_view2(&storage).expect_err("non-nested storage must fail");
        assert!(matches!(err, NdarrowError::InnerTypeMismatch { .. }));
    }

    #[test]
    fn complex32_matrix_rejects_outer_nulls() {
        let matrix = array![
            [Complex32::new(1.0_f32, -1.0), Complex32::new(2.5, 0.25)],
            [Complex32::new(-3.0, 0.5), Complex32::new(4.0, -2.0)],
        ];
        let storage = array2_complex32_to_fixed_size_list(matrix)
            .expect("complex32 matrix outbound should succeed");
        let item = match storage.data_type() {
            DataType::FixedSizeList(item, _) => Arc::clone(item),
            data_type => panic!("expected fixed-size list storage, found {data_type}"),
        };
        let values = storage.values();
        let bad_storage = FixedSizeListArray::new(
            item,
            storage.value_length(),
            Arc::clone(values),
            Some(NullBuffer::from(vec![true, false])),
        );

        let err = complex32_as_array_view2(&bad_storage).expect_err("outer nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn complex64_matrix_rejects_outer_nulls() {
        let matrix = array![
            [Complex64::new(1.0_f64, -1.0), Complex64::new(2.5, 0.25)],
            [Complex64::new(-3.0, 0.5), Complex64::new(4.0, -2.0)],
        ];
        let storage = array2_complex64_to_fixed_size_list(matrix)
            .expect("complex64 matrix outbound should succeed");
        let item = match storage.data_type() {
            DataType::FixedSizeList(item, _) => Arc::clone(item),
            data_type => panic!("expected fixed-size list storage, found {data_type}"),
        };
        let values = storage.values();
        let bad_storage = FixedSizeListArray::new(
            item,
            storage.value_length(),
            Arc::clone(values),
            Some(NullBuffer::from(vec![true, false])),
        );

        let err = complex64_as_array_view2(&bad_storage).expect_err("outer nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn complex32_fixed_shape_tensor_roundtrip_zero_copy() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                Complex32::new(1.0_f32, 0.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 1.0),
                Complex32::new(4.0, 0.5),
                Complex32::new(5.0, -0.5),
                Complex32::new(6.0, 2.0),
                Complex32::new(7.0, 0.25),
                Complex32::new(8.0, -2.5),
            ],
        )
        .expect("shape must be valid");
        let original_ptr = tensor.as_ptr();

        let (field, storage) = arrayd_complex32_to_fixed_shape_tensor("tensor32", tensor)
            .expect("complex32 fixed-shape tensor outbound should succeed");
        let view = complex32_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect("complex32 fixed-shape tensor inbound should succeed");

        assert_eq!(view.shape(), &[2, 2, 2]);
        assert_eq!(view.as_ptr(), original_ptr);
        assert_abs_diff_eq!(view[[1, 1, 1]].re, 8.0_f32);
        assert_abs_diff_eq!(view[[1, 1, 1]].im, -2.5_f32);
    }

    #[test]
    fn complex32_fixed_shape_tensor_rejects_outer_nulls() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex32::new(1.0_f32, 0.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 1.0),
                Complex32::new(4.0, 0.5),
            ],
        )
        .expect("shape must be valid");
        let (field, storage) = arrayd_complex32_to_fixed_shape_tensor("tensor32", tensor)
            .expect("complex32 fixed-shape tensor outbound should succeed");
        let item = match storage.data_type() {
            DataType::FixedSizeList(item, _) => Arc::clone(item),
            data_type => panic!("expected fixed-size list storage, found {data_type}"),
        };
        let values = storage.values();
        let bad_storage = FixedSizeListArray::new(
            item,
            storage.value_length(),
            Arc::clone(values),
            Some(NullBuffer::from(vec![true, false])),
        );

        let err = complex32_fixed_shape_tensor_as_array_viewd(&field, &bad_storage)
            .expect_err("outer nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn complex32_fixed_shape_tensor_rejects_field_type_mismatch() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex32::new(1.0_f32, 0.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 1.0),
                Complex32::new(4.0, 0.5),
            ],
        )
        .expect("shape must be valid");
        let (_field, storage) = arrayd_complex32_to_fixed_shape_tensor("tensor32", tensor)
            .expect("complex32 fixed-shape tensor outbound should succeed");
        let bad_field =
            Field::new("bad", DataType::new_fixed_size_list(DataType::Float32, 2, false), false);

        let err = complex32_fixed_shape_tensor_as_array_viewd(&bad_field, &storage)
            .expect_err("field mismatch must fail");
        assert!(matches!(err, NdarrowError::TypeMismatch { .. }));
    }

    #[test]
    fn complex32_fixed_shape_tensor_rejects_shape_mismatch() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            (0_u8..8).map(|v| Complex32::new(f32::from(v), 0.0)).collect(),
        )
        .expect("shape must be valid");
        let (mut field, storage) = arrayd_complex32_to_fixed_shape_tensor("tensor32", tensor)
            .expect("complex32 fixed-shape tensor outbound should succeed");
        let bad_metadata = serde_json::json!({ "shape": [3, 2] }).to_string();
        let mut metadata = field.metadata().clone();
        metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), bad_metadata);
        field = field.with_metadata(metadata);

        let err = complex32_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect_err("shape mismatch must fail");
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. } | NdarrowError::Arrow(_)));
    }

    #[test]
    fn complex32_fixed_shape_tensor_rejects_non_nested_storage() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f32, 2.0, 3.0, 4.0])
            .expect("shape must be valid");
        let (field, storage) = crate::tensor::arrayd_to_fixed_shape_tensor("tensor32", tensor)
            .expect("plain fixed-shape tensor outbound should succeed");

        let err = complex32_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect_err("plain dense tensor storage must fail complex decoding");
        assert!(matches!(err, NdarrowError::InnerTypeMismatch { .. }));
    }

    #[test]
    fn complex32_fixed_shape_tensor_rejects_missing_extension_metadata() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex32::new(1.0_f32, 0.0),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 1.0),
                Complex32::new(4.0, 0.5),
            ],
        )
        .expect("shape must be valid");
        let (_field, storage) = arrayd_complex32_to_fixed_shape_tensor("tensor32", tensor)
            .expect("complex32 fixed-shape tensor outbound should succeed");
        let field = Field::new("tensor32", storage.data_type().clone(), false);

        let err = complex32_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect_err("missing extension metadata must fail");
        assert!(matches!(err, NdarrowError::Arrow(_)));
    }

    #[test]
    fn complex64_fixed_shape_tensor_roundtrip_zero_copy() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3]),
            vec![
                Complex64::new(1.0_f64, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, 0.5),
                Complex64::new(5.0, -0.5),
                Complex64::new(6.0, 2.0),
                Complex64::new(7.0, 0.25),
                Complex64::new(8.0, -2.5),
                Complex64::new(9.0, 1.5),
                Complex64::new(10.0, -0.75),
                Complex64::new(11.0, 0.125),
                Complex64::new(12.0, -3.0),
            ],
        )
        .expect("shape must be valid");
        let original_ptr = tensor.as_ptr();

        let (field, storage) = arrayd_complex64_to_fixed_shape_tensor("tensor64", tensor)
            .expect("complex64 fixed-shape tensor outbound should succeed");
        let view = complex64_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect("complex64 fixed-shape tensor inbound should succeed");

        assert_eq!(view.shape(), &[2, 2, 3]);
        assert_eq!(view.as_ptr(), original_ptr);
        assert_abs_diff_eq!(view[[1, 1, 2]].re, 12.0_f64);
        assert_abs_diff_eq!(view[[1, 1, 2]].im, -3.0_f64);
    }

    #[test]
    fn complex64_fixed_shape_tensor_rejects_field_type_mismatch() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex64::new(1.0_f64, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, 0.5),
            ],
        )
        .expect("shape must be valid");
        let (_field, storage) = arrayd_complex64_to_fixed_shape_tensor("tensor64", tensor)
            .expect("complex64 fixed-shape tensor outbound should succeed");
        let bad_field =
            Field::new("bad", DataType::new_fixed_size_list(DataType::Float64, 2, false), false);

        let err = complex64_fixed_shape_tensor_as_array_viewd(&bad_field, &storage)
            .expect_err("field mismatch must fail");
        assert!(matches!(err, NdarrowError::TypeMismatch { .. }));
    }

    #[test]
    fn complex64_fixed_shape_tensor_rejects_outer_nulls() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex64::new(1.0_f64, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, 0.5),
            ],
        )
        .expect("shape must be valid");
        let (field, storage) = arrayd_complex64_to_fixed_shape_tensor("tensor64", tensor)
            .expect("complex64 fixed-shape tensor outbound should succeed");
        let item = match storage.data_type() {
            DataType::FixedSizeList(item, _) => Arc::clone(item),
            data_type => panic!("expected fixed-size list storage, found {data_type}"),
        };
        let values = storage.values();
        let bad_storage = FixedSizeListArray::new(
            item,
            storage.value_length(),
            Arc::clone(values),
            Some(NullBuffer::from(vec![true, false])),
        );

        let err = complex64_fixed_shape_tensor_as_array_viewd(&field, &bad_storage)
            .expect_err("outer nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn complex64_fixed_shape_tensor_rejects_non_nested_storage() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0])
            .expect("shape must be valid");
        let (field, storage) = crate::tensor::arrayd_to_fixed_shape_tensor("tensor64", tensor)
            .expect("plain fixed-shape tensor outbound should succeed");

        let err = complex64_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect_err("plain dense tensor storage must fail complex decoding");
        assert!(matches!(err, NdarrowError::InnerTypeMismatch { .. }));
    }

    #[test]
    fn complex64_fixed_shape_tensor_rejects_missing_extension_metadata() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex64::new(1.0_f64, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, 0.5),
            ],
        )
        .expect("shape must be valid");
        let (_field, storage) = arrayd_complex64_to_fixed_shape_tensor("tensor64", tensor)
            .expect("complex64 fixed-shape tensor outbound should succeed");
        let field = Field::new("tensor64", storage.data_type().clone(), false);

        let err = complex64_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect_err("missing extension metadata must fail");
        assert!(matches!(err, NdarrowError::Arrow(_)));
    }

    #[test]
    fn complex64_fixed_shape_tensor_rejects_shape_mismatch() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            (0..8).map(|v| Complex64::new(f64::from(v), 0.0)).collect(),
        )
        .expect("shape must be valid");
        let (mut field, storage) = arrayd_complex64_to_fixed_shape_tensor("tensor64", tensor)
            .expect("complex64 fixed-shape tensor outbound should succeed");
        let bad_metadata = serde_json::json!({ "shape": [3, 2] }).to_string();
        let mut metadata = field.metadata().clone();
        metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), bad_metadata);
        field = field.with_metadata(metadata);

        let err = complex64_fixed_shape_tensor_as_array_viewd(&field, &storage)
            .expect_err("shape mismatch must fail");
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. } | NdarrowError::Arrow(_)));
    }

    #[test]
    fn complex32_variable_shape_tensor_roundtrip() {
        let a = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex32::new(1.0_f32, -1.0),
                Complex32::new(2.0, 0.5),
                Complex32::new(3.0, 1.5),
                Complex32::new(4.0, -0.25),
            ],
        )
        .unwrap();
        let b = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![Complex32::new(5.0_f32, 0.0), Complex32::new(6.0, -2.0)],
        )
        .unwrap();

        let (field, array) = arrays_complex32_to_variable_shape_tensor(
            "ragged32",
            vec![a, b],
            Some(vec![None, Some(2)]),
        )
        .unwrap();

        let mut iter = complex32_variable_shape_tensor_iter(&field, &array).unwrap();
        let (row0, view0) = iter.next().unwrap().unwrap();
        assert_eq!(row0, 0);
        assert_eq!(view0.shape(), &[2, 2]);
        assert_abs_diff_eq!(view0[[0, 0]].re, 1.0_f32);
        assert_abs_diff_eq!(view0[[0, 0]].im, -1.0_f32);
        assert_abs_diff_eq!(view0[[1, 1]].re, 4.0_f32);
        assert_abs_diff_eq!(view0[[1, 1]].im, -0.25_f32);

        let (row1, view1) = iter.next().unwrap().unwrap();
        assert_eq!(row1, 1);
        assert_eq!(view1.shape(), &[1, 2]);
        assert_abs_diff_eq!(view1[[0, 0]].re, 5.0_f32);
        assert_abs_diff_eq!(view1[[0, 1]].im, -2.0_f32);
        assert!(iter.next().is_none());
    }

    #[test]
    fn complex64_variable_shape_tensor_roundtrip_and_zero_copy() {
        let a = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![
                Complex64::new(1.0_f64, -1.0),
                Complex64::new(2.0, 0.5),
                Complex64::new(3.0, 1.5),
                Complex64::new(4.0, -0.25),
            ],
        )
        .unwrap();
        let b = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![Complex64::new(5.0_f64, 0.0), Complex64::new(6.0, -2.0)],
        )
        .unwrap();

        let (field, array) = arrays_complex64_to_variable_shape_tensor(
            "ragged64",
            vec![a, b],
            Some(vec![None, Some(2)]),
        )
        .unwrap();

        let data = array.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        let data_values = data.values().as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let data_item_field = match data.data_type() {
            DataType::List(item) => item.as_ref(),
            data_type => panic!("expected list data type, found {data_type}"),
        };
        let flat = complex64_as_array_view1(data_item_field, data_values).unwrap();

        let mut iter = complex64_variable_shape_tensor_iter(&field, &array).unwrap();
        let (row0, view0) = iter.next().unwrap().unwrap();
        assert_eq!(row0, 0);
        assert_eq!(view0.shape(), &[2, 2]);
        assert_eq!(view0.as_ptr(), flat.as_ptr());
        assert_abs_diff_eq!(view0[[1, 0]].re, 3.0_f64);
        assert_abs_diff_eq!(view0[[1, 0]].im, 1.5_f64);

        let (row1, view1) = iter.next().unwrap().unwrap();
        assert_eq!(row1, 1);
        assert_eq!(view1.shape(), &[1, 2]);
        assert_eq!(view1.as_ptr(), flat.as_ptr().wrapping_add(4));
        assert_abs_diff_eq!(view1[[0, 1]].re, 6.0_f64);
        assert_abs_diff_eq!(view1[[0, 1]].im, -2.0_f64);
        assert!(iter.next().is_none());
    }

    #[test]
    fn complex32_variable_shape_tensor_rejects_non_complex_storage() {
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 2.0]).unwrap();
        let (field, array) =
            crate::tensor::arrays_to_variable_shape_tensor("ragged", vec![a], None).unwrap();

        let result = complex32_variable_shape_tensor_iter(&field, &array);
        assert!(result.is_err(), "primitive ragged tensors must not decode as complex");
        let err = result.err().expect("primitive ragged tensors must not decode as complex");
        assert!(matches!(err, NdarrowError::InnerTypeMismatch { .. }));
    }

    #[test]
    fn complex64_variable_shape_tensor_rejects_outer_nulls() {
        let a = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![Complex64::new(1.0_f64, 0.0), Complex64::new(2.0, -1.0)],
        )
        .unwrap();
        let (field, array) =
            arrays_complex64_to_variable_shape_tensor("ragged64", vec![a], None).unwrap();
        let with_nulls = StructArray::new(
            array.fields().clone(),
            array.columns().to_vec(),
            Some(NullBuffer::from(vec![false])),
        );

        let result = complex64_variable_shape_tensor_iter(&field, &with_nulls);
        assert!(result.is_err(), "outer nulls must fail");
        let err = result.err().expect("outer nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn arrays_complex32_to_variable_shape_tensor_reject_empty_and_bad_shapes() {
        let err = arrays_complex32_to_variable_shape_tensor("ragged32", Vec::new(), None)
            .expect_err("empty complex tensor batches must fail");
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));

        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[1, 2]),
            vec![Complex32::new(1.0_f32, 0.0), Complex32::new(2.0, -1.0)],
        )
        .unwrap();
        let err = arrays_complex32_to_variable_shape_tensor(
            "ragged32",
            vec![tensor.clone()],
            Some(vec![Some(1)]),
        )
        .expect_err("uniform_shape rank mismatch must fail");
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));

        let rank_three =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![Complex32::new(3.0_f32, 0.5)]).unwrap();
        let err =
            arrays_complex32_to_variable_shape_tensor("ragged32", vec![tensor, rank_three], None)
                .expect_err("mixed tensor ranks must fail");
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn arrays_complex64_to_variable_shape_tensor_rejects_uniform_shape_violation() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[1, 3]),
            vec![Complex64::new(1.0_f64, 0.0), Complex64::new(2.0, -1.0), Complex64::new(3.0, 0.5)],
        )
        .unwrap();
        let err = arrays_complex64_to_variable_shape_tensor(
            "ragged64",
            vec![tensor],
            Some(vec![Some(1), Some(2)]),
        )
        .expect_err("uniform shape violations must fail");
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn arrayd_complex32_to_fixed_shape_tensor_rejects_zero_dim() {
        let tensor = ArrayD::from_elem(IxDyn(&[]), Complex32::new(1.0_f32, -1.0));
        let err = arrayd_complex32_to_fixed_shape_tensor("tensor32", tensor)
            .expect_err("zero-dimensional arrays must fail");
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn arrayd_complex64_to_fixed_shape_tensor_rejects_zero_dim() {
        let tensor = ArrayD::from_elem(IxDyn(&[]), Complex64::new(1.0_f64, -1.0));
        let err = arrayd_complex64_to_fixed_shape_tensor("tensor64", tensor)
            .expect_err("zero-dimensional arrays must fail");
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
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
