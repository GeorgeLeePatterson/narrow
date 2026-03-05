//! Tensor Arrow/ndarray bridge utilities.
//!
//! This module covers canonical tensor extension types:
//! - `arrow.fixed_shape_tensor`
//! - `arrow.variable_shape_tensor`

use std::{marker::PhantomData, sync::Arc};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int32Array, ListArray, PrimitiveArray, StructArray,
    types::ArrowPrimitiveType,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{
    DataType, Field,
    extension::{ExtensionType, FixedShapeTensor, VariableShapeTensor},
};
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use serde::Deserialize;

use crate::{element::NdarrowElement, error::NdarrowError};

#[derive(Debug, Deserialize)]
struct FixedShapeTensorWireMetadata {
    shape: Vec<usize>,
}

fn parse_fixed_shape_metadata(field: &Field) -> Result<Vec<usize>, NdarrowError> {
    field.try_extension_type::<FixedShapeTensor>().map_err(NdarrowError::from)?;

    let raw_metadata =
        field.extension_type_metadata().ok_or_else(|| NdarrowError::InvalidMetadata {
            message: "arrow.fixed_shape_tensor metadata missing".to_owned(),
        })?;
    let metadata: FixedShapeTensorWireMetadata =
        serde_json::from_str(raw_metadata).map_err(|e| NdarrowError::InvalidMetadata {
            message: format!("arrow.fixed_shape_tensor metadata parse failed: {e}"),
        })?;

    Ok(metadata.shape)
}

/// Converts `arrow.fixed_shape_tensor` storage into an `ArrayViewD`.
///
/// # Does not allocate
///
/// The returned ndarray view borrows Arrow's underlying primitive buffer.
///
/// # Errors
///
/// Returns an error on nulls, type mismatches, invalid extension metadata, or
/// shape incompatibility.
pub fn fixed_shape_tensor_as_array_viewd<'a, T>(
    field: &Field,
    array: &'a FixedSizeListArray,
) -> Result<ArrayViewD<'a, T::Native>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
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

    let tensor_shape = parse_fixed_shape_metadata(field)?;
    let values = array.values().as_any().downcast_ref::<PrimitiveArray<T>>().ok_or_else(|| {
        NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected inner array type {:?}, found {}",
                T::DATA_TYPE,
                array.values().data_type()
            ),
        }
    })?;

    if values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: values.null_count() });
    }

    let mut full_shape = Vec::with_capacity(tensor_shape.len() + 1);
    full_shape.push(array.len());
    full_shape.extend_from_slice(&tensor_shape);

    let slice: &[T::Native] = values.values().as_ref();
    ArrayViewD::from_shape(IxDyn(&full_shape), slice).map_err(NdarrowError::from)
}

/// Converts an owned ndarray tensor batch to `arrow.fixed_shape_tensor`.
///
/// The first ndarray axis is interpreted as batch dimension. The remaining
/// axes are encoded into extension metadata as tensor shape.
///
/// # Allocation
///
/// This transfer is zero-copy for standard-layout arrays. Non-standard layout
/// is normalized first and may allocate.
///
/// # Errors
///
/// Returns an error on invalid shape or Arrow metadata constraints.
pub fn arrayd_to_fixed_shape_tensor<T>(
    field_name: &str,
    array: ArrayD<T>,
) -> Result<(Field, FixedSizeListArray), NdarrowError>
where
    T: NdarrowElement,
{
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

    let standard =
        if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() };
    let total_len = standard.len();
    let (mut raw_vec, offset) = standard.into_raw_vec_and_offset();
    let start = offset.unwrap_or(0);
    let vec = if start == 0 {
        raw_vec.truncate(total_len);
        raw_vec
    } else {
        raw_vec[start..start + total_len].to_vec()
    };

    let values_array = PrimitiveArray::<T::ArrowType>::new(ScalarBuffer::from(vec), None);
    let item_field = Arc::new(Field::new("item", T::data_type(), false));
    let value_length = i32::try_from(element_count).map_err(|_| NdarrowError::ShapeMismatch {
        message: format!("tensor element count {element_count} exceeds Arrow i32 limits"),
    })?;
    let fsl = FixedSizeListArray::new(item_field, value_length, Arc::new(values_array), None);

    let extension = FixedShapeTensor::try_new(T::data_type(), tensor_shape, None, None)
        .map_err(NdarrowError::from)?;
    let mut field = Field::new(field_name, fsl.data_type().clone(), false);
    field.try_with_extension_type(extension).map_err(NdarrowError::from)?;

    Ok((field, fsl))
}

/// Iterator over per-row views for `arrow.variable_shape_tensor`.
pub struct VariableShapeTensorIter<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    index:         usize,
    len:           usize,
    data:          &'a ListArray,
    data_values:   &'a PrimitiveArray<T>,
    shape_values:  &'a Int32Array,
    dimensions:    usize,
    uniform_shape: Option<Vec<Option<i32>>>,
    marker:        PhantomData<T>,
}

impl<'a, T> Iterator for VariableShapeTensorIter<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    type Item = Result<(usize, ArrayViewD<'a, T::Native>), NdarrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }

        let row = self.index;
        self.index += 1;

        let offsets = self.data.value_offsets();
        let start = usize::try_from(offsets[row]).expect("Arrow list offsets must be non-negative");
        let end =
            usize::try_from(offsets[row + 1]).expect("Arrow list offsets must be non-negative");

        let shape_start = row * self.dimensions;
        let shape_end = shape_start + self.dimensions;

        let mut shape = Vec::with_capacity(self.dimensions);
        for (dim_idx, raw) in
            self.shape_values.values().as_ref()[shape_start..shape_end].iter().copied().enumerate()
        {
            let dim = usize::try_from(raw).map_err(|_| NdarrowError::ShapeMismatch {
                message: format!("negative tensor dimension at row {row}, dim {dim_idx}: {raw}"),
            });
            let dim = match dim {
                Ok(dim) => dim,
                Err(err) => return Some(Err(err)),
            };

            if let Some(uniform_shape) = &self.uniform_shape {
                if let Some(expected) = uniform_shape[dim_idx] {
                    let expected = usize::try_from(expected).map_err(|_| NdarrowError::InvalidMetadata {
                        message: format!(
                            "uniform_shape contains negative dimension at index {dim_idx}: {expected}"
                        ),
                    });
                    let expected = match expected {
                        Ok(expected) => expected,
                        Err(err) => return Some(Err(err)),
                    };
                    if dim != expected {
                        return Some(Err(NdarrowError::ShapeMismatch {
                            message: format!(
                                "row {row} dimension {dim_idx} violates uniform_shape: expected {expected}, found {dim}"
                            ),
                        }));
                    }
                }
            }

            shape.push(dim);
        }

        let required_len = shape
            .iter()
            .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| NdarrowError::ShapeMismatch {
                message: format!("row {row} shape product overflows usize: {shape:?}"),
            });
        let required_len = match required_len {
            Ok(required_len) => required_len,
            Err(err) => return Some(Err(err)),
        };

        if required_len != (end - start) {
            return Some(Err(NdarrowError::ShapeMismatch {
                message: format!(
                    "row {row} shape product ({required_len}) does not match data length ({})",
                    end - start
                ),
            }));
        }

        let slice = &self.data_values.values().as_ref()[start..end];
        let view = ArrayViewD::from_shape(IxDyn(&shape), slice).map_err(NdarrowError::from);
        Some(view.map(|view| (row, view)))
    }
}

/// Creates an iterator over per-row zero-copy ndarray views for
/// `arrow.variable_shape_tensor`.
///
/// # Does not allocate
///
/// This borrows Arrow values buffers directly. Iterator row shape decoding uses
/// small per-row shape vectors.
///
/// # Errors
///
/// Returns an error if extension/type/null invariants are violated.
///
/// # Panics
///
/// Panics only if `field` has already been validated as
/// `arrow.variable_shape_tensor` but `array` does not match the validated
/// extension storage schema.
pub fn variable_shape_tensor_iter<'a, T>(
    field: &Field,
    array: &'a StructArray,
) -> Result<VariableShapeTensorIter<'a, T>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    let extension =
        field.try_extension_type::<VariableShapeTensor>().map_err(NdarrowError::from)?;
    extension.supports_data_type(array.data_type()).map_err(NdarrowError::from)?;

    let data = array
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("extension storage guarantees 'data' is ListArray");
    if data.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: data.null_count() });
    }

    let shape = array
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("extension storage guarantees 'shape' is FixedSizeListArray");
    if shape.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: shape.null_count() });
    }

    let dimensions = extension.dimensions();

    let data_values = data
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .expect("extension storage guarantees variable data inner type");
    let shape_values = shape
        .values()
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("extension storage guarantees variable shape inner Int32");
    Ok(VariableShapeTensorIter {
        index: 0,
        len: data.len(),
        data,
        data_values,
        shape_values,
        dimensions,
        uniform_shape: extension.uniform_shapes().map(<[Option<i32>]>::to_vec),
        marker: PhantomData,
    })
}

fn push_tensor_shape(
    shape: &[usize],
    row: usize,
    uniform_shape: Option<&[Option<i32>]>,
    packed_shapes: &mut Vec<i32>,
) -> Result<(), NdarrowError> {
    for (dim_idx, dim) in shape.iter().copied().enumerate() {
        let dim_i32 = i32::try_from(dim).map_err(|_| NdarrowError::ShapeMismatch {
            message: format!(
                "tensor dimension exceeds i32 limits at row {row}, dim {dim_idx}: {dim}"
            ),
        })?;
        if let Some(uniform) = uniform_shape {
            if let Some(expected) = uniform[dim_idx] {
                if expected != dim_i32 {
                    return Err(NdarrowError::ShapeMismatch {
                        message: format!(
                            "row {row} dimension {dim_idx} violates uniform_shape: expected {expected}, found {dim_i32}"
                        ),
                    });
                }
            }
        }
        packed_shapes.push(dim_i32);
    }
    Ok(())
}

fn append_row_values<T>(
    array: ArrayD<T>,
    row: usize,
    packed_values: &mut Vec<T>,
) -> Result<i32, NdarrowError>
where
    T: NdarrowElement,
{
    let standard =
        if array.is_standard_layout() { array } else { array.as_standard_layout().into_owned() };
    let element_count = standard.len();
    let (mut raw_vec, offset) = standard.into_raw_vec_and_offset();
    let start = offset.unwrap_or(0);

    if start == 0 {
        raw_vec.truncate(element_count);
        packed_values.extend(raw_vec);
    } else {
        packed_values.extend_from_slice(&raw_vec[start..start + element_count]);
    }

    i32::try_from(element_count).map_err(|_| NdarrowError::ShapeMismatch {
        message: format!(
            "tensor row element count exceeds i32 limits at row {row}: {element_count}"
        ),
    })
}

/// Converts owned tensors into `arrow.variable_shape_tensor` storage.
///
/// # Allocation
///
/// This packs ragged tensor rows into Arrow list/shape buffers and therefore
/// allocates.
///
/// # Errors
///
/// Returns an error when tensor ranks differ or metadata constraints fail.
pub fn arrays_to_variable_shape_tensor<T>(
    field_name: &str,
    arrays: Vec<ArrayD<T>>,
    uniform_shape: Option<Vec<Option<i32>>>,
) -> Result<(Field, StructArray), NdarrowError>
where
    T: NdarrowElement,
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

        push_tensor_shape(array.shape(), row, uniform_shape.as_deref(), &mut packed_shapes)?;
        let element_count_i32 = append_row_values(array, row, &mut packed_values)?;
        running_offset = running_offset.checked_add(element_count_i32).ok_or_else(|| {
            NdarrowError::ShapeMismatch {
                message: "packed variable tensor offsets exceed i32 limits".to_owned(),
            }
        })?;
        offsets.push(running_offset);
    }

    let data_offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let data_values = PrimitiveArray::<T::ArrowType>::new(ScalarBuffer::from(packed_values), None);
    let data_item_field = Arc::new(Field::new_list_field(T::data_type(), false));
    let data_list: ArrayRef =
        Arc::new(ListArray::new(data_item_field, data_offsets, Arc::new(data_values), None));

    let shape_values = Int32Array::new(ScalarBuffer::from(packed_shapes), None);
    let shape_item_field = Arc::new(Field::new("item", DataType::Int32, false));
    let shape_fsl: ArrayRef = Arc::new(FixedSizeListArray::new(
        shape_item_field,
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

    let extension =
        VariableShapeTensor::try_new(T::data_type(), dimensions, None, None, uniform_shape)
            .map_err(NdarrowError::from)?;
    let mut field = Field::new(field_name, struct_array.data_type().clone(), false);
    field.try_with_extension_type(extension).map_err(NdarrowError::from)?;

    Ok((field, struct_array))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use approx::assert_abs_diff_eq;
    use arrow_array::{Float32Array, Float64Array, Int32Array, types::Float32Type};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{
        DataType, Field,
        extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY},
    };

    use super::*;

    #[test]
    fn fixed_shape_tensor_roundtrip() {
        let data: Vec<f32> = (0_u16..24_u16).map(|v| f32::from(v) * 0.5).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), data.clone()).unwrap();

        let (field, fsl) = arrayd_to_fixed_shape_tensor("tensor", array).unwrap();
        let view = fixed_shape_tensor_as_array_viewd::<Float32Type>(&field, &fsl).unwrap();

        assert_eq!(view.shape(), &[2, 3, 4]);
        for (actual, expected) in view.iter().zip(data.iter()) {
            assert_abs_diff_eq!(*actual, *expected);
        }
    }

    #[test]
    fn fixed_shape_tensor_outbound_zero_copy_standard_layout() {
        let data: Vec<f64> = (0_u16..12_u16).map(f64::from).collect();
        let ptr = data.as_ptr();
        let array = ArrayD::from_shape_vec(IxDyn(&[3, 4]), data).unwrap();

        let (_field, fsl) = arrayd_to_fixed_shape_tensor("tensor", array).unwrap();
        let inner = fsl.values().as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(ptr, inner.values().as_ref().as_ptr());
    }

    #[test]
    fn variable_shape_tensor_roundtrip() {
        let a =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![7.0_f32, 8.0, 9.0]).unwrap();

        let (field, array) =
            arrays_to_variable_shape_tensor("ragged", vec![a, b], Some(vec![None, Some(3)]))
                .unwrap();

        let mut iter = variable_shape_tensor_iter::<Float32Type>(&field, &array).unwrap();

        let (row0, view0) = iter.next().unwrap().unwrap();
        assert_eq!(row0, 0);
        assert_eq!(view0.shape(), &[2, 3]);
        assert_abs_diff_eq!(view0[[0, 0]], 1.0_f32);
        assert_abs_diff_eq!(view0[[1, 2]], 6.0_f32);

        let (row1, view1) = iter.next().unwrap().unwrap();
        assert_eq!(row1, 1);
        assert_eq!(view1.shape(), &[1, 3]);
        assert_abs_diff_eq!(view1[[0, 0]], 7.0_f32);
        assert_abs_diff_eq!(view1[[0, 2]], 9.0_f32);

        assert!(iter.next().is_none());
    }

    #[test]
    fn variable_shape_tensor_uniform_shape_violation() {
        let a =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![7.0_f32, 8.0, 9.0, 10.0]).unwrap();

        let err = arrays_to_variable_shape_tensor("ragged", vec![a, b], Some(vec![None, Some(3)]))
            .unwrap_err();

        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn variable_shape_tensor_iterator_zero_copy() {
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![5.0_f32, 6.0]).unwrap();
        let (field, array) = arrays_to_variable_shape_tensor("ragged", vec![a, b], None).unwrap();

        let data_list =
            array.column_by_name("data").unwrap().as_any().downcast_ref::<ListArray>().unwrap();
        let data_values = data_list.values().as_any().downcast_ref::<Float32Array>().unwrap();
        let ptr = data_values.values().as_ref().as_ptr();

        let mut iter = variable_shape_tensor_iter::<Float32Type>(&field, &array).unwrap();
        let (_row, view) = iter.next().unwrap().unwrap();
        assert_eq!(view.as_ptr(), ptr);
    }

    #[test]
    fn fixed_shape_tensor_rejects_missing_extension() {
        let values = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let fsl = FixedSizeListArray::new(item_field, 2, Arc::new(values), None);
        let field = Field::new("tensor", fsl.data_type().clone(), false);
        let err = fixed_shape_tensor_as_array_viewd::<Float32Type>(&field, &fsl).unwrap_err();
        assert!(matches!(err, NdarrowError::Arrow(_)));
    }

    #[test]
    fn fixed_shape_tensor_rejects_invalid_extension_metadata() {
        let values = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let fsl = FixedSizeListArray::new(item_field, 2, Arc::new(values), None);

        let mut metadata = HashMap::new();
        metadata.insert(EXTENSION_TYPE_NAME_KEY.to_owned(), FixedShapeTensor::NAME.to_owned());
        metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), "{bad json".to_owned());
        let field = Field::new("tensor", fsl.data_type().clone(), false).with_metadata(metadata);

        let err = fixed_shape_tensor_as_array_viewd::<Float32Type>(&field, &fsl).unwrap_err();
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. } | NdarrowError::Arrow(_)));
    }

    #[test]
    fn fixed_shape_tensor_rejects_outer_nulls() {
        let values = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let nulls = NullBuffer::from(vec![true, false]);
        let fsl = FixedSizeListArray::new(item_field, 2, Arc::new(values), Some(nulls));

        let mut metadata = HashMap::new();
        metadata.insert(EXTENSION_TYPE_NAME_KEY.to_owned(), FixedShapeTensor::NAME.to_owned());
        metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), r#"{"shape":[2]}"#.to_owned());
        let field = Field::new("tensor", fsl.data_type().clone(), false).with_metadata(metadata);

        let err = fixed_shape_tensor_as_array_viewd::<Float32Type>(&field, &fsl).unwrap_err();
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn fixed_shape_tensor_rejects_inner_type_mismatch() {
        let values = Float64Array::from(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let item_field = Arc::new(Field::new("item", DataType::Float64, false));
        let fsl = FixedSizeListArray::new(item_field, 2, Arc::new(values), None);
        let fixed_ext = FixedShapeTensor::try_new(DataType::Float64, [2], None, None).unwrap();
        let mut field = Field::new("tensor", fsl.data_type().clone(), false);
        field.try_with_extension_type(fixed_ext).unwrap();

        let err = fixed_shape_tensor_as_array_viewd::<Float32Type>(&field, &fsl).unwrap_err();
        assert!(matches!(err, NdarrowError::InnerTypeMismatch { .. }));
    }

    #[test]
    fn arrayd_to_fixed_shape_tensor_rejects_zero_dim() {
        let scalar = ArrayD::from_elem(IxDyn(&[]), 1.0_f32);
        let err = arrayd_to_fixed_shape_tensor("tensor", scalar).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn variable_shape_tensor_iter_rejects_struct_nulls() {
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 2.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![3.0_f32, 4.0]).unwrap();
        let (field, array) = arrays_to_variable_shape_tensor("ragged", vec![a, b], None).unwrap();

        let with_nulls = StructArray::new(
            array.fields().clone(),
            array.columns().to_vec(),
            Some(NullBuffer::from(vec![true, false])),
        );

        let result = variable_shape_tensor_iter::<Float32Type>(&field, &with_nulls);
        assert!(matches!(result, Err(NdarrowError::NullsPresent { .. })));
    }

    #[test]
    fn variable_shape_tensor_iter_rejects_negative_shape_dimension() {
        let data_values = Float32Array::from(vec![1.0_f32, 2.0, 3.0]);
        let data_offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 3_i32]));
        let data_item_field = Arc::new(Field::new_list_field(DataType::Float32, false));
        let data: ArrayRef =
            Arc::new(ListArray::new(data_item_field, data_offsets, Arc::new(data_values), None));

        let shape_values = Int32Array::from(vec![-1_i32, 3_i32]);
        let shape_item_field = Arc::new(Field::new("item", DataType::Int32, false));
        let shape: ArrayRef =
            Arc::new(FixedSizeListArray::new(shape_item_field, 2, Arc::new(shape_values), None));

        let struct_fields = vec![
            Field::new("data", data.data_type().clone(), false),
            Field::new("shape", shape.data_type().clone(), false),
        ];
        let array = StructArray::new(struct_fields.into(), vec![data, shape], None);
        let ext = VariableShapeTensor::try_new(DataType::Float32, 2, None, None, None).unwrap();
        let mut field = Field::new("ragged", array.data_type().clone(), false);
        field.try_with_extension_type(ext).unwrap();

        let mut iter = variable_shape_tensor_iter::<Float32Type>(&field, &array).unwrap();
        let err = iter.next().unwrap().unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn variable_shape_tensor_iter_rejects_negative_uniform_shape_metadata() {
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 2.0]).unwrap();
        let (_field, array) = arrays_to_variable_shape_tensor("ragged", vec![a], None).unwrap();
        let ext = VariableShapeTensor::try_new(
            DataType::Float32,
            2,
            None,
            None,
            Some(vec![Some(-1), None]),
        )
        .unwrap();
        let mut field = Field::new("ragged", array.data_type().clone(), false);
        field.try_with_extension_type(ext).unwrap();
        let mut iter = variable_shape_tensor_iter::<Float32Type>(&field, &array).unwrap();
        let err = iter.next().unwrap().unwrap_err();
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }

    #[test]
    fn push_tensor_shape_rejects_uniform_mismatch() {
        let mut packed = Vec::new();
        let err =
            push_tensor_shape(&[2, 4], 0, Some(&[Some(2), Some(3)]), &mut packed).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn push_tensor_shape_rejects_dimension_overflow() {
        let mut packed = Vec::new();
        let err = push_tensor_shape(&[usize::MAX], 0, None, &mut packed).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn append_row_values_handles_non_zero_offset() {
        let base =
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sliced = base.slice_move(ndarray::s![1.., ..]).into_dyn();
        let mut packed = Vec::new();
        let count = append_row_values(sliced, 0, &mut packed).unwrap();
        assert_eq!(count, 4);
        assert_eq!(packed, vec![3.0_f64, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn arrays_to_variable_shape_tensor_rejects_empty_input() {
        let err = arrays_to_variable_shape_tensor::<f32>("ragged", vec![], None).unwrap_err();
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }

    #[test]
    fn arrays_to_variable_shape_tensor_rejects_uniform_shape_rank_mismatch() {
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 2.0]).unwrap();
        let err = arrays_to_variable_shape_tensor(
            "ragged",
            vec![a],
            Some(vec![Some(1), Some(2), Some(3)]),
        )
        .unwrap_err();
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }

    #[test]
    fn arrays_to_variable_shape_tensor_rejects_rank_mismatch() {
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f64, 2.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![3.0_f64]).unwrap();
        let err = arrays_to_variable_shape_tensor("ragged", vec![a, b], None).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn variable_shape_tensor_iter_rejects_shape_product_mismatch() {
        let data_values = Float32Array::from(vec![1.0_f32, 2.0, 3.0]);
        let data_offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 3_i32]));
        let data_item_field = Arc::new(Field::new_list_field(DataType::Float32, false));
        let data: ArrayRef =
            Arc::new(ListArray::new(data_item_field, data_offsets, Arc::new(data_values), None));

        let shape_values = Int32Array::from(vec![1_i32, 2_i32]);
        let shape_item_field = Arc::new(Field::new("item", DataType::Int32, false));
        let shape: ArrayRef =
            Arc::new(FixedSizeListArray::new(shape_item_field, 2, Arc::new(shape_values), None));

        let struct_fields = vec![
            Field::new("data", data.data_type().clone(), false),
            Field::new("shape", shape.data_type().clone(), false),
        ];
        let array = StructArray::new(struct_fields.into(), vec![data, shape], None);
        let ext = VariableShapeTensor::try_new(DataType::Float32, 2, None, None, None).unwrap();
        let mut field = Field::new("ragged", array.data_type().clone(), false);
        field.try_with_extension_type(ext).unwrap();

        let mut iter = variable_shape_tensor_iter::<Float32Type>(&field, &array).unwrap();
        let err = iter.next().unwrap().unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn variable_shape_tensor_iter_rejects_wrong_extension_name() {
        let a = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 2.0]).unwrap();
        let (_field, array) = arrays_to_variable_shape_tensor("ragged", vec![a], None).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert(EXTENSION_TYPE_NAME_KEY.to_owned(), "ndarrow.not_tensor".to_owned());
        metadata.insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), "{}".to_owned());
        let bad_field =
            Field::new("ragged", array.data_type().clone(), false).with_metadata(metadata);

        let result = variable_shape_tensor_iter::<Float32Type>(&bad_field, &array);
        assert!(matches!(result, Err(NdarrowError::Arrow(_))));
    }
}
