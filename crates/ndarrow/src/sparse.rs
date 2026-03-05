//! Sparse Arrow/ndarray bridge utilities.
//!
//! This module defines the custom `ndarrow.csr_matrix` extension type and
//! zero-copy sparse inbound conversions.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, ListArray, PrimitiveArray, StructArray, UInt32Array, types::ArrowPrimitiveType,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, extension::ExtensionType};
use serde::{Deserialize, Serialize};

use crate::{element::NdarrowElement, error::NdarrowError};

/// Metadata carried by `ndarrow.csr_matrix`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsrMatrixMetadata {
    /// Number of columns in the sparse matrix.
    pub ncols: usize,
}

/// `ndarrow.csr_matrix` extension type.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrixExtension {
    value_type: DataType,
    metadata:   CsrMatrixMetadata,
}

impl CsrMatrixExtension {
    /// Returns the value type stored in the CSR values buffer.
    #[must_use]
    pub fn value_type(&self) -> &DataType {
        &self.value_type
    }

    /// Returns the number of columns in the sparse matrix.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.metadata.ncols
    }

    fn expected_storage_type(&self) -> DataType {
        DataType::Struct(
            vec![
                Field::new("indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::new_list(self.value_type.clone(), false), false),
            ]
            .into(),
        )
    }
}

impl ExtensionType for CsrMatrixExtension {
    type Metadata = CsrMatrixMetadata;

    const NAME: &'static str = "ndarrow.csr_matrix";

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(serde_json::to_string(&self.metadata).expect("csr metadata serialization"))
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        metadata.map_or_else(
            || {
                Err(ArrowError::InvalidArgumentError(
                    "ndarrow.csr_matrix extension type requires metadata".to_owned(),
                ))
            },
            |value| {
                serde_json::from_str(value).map_err(|e| {
                    ArrowError::InvalidArgumentError(format!(
                        "ndarrow.csr_matrix metadata deserialization failed: {e}"
                    ))
                })
            },
        )
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        let expected = self.expected_storage_type();
        data_type.equals_datatype(&expected).then_some(()).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "ndarrow.csr_matrix data type mismatch, expected {expected}, found {data_type}"
            ))
        })
    }

    fn try_new(data_type: &DataType, metadata: Self::Metadata) -> Result<Self, ArrowError> {
        match data_type {
            DataType::Struct(fields)
                if fields.len() == 2
                    && matches!(fields.find("indices"), Some((0, _)))
                    && matches!(fields.find("values"), Some((1, _))) =>
            {
                let indices_field = &fields[0];
                let value_type = match indices_field.data_type() {
                    DataType::List(inner) if inner.data_type() == &DataType::UInt32 => {
                        let values_field = &fields[1];
                        match values_field.data_type() {
                            DataType::List(values_inner) => values_inner.data_type().clone(),
                            other => {
                                return Err(ArrowError::InvalidArgumentError(format!(
                                    "ndarrow.csr_matrix data type mismatch, expected List for values field, found {other}"
                                )));
                            }
                        }
                    }
                    other => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "ndarrow.csr_matrix data type mismatch, expected List<UInt32> for indices field, found {other}"
                        )));
                    }
                };

                let extension = Self { value_type, metadata };
                extension.supports_data_type(data_type)?;
                Ok(extension)
            }
            other => Err(ArrowError::InvalidArgumentError(format!(
                "ndarrow.csr_matrix data type mismatch, expected Struct{{indices,values}}, found {other}"
            ))),
        }
    }
}

/// Borrowed CSR view over Arrow buffers.
#[derive(Debug, Clone, Copy)]
pub struct CsrView<'a, T> {
    /// Number of rows.
    pub nrows:       usize,
    /// Number of columns.
    pub ncols:       usize,
    /// CSR row pointer buffer (Arrow `List<i32>` offsets).
    pub row_ptrs:    &'a [i32],
    /// CSR column indices.
    pub col_indices: &'a [u32],
    /// CSR non-zero values.
    pub values:      &'a [T],
}

impl<T> CsrView<'_, T> {
    /// Returns number of non-zero values.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

fn offset_to_usize(offset: i32, context: &str) -> Result<usize, NdarrowError> {
    usize::try_from(offset).map_err(|_| NdarrowError::InvalidMetadata {
        message: format!("invalid negative offset in {context}: {offset}"),
    })
}

fn list_as_u32_values<'a>(
    array: &'a ListArray,
    column_name: &str,
) -> Result<&'a UInt32Array, NdarrowError> {
    array.values().as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
        NdarrowError::TypeMismatch {
            message: format!(
                "column '{column_name}' must be List<UInt32>, found {}",
                array.values().data_type()
            ),
        }
    })
}

fn list_as_t_values<'a, T>(
    array: &'a ListArray,
    column_name: &str,
) -> Result<&'a PrimitiveArray<T>, NdarrowError>
where
    T: ArrowPrimitiveType,
{
    array.values().as_any().downcast_ref::<PrimitiveArray<T>>().ok_or_else(|| {
        NdarrowError::TypeMismatch {
            message: format!(
                "column '{column_name}' must be List<{:?}>, found {}",
                T::DATA_TYPE,
                array.values().data_type()
            ),
        }
    })
}

/// Builds a zero-copy CSR view from two Arrow list columns.
///
/// # Does not allocate
///
/// This borrows offsets and value buffers directly.
///
/// # Errors
///
/// Returns an error if types, lengths, offsets, or null semantics are invalid.
pub fn csr_view_from_columns<'a, T>(
    indices: &'a ListArray,
    values: &'a ListArray,
    ncols: usize,
) -> Result<CsrView<'a, T::Native>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    if indices.len() != values.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "indices and values row count mismatch: {} vs {}",
                indices.len(),
                values.len()
            ),
        });
    }

    if indices.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: indices.null_count() });
    }
    if values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: values.null_count() });
    }
    if indices.value_offsets() != values.value_offsets() {
        return Err(NdarrowError::SparseOffsetMismatch);
    }

    let indices_values = list_as_u32_values(indices, "indices")?;
    let value_values = list_as_t_values::<T>(values, "values")?;

    if indices_values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: indices_values.null_count() });
    }
    if value_values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: value_values.null_count() });
    }

    if indices_values.len() != value_values.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "nnz length mismatch between indices and values: {} vs {}",
                indices_values.len(),
                value_values.len()
            ),
        });
    }

    let row_ptrs: &[i32] = indices.offsets().as_ref();
    let first_offset = row_ptrs.first().copied().ok_or_else(|| NdarrowError::InvalidMetadata {
        message: "empty offsets buffer for CSR lists".to_owned(),
    })?;
    if first_offset != 0 {
        return Err(NdarrowError::InvalidMetadata {
            message: format!("CSR offsets must start at 0, found {first_offset}"),
        });
    }

    let last_offset = row_ptrs.last().copied().ok_or_else(|| NdarrowError::InvalidMetadata {
        message: "empty offsets buffer for CSR lists".to_owned(),
    })?;
    let nnz = offset_to_usize(last_offset, "csr row_ptrs")?;
    if nnz != indices_values.len() || nnz != value_values.len() {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "CSR offsets last value ({nnz}) must match nnz lengths (indices={}, values={})",
                indices_values.len(),
                value_values.len()
            ),
        });
    }

    Ok(CsrView {
        nrows: indices.len(),
        ncols,
        row_ptrs,
        col_indices: indices_values.values().as_ref(),
        values: value_values.values().as_ref(),
    })
}

/// Builds a zero-copy CSR view from a `StructArray` tagged as `ndarrow.csr_matrix`.
///
/// # Does not allocate
///
/// This borrows offsets and values from Arrow buffers.
///
/// # Errors
///
/// Returns an error if the extension type or storage layout is invalid.
///
/// # Panics
///
/// Panics only if `field` has already been validated as `ndarrow.csr_matrix`
/// but `array` does not match the validated extension storage schema.
pub fn csr_view_from_extension<'a, T>(
    field: &Field,
    array: &'a StructArray,
) -> Result<CsrView<'a, T::Native>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    let extension = field.try_extension_type::<CsrMatrixExtension>().map_err(NdarrowError::from)?;
    extension.supports_data_type(array.data_type()).map_err(NdarrowError::from)?;

    let indices = array
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("extension storage guarantees 'indices' is ListArray");

    let values = array
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("extension storage guarantees 'values' is ListArray");

    csr_view_from_columns::<T>(indices, values, extension.ncols())
}

fn validate_csr_parts(
    row_ptrs: &[i32],
    col_indices_len: usize,
    values_len: usize,
) -> Result<(), NdarrowError> {
    if row_ptrs.is_empty() {
        return Err(NdarrowError::InvalidMetadata {
            message: "row_ptrs must contain at least one offset (0)".to_owned(),
        });
    }
    if row_ptrs[0] != 0 {
        return Err(NdarrowError::InvalidMetadata {
            message: format!("row_ptrs must start at 0, found {}", row_ptrs[0]),
        });
    }

    for window in row_ptrs.windows(2) {
        if window[1] < window[0] {
            return Err(NdarrowError::InvalidMetadata {
                message: format!("row_ptrs must be non-decreasing, found {row_ptrs:?}"),
            });
        }
    }

    if col_indices_len != values_len {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "col_indices and values lengths must match: {col_indices_len} vs {values_len}"
            ),
        });
    }

    let last = row_ptrs.last().copied().ok_or_else(|| NdarrowError::InvalidMetadata {
        message: "row_ptrs must not be empty".to_owned(),
    })?;
    let last_usize = offset_to_usize(last, "row_ptrs")?;
    if last_usize != col_indices_len {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "row_ptrs last offset ({last_usize}) must equal number of non-zeros ({col_indices_len})"
            ),
        });
    }
    Ok(())
}

/// Builds an Arrow `StructArray` plus extension field for `ndarrow.csr_matrix`.
///
/// # Allocation
///
/// This function allocates Arrow array wrappers and offset/value containers,
/// while transferring ownership of the provided vectors.
///
/// # Errors
///
/// Returns an error if CSR structural invariants are violated.
pub fn csr_to_extension_array<T>(
    field_name: &str,
    ncols: usize,
    row_ptrs: Vec<i32>,
    col_indices: Vec<u32>,
    values: Vec<T>,
) -> Result<(Field, StructArray), NdarrowError>
where
    T: NdarrowElement,
{
    validate_csr_parts(&row_ptrs, col_indices.len(), values.len())?;

    let offsets = OffsetBuffer::new(ScalarBuffer::from(row_ptrs));

    let indices_values = UInt32Array::new(ScalarBuffer::from(col_indices), None);
    let indices_item_field = Arc::new(Field::new_list_field(DataType::UInt32, false));
    let indices: ArrayRef = Arc::new(ListArray::new(
        indices_item_field,
        offsets.clone(),
        Arc::new(indices_values),
        None,
    ));

    let values_values = PrimitiveArray::<T::ArrowType>::new(ScalarBuffer::from(values), None);
    let values_item_field = Arc::new(Field::new_list_field(T::data_type(), false));
    let values_array: ArrayRef =
        Arc::new(ListArray::new(values_item_field, offsets, Arc::new(values_values), None));

    let struct_fields = vec![
        Field::new("indices", indices.data_type().clone(), false),
        Field::new("values", values_array.data_type().clone(), false),
    ];
    let struct_array =
        StructArray::new(struct_fields.clone().into(), vec![indices, values_array], None);

    let extension =
        CsrMatrixExtension::try_new(struct_array.data_type(), CsrMatrixMetadata { ncols })
            .map_err(NdarrowError::from)?;
    let mut field = Field::new(field_name, struct_array.data_type().clone(), false);
    field.try_with_extension_type(extension).map_err(NdarrowError::from)?;

    Ok((field, struct_array))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use approx::assert_abs_diff_eq;
    use arrow_array::{
        Float64Array, Int32Array,
        types::{Float64Type, Int32Type, UInt32Type},
    };
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;

    use super::*;

    fn make_columns() -> (ListArray, ListArray) {
        let indices = ListArray::from_iter_primitive::<Int32Type, _, _>([
            Some(vec![Some(0), Some(2)]),
            Some(vec![Some(1)]),
            Some(vec![Some(0), Some(3)]),
        ]);
        let values = ListArray::from_iter_primitive::<Float64Type, _, _>([
            Some(vec![Some(1.0), Some(5.0)]),
            Some(vec![Some(2.0)]),
            Some(vec![Some(3.0), Some(4.0)]),
        ]);
        let indices_u32 = {
            let child = indices.values().as_any().downcast_ref::<Int32Array>().unwrap();
            let converted = UInt32Array::from(
                child
                    .values()
                    .iter()
                    .map(|v| u32::try_from(*v).expect("test indices must fit u32"))
                    .collect::<Vec<_>>(),
            );
            let item_field = Arc::new(Field::new_list_field(DataType::UInt32, false));
            ListArray::new(item_field, indices.offsets().clone(), Arc::new(converted), None)
        };
        (indices_u32, values)
    }

    fn csr_storage_type(value_type: DataType) -> DataType {
        DataType::Struct(
            vec![
                Field::new("indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::new_list(value_type, false), false),
            ]
            .into(),
        )
    }

    #[test]
    fn csr_view_from_columns_success() {
        let (indices, values) = make_columns();
        let view = csr_view_from_columns::<Float64Type>(&indices, &values, 4).unwrap();

        assert_eq!(view.nrows, 3);
        assert_eq!(view.ncols, 4);
        assert_eq!(view.row_ptrs, &[0, 2, 3, 5]);
        assert_eq!(view.col_indices, &[0, 2, 1, 0, 3]);
        assert_eq!(view.nnz(), 5);
        assert_abs_diff_eq!(view.values[0], 1.0);
        assert_abs_diff_eq!(view.values[4], 4.0);
    }

    #[test]
    fn csr_view_from_columns_mismatched_offsets() {
        let (indices, _) = make_columns();
        let bad_values = ListArray::from_iter_primitive::<Float64Type, _, _>([
            Some(vec![Some(1.0)]),
            Some(vec![Some(2.0)]),
            Some(vec![Some(3.0)]),
        ]);
        let err = csr_view_from_columns::<Float64Type>(&indices, &bad_values, 4).unwrap_err();
        assert!(matches!(err, NdarrowError::SparseOffsetMismatch));
    }

    #[test]
    fn csr_to_extension_array_roundtrip() {
        let row_ptrs = vec![0, 2, 3, 5];
        let col_indices = vec![0_u32, 2, 1, 0, 3];
        let values = vec![1.0_f64, 5.0, 2.0, 3.0, 4.0];

        let (field, array) =
            csr_to_extension_array("sparse", 4, row_ptrs, col_indices, values).unwrap();

        assert_eq!(field.extension_type_name(), Some(CsrMatrixExtension::NAME));
        assert_eq!(
            field.metadata().get(EXTENSION_TYPE_NAME_KEY).map(String::as_str),
            Some(CsrMatrixExtension::NAME)
        );

        let view = csr_view_from_extension::<Float64Type>(&field, &array).unwrap();
        assert_eq!(view.nrows, 3);
        assert_eq!(view.ncols, 4);
        assert_eq!(view.row_ptrs, &[0, 2, 3, 5]);
        assert_eq!(view.col_indices, &[0, 2, 1, 0, 3]);
        assert_abs_diff_eq!(view.values[0], 1.0);
        assert_abs_diff_eq!(view.values[4], 4.0);
    }

    #[test]
    fn csr_to_extension_array_rejects_invalid_row_ptrs() {
        let err = csr_to_extension_array::<f64>(
            "sparse",
            3,
            vec![1, 2, 2],
            vec![0_u32, 1],
            vec![1.0, 2.0],
        )
        .unwrap_err();
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }

    #[test]
    fn csr_to_extension_array_rejects_shape_mismatch() {
        let err = csr_to_extension_array::<f64>(
            "sparse",
            3,
            vec![0, 1, 3],
            vec![0_u32, 1],
            vec![1.0, 2.0],
        )
        .unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn csr_extension_type_roundtrip() {
        let data_type = DataType::Struct(
            vec![
                Field::new("indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::new_list(DataType::Float64, false), false),
            ]
            .into(),
        );
        let ext =
            CsrMatrixExtension::try_new(&data_type, CsrMatrixMetadata { ncols: 1024 }).unwrap();
        let metadata = ext.serialize_metadata().unwrap();
        let deserialized = CsrMatrixExtension::deserialize_metadata(Some(&metadata)).unwrap();
        assert_eq!(deserialized.ncols, 1024);
    }

    #[test]
    fn csr_extension_accessors_and_metadata() {
        let data_type = csr_storage_type(DataType::Float64);
        let extension =
            CsrMatrixExtension::try_new(&data_type, CsrMatrixMetadata { ncols: 7 }).unwrap();
        assert_eq!(extension.value_type(), &DataType::Float64);
        assert_eq!(extension.ncols(), 7);
        assert_eq!(extension.metadata().ncols, 7);
    }

    #[test]
    fn csr_extension_deserialize_errors() {
        let missing = CsrMatrixExtension::deserialize_metadata(None).unwrap_err();
        assert!(missing.to_string().contains("requires metadata"));

        let invalid = CsrMatrixExtension::deserialize_metadata(Some("{not-json}")).unwrap_err();
        assert!(invalid.to_string().contains("deserialization failed"));
    }

    #[test]
    fn csr_extension_supports_data_type_mismatch() {
        let data_type = csr_storage_type(DataType::Float64);
        let extension =
            CsrMatrixExtension::try_new(&data_type, CsrMatrixMetadata { ncols: 3 }).unwrap();
        let err = extension.supports_data_type(&DataType::Int32).unwrap_err();
        assert!(err.to_string().contains("data type mismatch"));
    }

    #[test]
    fn csr_extension_try_new_invalid_storage_types() {
        let err = CsrMatrixExtension::try_new(&DataType::Int32, CsrMatrixMetadata { ncols: 3 })
            .unwrap_err();
        assert!(err.to_string().contains("expected Struct"));

        let bad_indices = DataType::Struct(
            vec![
                Field::new("indices", DataType::new_list(DataType::Int32, false), false),
                Field::new("values", DataType::new_list(DataType::Float64, false), false),
            ]
            .into(),
        );
        let err =
            CsrMatrixExtension::try_new(&bad_indices, CsrMatrixMetadata { ncols: 3 }).unwrap_err();
        assert!(err.to_string().contains("expected List<UInt32>"));

        let bad_values = DataType::Struct(
            vec![
                Field::new("indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::Float64, false),
            ]
            .into(),
        );
        let err =
            CsrMatrixExtension::try_new(&bad_values, CsrMatrixMetadata { ncols: 3 }).unwrap_err();
        assert!(err.to_string().contains("expected List for values field"));
    }

    #[test]
    fn offset_to_usize_rejects_negative() {
        let err = offset_to_usize(-1, "test").unwrap_err();
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }

    #[test]
    fn csr_view_from_extension_type_mismatch() {
        let row_ptrs = vec![0, 1];
        let col_indices = vec![0_u32];
        let values = vec![1.0_f64];
        let (field, array) =
            csr_to_extension_array("sparse", 1, row_ptrs, col_indices, values).unwrap();

        let err =
            csr_view_from_extension::<arrow_array::types::Float32Type>(&field, &array).unwrap_err();
        assert!(matches!(err, NdarrowError::TypeMismatch { .. }));
    }

    #[test]
    fn csr_view_is_zero_copy_from_columns() {
        let (indices, values) = make_columns();
        let indices_child = indices.values().as_any().downcast_ref::<UInt32Array>().unwrap();
        let values_child = values.values().as_any().downcast_ref::<Float64Array>().unwrap();

        let view = csr_view_from_columns::<Float64Type>(&indices, &values, 4).unwrap();
        assert_eq!(view.col_indices.as_ptr(), indices_child.values().as_ref().as_ptr());
        assert_eq!(view.values.as_ptr(), values_child.values().as_ref().as_ptr());
    }

    #[test]
    fn csr_view_from_columns_rejects_row_count_mismatch() {
        let indices = ListArray::from_iter_primitive::<UInt32Type, _, _>([
            Some(vec![Some(0_u32)]),
            Some(vec![Some(1_u32)]),
        ]);
        let values =
            ListArray::from_iter_primitive::<Float64Type, _, _>([Some(vec![Some(1.0_f64)])]);
        let err = csr_view_from_columns::<Float64Type>(&indices, &values, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn csr_view_from_columns_rejects_outer_nulls() {
        let indices =
            ListArray::from_iter_primitive::<UInt32Type, _, _>([Some(vec![Some(0_u32)]), None]);
        let values = ListArray::from_iter_primitive::<Float64Type, _, _>([
            Some(vec![Some(1.0_f64)]),
            Some(vec![Some(2.0_f64)]),
        ]);
        let err = csr_view_from_columns::<Float64Type>(&indices, &values, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn csr_view_from_columns_rejects_inner_nulls() {
        let indices =
            ListArray::from_iter_primitive::<UInt32Type, _, _>([Some(vec![Some(0_u32), None])]);
        let values =
            ListArray::from_iter_primitive::<Float64Type, _, _>([Some(vec![Some(1.0), Some(2.0)])]);
        let err = csr_view_from_columns::<Float64Type>(&indices, &values, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn csr_view_from_columns_rejects_type_mismatches() {
        let bad_indices =
            ListArray::from_iter_primitive::<Int32Type, _, _>([Some(vec![Some(0_i32), Some(1)])]);
        let values =
            ListArray::from_iter_primitive::<Float64Type, _, _>([Some(vec![Some(1.0), Some(2.0)])]);
        let err = csr_view_from_columns::<Float64Type>(&bad_indices, &values, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::TypeMismatch { .. }));

        let good_indices = ListArray::from_iter_primitive::<UInt32Type, _, _>([Some(vec![
            Some(0_u32),
            Some(1_u32),
        ])]);
        let values_f32 = ListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>([
            Some(vec![Some(1.0_f32), Some(2.0_f32)]),
        ]);
        let err = csr_view_from_columns::<Float64Type>(&good_indices, &values_f32, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::TypeMismatch { .. }));
    }

    #[test]
    fn csr_view_from_columns_rejects_sliced_offsets_not_zero_based() {
        let (indices, values) = make_columns();
        let indices_slice_ref = indices.slice(1, 2);
        let values_slice_ref = values.slice(1, 2);
        let indices_slice = indices_slice_ref.as_any().downcast_ref::<ListArray>().unwrap();
        let values_slice = values_slice_ref.as_any().downcast_ref::<ListArray>().unwrap();
        let err = csr_view_from_columns::<Float64Type>(indices_slice, values_slice, 4).unwrap_err();
        assert!(matches!(err, NdarrowError::InvalidMetadata { .. }));
    }
}
