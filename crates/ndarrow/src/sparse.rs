//! Sparse Arrow/ndarray bridge utilities.
//!
//! This module defines the custom `ndarrow.csr_matrix` extension type and
//! zero-copy sparse inbound conversions.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int32Array, ListArray, PrimitiveArray, StructArray,
    UInt32Array, types::ArrowPrimitiveType,
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

/// `ndarrow.csr_matrix_batch` extension type.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrixBatchExtension {
    value_type: DataType,
}

impl CsrMatrixBatchExtension {
    /// Returns the value type stored in the CSR values buffer.
    #[must_use]
    pub fn value_type(&self) -> &DataType {
        &self.value_type
    }

    fn expected_storage_type(&self) -> DataType {
        DataType::Struct(
            vec![
                Field::new(
                    "shape",
                    DataType::new_fixed_size_list(DataType::Int32, 2, false),
                    false,
                ),
                Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false),
                Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::new_list(self.value_type.clone(), false), false),
            ]
            .into(),
        )
    }
}

impl ExtensionType for CsrMatrixBatchExtension {
    type Metadata = ();

    const NAME: &'static str = "ndarrow.csr_matrix_batch";

    fn metadata(&self) -> &Self::Metadata {
        &()
    }

    fn serialize_metadata(&self) -> Option<String> {
        None
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata, ArrowError> {
        if metadata.is_some() {
            return Err(ArrowError::InvalidArgumentError(
                "ndarrow.csr_matrix_batch expects no metadata".to_owned(),
            ));
        }
        Ok(())
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        let expected = self.expected_storage_type();
        data_type.equals_datatype(&expected).then_some(()).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "ndarrow.csr_matrix_batch data type mismatch, expected {expected}, found {data_type}"
            ))
        })
    }

    fn try_new(data_type: &DataType, _metadata: Self::Metadata) -> Result<Self, ArrowError> {
        match data_type {
            DataType::Struct(fields)
                if fields.len() == 4
                    && matches!(fields.find("shape"), Some((0, _)))
                    && matches!(fields.find("row_ptrs"), Some((1, _)))
                    && matches!(fields.find("col_indices"), Some((2, _)))
                    && matches!(fields.find("values"), Some((3, _))) =>
            {
                let shape_field = &fields[0];
                match shape_field.data_type() {
                    DataType::FixedSizeList(inner, 2) if inner.data_type() == &DataType::Int32 => {}
                    other => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "ndarrow.csr_matrix_batch data type mismatch, expected FixedSizeList<Int32>(2) for shape field, found {other}"
                        )));
                    }
                }

                let row_ptrs_field = &fields[1];
                match row_ptrs_field.data_type() {
                    DataType::List(inner) if inner.data_type() == &DataType::Int32 => {}
                    other => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "ndarrow.csr_matrix_batch data type mismatch, expected List<Int32> for row_ptrs field, found {other}"
                        )));
                    }
                }

                let col_indices_field = &fields[2];
                match col_indices_field.data_type() {
                    DataType::List(inner) if inner.data_type() == &DataType::UInt32 => {}
                    other => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "ndarrow.csr_matrix_batch data type mismatch, expected List<UInt32> for col_indices field, found {other}"
                        )));
                    }
                }

                let values_field = &fields[3];
                let value_type = match values_field.data_type() {
                    DataType::List(inner) => inner.data_type().clone(),
                    other => {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "ndarrow.csr_matrix_batch data type mismatch, expected List for values field, found {other}"
                        )));
                    }
                };

                let extension = Self { value_type };
                extension.supports_data_type(data_type)?;
                Ok(extension)
            }
            other => Err(ArrowError::InvalidArgumentError(format!(
                "ndarrow.csr_matrix_batch data type mismatch, expected Struct{{shape,row_ptrs,col_indices,values}}, found {other}"
            ))),
        }
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

/// Iterator over per-row sparse matrix views for `ndarrow.csr_matrix_batch`.
pub struct CsrMatrixBatchIter<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    index:            usize,
    len:              usize,
    shapes:           &'a Int32Array,
    row_ptrs:         &'a ListArray,
    row_ptr_values:   &'a Int32Array,
    col_indices:      &'a ListArray,
    col_index_values: &'a UInt32Array,
    value_values:     &'a PrimitiveArray<T>,
}

struct PackedCsrBatch<T> {
    shapes:           Vec<i32>,
    row_ptr_offsets:  Vec<i32>,
    row_ptr_values:   Vec<i32>,
    nnz_offsets:      Vec<i32>,
    col_index_values: Vec<u32>,
    values:           Vec<T>,
}

impl<'a, T> Iterator for CsrMatrixBatchIter<'a, T>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    type Item = Result<(usize, CsrView<'a, T::Native>), NdarrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }

        let row = self.index;
        self.index += 1;

        let Some(shape_start) = row.checked_mul(2) else {
            return Some(Err(NdarrowError::ShapeMismatch {
                message: format!("shape index overflow at row {row}"),
            }));
        };
        let raw_nrows = self.shapes.value(shape_start);
        let raw_ncols = self.shapes.value(shape_start + 1);
        let Ok(nrows) = usize::try_from(raw_nrows) else {
            return Some(Err(NdarrowError::ShapeMismatch {
                message: format!("negative sparse matrix row count at row {row}: {raw_nrows}"),
            }));
        };
        let Ok(ncols) = usize::try_from(raw_ncols) else {
            return Some(Err(NdarrowError::ShapeMismatch {
                message: format!("negative sparse matrix column count at row {row}: {raw_ncols}"),
            }));
        };

        let row_ptr_offsets = self.row_ptrs.value_offsets();
        let Ok(row_ptr_start) = usize::try_from(row_ptr_offsets[row]) else {
            return Some(Err(NdarrowError::InvalidMetadata {
                message: format!("negative row_ptrs offset at row {row}: {}", row_ptr_offsets[row]),
            }));
        };
        let Ok(row_ptr_end) = usize::try_from(row_ptr_offsets[row + 1]) else {
            return Some(Err(NdarrowError::InvalidMetadata {
                message: format!(
                    "negative row_ptrs end offset at row {row}: {}",
                    row_ptr_offsets[row + 1]
                ),
            }));
        };

        let nnz_offsets = self.col_indices.value_offsets();
        let Ok(nnz_start) = usize::try_from(nnz_offsets[row]) else {
            return Some(Err(NdarrowError::InvalidMetadata {
                message: format!("negative nnz offset at row {row}: {}", nnz_offsets[row]),
            }));
        };
        let Ok(nnz_end) = usize::try_from(nnz_offsets[row + 1]) else {
            return Some(Err(NdarrowError::InvalidMetadata {
                message: format!("negative nnz end offset at row {row}: {}", nnz_offsets[row + 1]),
            }));
        };

        let row_ptr_slice = &self.row_ptr_values.values().as_ref()[row_ptr_start..row_ptr_end];
        let col_indices_slice = &self.col_index_values.values().as_ref()[nnz_start..nnz_end];
        let value_slice = &self.value_values.values().as_ref()[nnz_start..nnz_end];

        if let Err(err) = validate_csr_batch_row(
            row,
            nrows,
            row_ptr_slice,
            col_indices_slice.len(),
            value_slice.len(),
        ) {
            return Some(Err(err));
        }

        Some(Ok((
            row,
            CsrView {
                nrows,
                ncols,
                row_ptrs: row_ptr_slice,
                col_indices: col_indices_slice,
                values: value_slice,
            },
        )))
    }
}

fn offset_to_usize(offset: i32, context: &str) -> Result<usize, NdarrowError> {
    usize::try_from(offset).map_err(|_| NdarrowError::InvalidMetadata {
        message: format!("invalid negative offset in {context}: {offset}"),
    })
}

fn struct_column_as_fixed_size_list<'a>(
    array: &'a StructArray,
    index: usize,
    column_name: &str,
) -> Result<&'a FixedSizeListArray, NdarrowError> {
    array.column(index).as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
        NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected struct column '{column_name}' as FixedSizeListArray, found {}",
                array.column(index).data_type()
            ),
        }
    })
}

fn struct_column_as_list<'a>(
    array: &'a StructArray,
    index: usize,
    column_name: &str,
) -> Result<&'a ListArray, NdarrowError> {
    array.column(index).as_any().downcast_ref::<ListArray>().ok_or_else(|| {
        NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected struct column '{column_name}' as ListArray, found {}",
                array.column(index).data_type()
            ),
        }
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

fn list_as_i32_values<'a>(
    array: &'a ListArray,
    column_name: &str,
) -> Result<&'a Int32Array, NdarrowError> {
    array
        .values()
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| NdarrowError::TypeMismatch {
            message: format!(
                "column '{column_name}' must be List<Int32>, found {}",
                array.values().data_type()
            ),
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

    let indices = struct_column_as_list(array, 0, "indices")?;
    let values = struct_column_as_list(array, 1, "values")?;

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

fn validate_csr_batch_row(
    row: usize,
    nrows: usize,
    row_ptrs: &[i32],
    col_indices_len: usize,
    values_len: usize,
) -> Result<(), NdarrowError> {
    validate_csr_parts(row_ptrs, col_indices_len, values_len)?;

    let expected_row_ptr_len = nrows.checked_add(1).ok_or_else(|| NdarrowError::ShapeMismatch {
        message: format!("sparse matrix row count overflows usize at row {row}: {nrows}"),
    })?;
    if row_ptrs.len() != expected_row_ptr_len {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "row {row} CSR row_ptr length mismatch: expected {expected_row_ptr_len}, found {}",
                row_ptrs.len()
            ),
        });
    }

    Ok(())
}

/// Creates an iterator over per-row zero-copy sparse matrix views for
/// `ndarrow.csr_matrix_batch`.
///
/// # Does not allocate
///
/// This borrows Arrow offsets and value buffers directly.
///
/// # Errors
///
/// Returns an error when extension/type/null invariants are violated.
pub fn csr_matrix_batch_iter<'a, T>(
    field: &Field,
    array: &'a StructArray,
) -> Result<CsrMatrixBatchIter<'a, T>, NdarrowError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    if array.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: array.null_count() });
    }

    let extension =
        field.try_extension_type::<CsrMatrixBatchExtension>().map_err(NdarrowError::from)?;
    extension.supports_data_type(array.data_type()).map_err(NdarrowError::from)?;

    let shapes = struct_column_as_fixed_size_list(array, 0, "shape")?;
    if shapes.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: shapes.null_count() });
    }

    let row_ptrs = struct_column_as_list(array, 1, "row_ptrs")?;
    if row_ptrs.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: row_ptrs.null_count() });
    }

    let col_indices = struct_column_as_list(array, 2, "col_indices")?;
    if col_indices.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: col_indices.null_count() });
    }

    let values = struct_column_as_list(array, 3, "values")?;
    if values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: values.null_count() });
    }

    if col_indices.value_offsets() != values.value_offsets() {
        return Err(NdarrowError::SparseOffsetMismatch);
    }

    let shape_values = shapes.values().as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
        NdarrowError::InnerTypeMismatch {
            message: format!(
                "expected batched CSR shape values as Int32, found {}",
                shapes.values().data_type()
            ),
        }
    })?;
    if shape_values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: shape_values.null_count() });
    }

    let row_ptr_values = list_as_i32_values(row_ptrs, "row_ptrs")?;
    if row_ptr_values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: row_ptr_values.null_count() });
    }

    let col_index_values = list_as_u32_values(col_indices, "col_indices")?;
    if col_index_values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: col_index_values.null_count() });
    }

    let value_values = list_as_t_values::<T>(values, "values")?;
    if value_values.null_count() > 0 {
        return Err(NdarrowError::NullsPresent { null_count: value_values.null_count() });
    }

    Ok(CsrMatrixBatchIter {
        index: 0,
        len: array.len(),
        shapes: shape_values,
        row_ptrs,
        row_ptr_values,
        col_indices,
        col_index_values,
        value_values,
    })
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

fn pack_csr_batch_parts<T>(
    shapes: Vec<[usize; 2]>,
    row_ptr_batches: Vec<Vec<i32>>,
    col_index_batches: Vec<Vec<u32>>,
    value_batches: Vec<Vec<T>>,
) -> Result<PackedCsrBatch<T>, NdarrowError>
where
    T: NdarrowElement,
{
    let batch_len = shapes.len();
    if row_ptr_batches.len() != batch_len
        || col_index_batches.len() != batch_len
        || value_batches.len() != batch_len
    {
        return Err(NdarrowError::ShapeMismatch {
            message: format!(
                "batched CSR input length mismatch: shapes={}, row_ptrs={}, col_indices={}, values={}",
                batch_len,
                row_ptr_batches.len(),
                col_index_batches.len(),
                value_batches.len()
            ),
        });
    }

    let mut packed_shapes = Vec::with_capacity(batch_len.checked_mul(2).ok_or_else(|| {
        NdarrowError::ShapeMismatch {
            message: format!("batched CSR shape count overflows usize: {batch_len}"),
        }
    })?);
    let mut row_ptr_offsets = Vec::with_capacity(batch_len + 1);
    row_ptr_offsets.push(0_i32);
    let mut packed_row_ptrs = Vec::new();
    let mut nnz_offsets = Vec::with_capacity(batch_len + 1);
    nnz_offsets.push(0_i32);
    let mut packed_col_indices = Vec::new();
    let mut packed_values = Vec::new();
    let mut running_row_ptr_offset = 0_i32;
    let mut running_nnz_offset = 0_i32;

    for (row, (((shape, row_ptrs), col_indices), values)) in shapes
        .into_iter()
        .zip(row_ptr_batches)
        .zip(col_index_batches)
        .zip(value_batches)
        .enumerate()
    {
        let [nrows, ncols] = shape;
        validate_csr_batch_row(row, nrows, &row_ptrs, col_indices.len(), values.len())?;

        packed_shapes.push(i32::try_from(nrows).map_err(|_| NdarrowError::ShapeMismatch {
            message: format!("sparse matrix row count exceeds i32 limits at row {row}: {nrows}"),
        })?);
        packed_shapes.push(i32::try_from(ncols).map_err(|_| NdarrowError::ShapeMismatch {
            message: format!("sparse matrix column count exceeds i32 limits at row {row}: {ncols}"),
        })?);

        let row_ptr_len =
            i32::try_from(row_ptrs.len()).map_err(|_| NdarrowError::ShapeMismatch {
                message: format!(
                    "row_ptr length exceeds i32 limits at row {row}: {}",
                    row_ptrs.len()
                ),
            })?;
        running_row_ptr_offset =
            running_row_ptr_offset.checked_add(row_ptr_len).ok_or_else(|| {
                NdarrowError::ShapeMismatch {
                    message: "batched CSR row_ptr offsets exceed i32 limits".to_owned(),
                }
            })?;
        row_ptr_offsets.push(running_row_ptr_offset);
        packed_row_ptrs.extend(row_ptrs);

        let nnz_len =
            i32::try_from(col_indices.len()).map_err(|_| NdarrowError::ShapeMismatch {
                message: format!("nnz exceeds i32 limits at row {row}: {}", col_indices.len()),
            })?;
        running_nnz_offset =
            running_nnz_offset.checked_add(nnz_len).ok_or_else(|| NdarrowError::ShapeMismatch {
                message: "batched CSR nnz offsets exceed i32 limits".to_owned(),
            })?;
        nnz_offsets.push(running_nnz_offset);
        packed_col_indices.extend(col_indices);
        packed_values.extend(values);
    }

    Ok(PackedCsrBatch {
        shapes: packed_shapes,
        row_ptr_offsets,
        row_ptr_values: packed_row_ptrs,
        nnz_offsets,
        col_index_values: packed_col_indices,
        values: packed_values,
    })
}

fn csr_batch_struct_array<T>(packed: PackedCsrBatch<T>) -> StructArray
where
    T: NdarrowElement,
{
    let shape_values = Int32Array::new(ScalarBuffer::from(packed.shapes), None);
    let shape_column: ArrayRef = Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Int32, false)),
        2,
        Arc::new(shape_values),
        None,
    ));

    let row_ptr_value_array = Int32Array::new(ScalarBuffer::from(packed.row_ptr_values), None);
    let row_ptrs_column: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new_list_field(DataType::Int32, false)),
        OffsetBuffer::new(ScalarBuffer::from(packed.row_ptr_offsets)),
        Arc::new(row_ptr_value_array),
        None,
    ));

    let col_index_value_array = UInt32Array::new(ScalarBuffer::from(packed.col_index_values), None);
    let col_indices_column: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new_list_field(DataType::UInt32, false)),
        OffsetBuffer::new(ScalarBuffer::from(packed.nnz_offsets.clone())),
        Arc::new(col_index_value_array),
        None,
    ));

    let flat_value_array =
        PrimitiveArray::<T::ArrowType>::new(ScalarBuffer::from(packed.values), None);
    let values_column: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new_list_field(T::data_type(), false)),
        OffsetBuffer::new(ScalarBuffer::from(packed.nnz_offsets)),
        Arc::new(flat_value_array),
        None,
    ));

    let struct_fields = vec![
        Field::new("shape", shape_column.data_type().clone(), false),
        Field::new("row_ptrs", row_ptrs_column.data_type().clone(), false),
        Field::new("col_indices", col_indices_column.data_type().clone(), false),
        Field::new("values", values_column.data_type().clone(), false),
    ];
    StructArray::new(
        struct_fields.clone().into(),
        vec![shape_column, row_ptrs_column, col_indices_column, values_column],
        None,
    )
}

/// Builds an Arrow `StructArray` plus extension field for `ndarrow.csr_matrix_batch`.
///
/// # Allocation
///
/// This function allocates Arrow array wrappers and nested offset/value
/// containers while transferring ownership of the provided vectors.
///
/// # Errors
///
/// Returns an error if batch lengths, CSR row invariants, or Arrow shape limits
/// are violated.
pub fn csr_batch_to_extension_array<T>(
    field_name: &str,
    shapes: Vec<[usize; 2]>,
    row_ptrs: Vec<Vec<i32>>,
    col_indices: Vec<Vec<u32>>,
    values: Vec<Vec<T>>,
) -> Result<(Field, StructArray), NdarrowError>
where
    T: NdarrowElement,
{
    let packed = pack_csr_batch_parts(shapes, row_ptrs, col_indices, values)?;
    let struct_array = csr_batch_struct_array(packed);

    let extension = CsrMatrixBatchExtension::try_new(struct_array.data_type(), ())
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
    use arrow_buffer::NullBuffer;
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

    fn csr_batch_storage_type(value_type: DataType) -> DataType {
        DataType::Struct(
            vec![
                Field::new(
                    "shape",
                    DataType::new_fixed_size_list(DataType::Int32, 2, false),
                    false,
                ),
                Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false),
                Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false),
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
    fn csr_batch_to_extension_array_roundtrip() {
        let shapes = vec![[2_usize, 4_usize], [3_usize, 5_usize]];
        let row_ptrs = vec![vec![0_i32, 1, 2], vec![0_i32, 2, 3, 4]];
        let col_indices = vec![vec![0_u32, 3_u32], vec![1_u32, 4_u32, 0_u32, 2_u32]];
        let values = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64]];

        let (field, array) =
            csr_batch_to_extension_array("sparse_batch", shapes, row_ptrs, col_indices, values)
                .unwrap();

        assert_eq!(field.extension_type_name(), Some(CsrMatrixBatchExtension::NAME));
        assert_eq!(
            field.metadata().get(EXTENSION_TYPE_NAME_KEY).map(String::as_str),
            Some(CsrMatrixBatchExtension::NAME)
        );

        let mut iter = csr_matrix_batch_iter::<Float64Type>(&field, &array).unwrap();
        let (row0, view0) = iter.next().unwrap().unwrap();
        assert_eq!(row0, 0);
        assert_eq!(view0.nrows, 2);
        assert_eq!(view0.ncols, 4);
        assert_eq!(view0.row_ptrs, &[0, 1, 2]);
        assert_eq!(view0.col_indices, &[0, 3]);
        assert_abs_diff_eq!(view0.values[0], 1.0);
        assert_abs_diff_eq!(view0.values[1], 2.0);

        let (row1, view1) = iter.next().unwrap().unwrap();
        assert_eq!(row1, 1);
        assert_eq!(view1.nrows, 3);
        assert_eq!(view1.ncols, 5);
        assert_eq!(view1.row_ptrs, &[0, 2, 3, 4]);
        assert_eq!(view1.col_indices, &[1, 4, 0, 2]);
        assert_abs_diff_eq!(view1.values[0], 3.0);
        assert_abs_diff_eq!(view1.values[3], 6.0);

        assert!(iter.next().is_none());
    }

    #[test]
    fn csr_batch_extension_accessors_and_metadata() {
        let data_type = csr_batch_storage_type(DataType::Float64);
        let extension = CsrMatrixBatchExtension::try_new(&data_type, ()).unwrap();
        assert_eq!(extension.value_type(), &DataType::Float64);
        assert_eq!(extension.metadata(), &());
    }

    #[test]
    fn csr_batch_extension_rejects_metadata_payload() {
        let err = CsrMatrixBatchExtension::deserialize_metadata(Some("unexpected")).unwrap_err();
        assert!(err.to_string().contains("expects no metadata"));
    }

    #[test]
    fn csr_batch_extension_supports_data_type_mismatch() {
        let data_type = csr_batch_storage_type(DataType::Float64);
        let extension = CsrMatrixBatchExtension::try_new(&data_type, ()).unwrap();
        let err = extension.supports_data_type(&DataType::Int32).unwrap_err();
        assert!(err.to_string().contains("data type mismatch"));
    }

    #[test]
    fn csr_batch_extension_try_new_invalid_storage_types() {
        let err = CsrMatrixBatchExtension::try_new(&DataType::Int32, ()).unwrap_err();
        assert!(err.to_string().contains("expected Struct"));

        let bad_shape = DataType::Struct(
            vec![
                Field::new("shape", DataType::new_list(DataType::Int32, false), false),
                Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false),
                Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::new_list(DataType::Float64, false), false),
            ]
            .into(),
        );
        let err = CsrMatrixBatchExtension::try_new(&bad_shape, ()).unwrap_err();
        assert!(err.to_string().contains("FixedSizeList<Int32>(2)"));
    }

    #[test]
    fn csr_batch_extension_try_new_rejects_wrong_field_types() {
        let bad_row_ptrs = DataType::Struct(
            vec![
                Field::new(
                    "shape",
                    DataType::new_fixed_size_list(DataType::Int32, 2, false),
                    false,
                ),
                Field::new("row_ptrs", DataType::new_list(DataType::UInt32, false), false),
                Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::new_list(DataType::Float64, false), false),
            ]
            .into(),
        );
        let err = CsrMatrixBatchExtension::try_new(&bad_row_ptrs, ()).unwrap_err();
        assert!(err.to_string().contains("List<Int32> for row_ptrs"));

        let bad_col_indices = DataType::Struct(
            vec![
                Field::new(
                    "shape",
                    DataType::new_fixed_size_list(DataType::Int32, 2, false),
                    false,
                ),
                Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false),
                Field::new("col_indices", DataType::new_list(DataType::Int32, false), false),
                Field::new("values", DataType::new_list(DataType::Float64, false), false),
            ]
            .into(),
        );
        let err = CsrMatrixBatchExtension::try_new(&bad_col_indices, ()).unwrap_err();
        assert!(err.to_string().contains("List<UInt32> for col_indices"));

        let bad_values = DataType::Struct(
            vec![
                Field::new(
                    "shape",
                    DataType::new_fixed_size_list(DataType::Int32, 2, false),
                    false,
                ),
                Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false),
                Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::Float64, false),
            ]
            .into(),
        );
        let err = CsrMatrixBatchExtension::try_new(&bad_values, ()).unwrap_err();
        assert!(err.to_string().contains("expected List for values field"));
    }

    #[test]
    fn csr_extension_try_new_rejects_wrong_field_types() {
        let bad_indices = DataType::Struct(
            vec![
                Field::new("indices", DataType::new_list(DataType::Int32, false), false),
                Field::new("values", DataType::new_list(DataType::Float64, false), false),
            ]
            .into(),
        );
        let err =
            CsrMatrixExtension::try_new(&bad_indices, CsrMatrixMetadata { ncols: 2 }).unwrap_err();
        assert!(err.to_string().contains("List<UInt32> for indices field"));

        let bad_values = DataType::Struct(
            vec![
                Field::new("indices", DataType::new_list(DataType::UInt32, false), false),
                Field::new("values", DataType::Float64, false),
            ]
            .into(),
        );
        let err =
            CsrMatrixExtension::try_new(&bad_values, CsrMatrixMetadata { ncols: 2 }).unwrap_err();
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
    fn csr_view_from_columns_rejects_value_nulls_and_nnz_length_mismatch() {
        let good_indices =
            ListArray::from_iter_primitive::<UInt32Type, _, _>([Some(vec![Some(0_u32)])]);
        let outer_null_values =
            ListArray::from_iter_primitive::<Float64Type, _, _>([None::<Vec<Option<f64>>>]);
        let err =
            csr_view_from_columns::<Float64Type>(&good_indices, &outer_null_values, 2).unwrap_err();
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));

        let inner_null_values =
            ListArray::from_iter_primitive::<Float64Type, _, _>([Some(vec![Some(1.0_f64), None])]);
        let two_index_row = ListArray::from_iter_primitive::<UInt32Type, _, _>([Some(vec![
            Some(0_u32),
            Some(1_u32),
        ])]);
        let err = csr_view_from_columns::<Float64Type>(&two_index_row, &inner_null_values, 2)
            .unwrap_err();
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));

        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 1_i32]));
        let mismatched_indices = ListArray::new(
            Arc::new(Field::new_list_field(DataType::UInt32, false)),
            offsets.clone(),
            Arc::new(UInt32Array::new(ScalarBuffer::from(vec![0_u32]), None)),
            None,
        );
        let mismatched_values = ListArray::new(
            Arc::new(Field::new_list_field(DataType::Float64, false)),
            offsets,
            Arc::new(Float64Array::new(ScalarBuffer::from(vec![1.0_f64, 2.0_f64]), None)),
            None,
        );
        let err = csr_view_from_columns::<Float64Type>(&mismatched_indices, &mismatched_values, 2)
            .unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
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

    #[test]
    fn csr_matrix_batch_iter_is_zero_copy() {
        let shapes = vec![[2_usize, 4_usize], [1_usize, 3_usize]];
        let row_ptrs = vec![vec![0_i32, 1, 2], vec![0_i32, 1]];
        let col_indices = vec![vec![0_u32, 3_u32], vec![2_u32]];
        let values = vec![vec![1.0_f64, 2.0_f64], vec![3.0_f64]];
        let (field, array) =
            csr_batch_to_extension_array("sparse_batch", shapes, row_ptrs, col_indices, values)
                .unwrap();

        let shape_child = array
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let row_ptr_child = array
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let col_index_child = array
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let value_child = array
            .column(3)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let mut iter = csr_matrix_batch_iter::<Float64Type>(&field, &array).unwrap();
        let (_row0, view0) = iter.next().unwrap().unwrap();
        let (_row1, view1) = iter.next().unwrap().unwrap();

        assert_eq!(view0.row_ptrs.as_ptr(), row_ptr_child.values().as_ref().as_ptr());
        assert_eq!(view0.col_indices.as_ptr(), col_index_child.values().as_ref().as_ptr());
        assert_eq!(view0.values.as_ptr(), value_child.values().as_ref().as_ptr());
        assert_eq!(
            shape_child.values().as_ref().as_ptr(),
            array
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .values()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .as_ref()
                .as_ptr()
        );
        assert_eq!(view1.row_ptrs.as_ptr(), row_ptr_child.values().as_ref()[3..].as_ptr());
        assert_eq!(view1.col_indices.as_ptr(), col_index_child.values().as_ref()[2..].as_ptr());
        assert_eq!(view1.values.as_ptr(), value_child.values().as_ref()[2..].as_ptr());
    }

    #[test]
    fn csr_matrix_batch_iter_rejects_bad_row_ptr_length() {
        let shapes = vec![[2_usize, 4_usize]];
        let row_ptrs = vec![vec![0_i32, 2_i32]];
        let col_indices = vec![vec![0_u32, 3_u32]];
        let values = vec![vec![1.0_f64, 2.0_f64]];
        let err =
            csr_batch_to_extension_array("sparse_batch", shapes, row_ptrs, col_indices, values)
                .unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn csr_matrix_batch_iter_rejects_negative_shape_components() {
        let shape_values = Int32Array::new(ScalarBuffer::from(vec![-1_i32, 3_i32]), None);
        let shape: ArrayRef = Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Int32, false)),
            2,
            Arc::new(shape_values),
            None,
        ));

        let row_ptr_values = Int32Array::new(ScalarBuffer::from(vec![0_i32]), None);
        let row_ptrs: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(DataType::Int32, false)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 1_i32])),
            Arc::new(row_ptr_values),
            None,
        ));

        let col_indices: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(DataType::UInt32, false)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 0_i32])),
            Arc::new(UInt32Array::new(ScalarBuffer::from(vec![]), None)),
            None,
        ));
        let values: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(DataType::Float64, false)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 0_i32])),
            Arc::new(Float64Array::new(ScalarBuffer::from(vec![]), None)),
            None,
        ));

        let struct_array = StructArray::new(
            vec![
                Field::new("shape", shape.data_type().clone(), false),
                Field::new("row_ptrs", row_ptrs.data_type().clone(), false),
                Field::new("col_indices", col_indices.data_type().clone(), false),
                Field::new("values", values.data_type().clone(), false),
            ]
            .into(),
            vec![shape, row_ptrs, col_indices, values],
            None,
        );
        let mut field = Field::new("sparse_batch", struct_array.data_type().clone(), false);
        field
            .try_with_extension_type(
                CsrMatrixBatchExtension::try_new(struct_array.data_type(), ()).unwrap(),
            )
            .unwrap();

        let mut iter = csr_matrix_batch_iter::<Float64Type>(&field, &struct_array).unwrap();
        let err = iter.next().unwrap().unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }

    #[test]
    fn csr_matrix_batch_iter_rejects_outer_nulls() {
        let shapes = vec![[1_usize, 2_usize]];
        let row_ptrs = vec![vec![0_i32, 1_i32]];
        let col_indices = vec![vec![0_u32]];
        let values = vec![vec![1.0_f64]];
        let (field, array) =
            csr_batch_to_extension_array("sparse_batch", shapes, row_ptrs, col_indices, values)
                .unwrap();

        let with_nulls = StructArray::new(
            array.fields().clone(),
            array.columns().to_vec(),
            Some(NullBuffer::from(vec![false])),
        );
        let result = csr_matrix_batch_iter::<Float64Type>(&field, &with_nulls);
        assert!(result.is_err(), "outer nulls must fail");
        let err = result.err().expect("outer nulls must fail");
        assert!(matches!(err, NdarrowError::NullsPresent { .. }));
    }

    #[test]
    fn csr_batch_to_extension_array_rejects_batch_length_and_shape_overflow() {
        let err = csr_batch_to_extension_array::<f64>(
            "sparse_batch",
            vec![[1_usize, 2_usize]],
            Vec::new(),
            vec![vec![0_u32]],
            vec![vec![1.0_f64]],
        )
        .unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));

        let err = csr_batch_to_extension_array::<f64>(
            "sparse_batch",
            vec![[usize::MAX, 1_usize]],
            vec![vec![0_i32]],
            vec![Vec::<u32>::new()],
            vec![Vec::<f64>::new()],
        )
        .unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));

        let err = csr_batch_to_extension_array::<f64>(
            "sparse_batch",
            vec![[0_usize, i32::MAX as usize + 1]],
            vec![vec![0_i32]],
            vec![Vec::<u32>::new()],
            vec![Vec::<f64>::new()],
        )
        .unwrap_err();
        assert!(matches!(err, NdarrowError::ShapeMismatch { .. }));
    }
}
