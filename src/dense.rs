//! Dense vector arrays backed by Arrow's FixedSizeListArray.
//!
//! This module provides the core `DenseVectorArray` type, which wraps Arrow's
//! `FixedSizeListArray` to provide vector semantics with compile-time type safety
//! and runtime dimension checking.

use crate::error::{NarrowError, Result};
use crate::scalar::VectorScalar;
use arrow::array::{Array, ArrayRef, AsArray, FixedSizeListArray, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Float32Type, Float64Type};
use std::marker::PhantomData;
use std::sync::Arc;

/// A dense vector array backed by Arrow's `FixedSizeListArray`.
///
/// This type wraps `FixedSizeListArray<T>` where T is a floating-point type (Float32 or Float64).
/// It provides:
/// - Compile-time type safety for element types
/// - Runtime dimension validation
/// - Zero-copy conversions to ndarray and nalgebra
/// - Vector arithmetic and similarity operations
///
/// # Type Parameters
///
/// - `T`: The scalar type for vector elements (Float32Type or Float64Type)
///
/// # Invariants
///
/// - All vectors in the array have the same dimension (enforced by FixedSizeListArray)
/// - Dimension must be > 0
/// - The values array must be a PrimitiveArray of type T
///
/// # Example
///
/// ```rust,ignore
/// use narrow::DenseVectorArray;
/// use arrow::datatypes::Float32Type;
///
/// // Create from Arrow array
/// let vectors = DenseVectorArray::<Float32Type>::from_arrow(array)?;
///
/// // Check dimensions
/// assert_eq!(vectors.dimension(), 128);
/// assert_eq!(vectors.len(), 1000);
/// ```
pub struct DenseVectorArray<T: VectorScalar> {
    /// The underlying Arrow FixedSizeListArray
    inner: FixedSizeListArray,
    /// The dimension of each vector
    dimension: usize,
    /// Phantom data to track the scalar type
    _phantom: PhantomData<T>,
}

impl<T: VectorScalar> DenseVectorArray<T> {
    /// Create a new DenseVectorArray from a FixedSizeListArray.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The array is empty and dimension cannot be determined
    /// - The dimension is 0
    /// - The inner values array is not of type T
    /// - The array contains null values
    pub fn try_new(inner: FixedSizeListArray) -> Result<Self> {
        // Validate dimension
        let dimension = inner.value_length() as usize;
        if dimension == 0 {
            return Err(NarrowError::InvalidDimension(dimension));
        }

        // Validate no null values
        if inner.null_count() > 0 {
            return Err(NarrowError::NullNotSupported(
                "DenseVectorArray does not support null vectors".to_string(),
            ));
        }

        // Validate inner array type
        let values = inner.values();
        if !T::matches_array(values.as_ref()) {
            return Err(NarrowError::InvalidArrayType {
                expected: format!("{:?}", std::any::type_name::<T>()),
                actual: format!("{:?}", values.data_type()),
            });
        }

        Ok(Self {
            inner,
            dimension,
            _phantom: PhantomData,
        })
    }

    /// Create a DenseVectorArray from raw vector data.
    ///
    /// # Arguments
    ///
    /// - `vectors`: A slice of vectors, where each inner slice is a vector
    /// - `dimension`: The dimension of each vector (must match the length of each inner slice)
    ///
    /// # Errors
    ///
    /// Returns an error if any vector has a length different from `dimension`.
    pub fn from_vecs(vectors: &[Vec<T::Native>], dimension: usize) -> Result<Self> {
        // Validate all vectors have the correct dimension
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != dimension {
                return Err(NarrowError::DimensionMismatch {
                    expected: dimension,
                    actual: vec.len(),
                });
            }
        }

        // Flatten all vectors into a single array
        let flat_data: Vec<T::Native> = vectors.iter().flatten().copied().collect();

        // Create the values array
        let values = PrimitiveArray::<T>::from_iter_values(flat_data);

        // Create the FixedSizeListArray
        let field = Arc::new(Field::new("item", T::DATA_TYPE, false));
        let inner = FixedSizeListArray::new(
            field,
            dimension as i32,
            Arc::new(values),
            None, // No nulls for now
        );

        Self::try_new(inner)
    }

    /// Get the dimension of vectors in this array.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of vectors in this array.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get a reference to the underlying FixedSizeListArray.
    #[inline]
    pub fn as_arrow(&self) -> &FixedSizeListArray {
        &self.inner
    }

    /// Get the underlying values array as a typed PrimitiveArray.
    #[inline]
    pub fn values(&self) -> &PrimitiveArray<T> {
        // Safety: We validated the type in the constructor
        T::as_primitive_array(self.inner.values().as_ref())
            .expect("values array type was validated in constructor")
    }

    /// Get a single vector at the given index.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn get(&self, index: usize) -> Option<Vec<T::Native>> {
        if index >= self.len() {
            return None;
        }

        let values = self.values();
        let start = index * self.dimension;
        let end = start + self.dimension;

        Some((start..end).map(|i| values.value(i)).collect())
    }

    /// Validate that another array has the same dimension.
    pub(crate) fn validate_dimension(&self, other: &Self) -> Result<()> {
        if self.dimension != other.dimension {
            return Err(NarrowError::DimensionMismatch {
                expected: self.dimension,
                actual: other.dimension,
            });
        }
        Ok(())
    }

    /// Validate that another array has the same length.
    pub(crate) fn validate_length(&self, other: &Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(NarrowError::LengthMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(())
    }
}

// Type aliases for common cases
/// Dense vector array with Float32 elements.
pub type DenseVectorArrayF32 = DenseVectorArray<Float32Type>;

/// Dense vector array with Float64 elements.
pub type DenseVectorArrayF64 = DenseVectorArray<Float64Type>;

impl<T: VectorScalar> Clone for DenseVectorArray<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dimension: self.dimension,
            _phantom: PhantomData,
        }
    }
}

impl<T: VectorScalar> std::fmt::Debug for DenseVectorArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseVectorArray")
            .field("dimension", &self.dimension)
            .field("len", &self.len())
            .field("type", &std::any::type_name::<T>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Float32Type;

    #[test]
    fn test_create_from_vecs() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let array = DenseVectorArrayF32::from_vecs(&vectors, 3).unwrap();

        assert_eq!(array.dimension(), 3);
        assert_eq!(array.len(), 3);
        assert!(!array.is_empty());
    }

    #[test]
    fn test_get_vector() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];

        let array = DenseVectorArrayF32::from_vecs(&vectors, 3).unwrap();

        assert_eq!(array.get(0).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(array.get(1).unwrap(), vec![4.0, 5.0, 6.0]);
        assert!(array.get(2).is_none());
    }

    #[test]
    fn test_dimension_mismatch() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],  // Wrong dimension!
        ];

        let result = DenseVectorArrayF32::from_vecs(&vectors, 3);
        assert!(matches!(result, Err(NarrowError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_invalid_dimension_zero() {
        let vectors: Vec<Vec<f32>> = vec![vec![]];
        let result = DenseVectorArrayF32::from_vecs(&vectors, 0);
        assert!(matches!(result, Err(NarrowError::InvalidDimension(0))));
    }

    #[test]
    fn test_null_rejection() {
        use arrow::array::{Array, Float32Array};
        use arrow::buffer::NullBuffer;
        use arrow::datatypes::{DataType, Field};
        use std::sync::Arc;

        // Create a Float32Array with some data
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Create a FixedSizeListArray with a null value
        let field = Arc::new(Field::new("item", DataType::Float32, false));

        // Create null buffer with one null at index 1
        let null_buffer = NullBuffer::from(vec![true, false, true]);

        let array = FixedSizeListArray::new(
            field,
            2, // dimension
            Arc::new(values),
            Some(null_buffer),
        );

        // Should reject arrays with nulls
        let result = DenseVectorArrayF32::try_new(array);
        assert!(matches!(result, Err(NarrowError::NullNotSupported(_))));
    }
}
