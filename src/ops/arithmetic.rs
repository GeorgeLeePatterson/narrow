//! Arithmetic operations on dense vector arrays.
//!
//! This module provides element-wise arithmetic operations on vector arrays,
//! including addition, subtraction, and scalar multiplication.

use crate::dense::DenseVectorArray;
use crate::error::{NarrowError, Result};
use crate::scalar::VectorScalar;
use arrow::array::PrimitiveArray;
use arrow::datatypes::Field;
use arrow::array::FixedSizeListArray;
use std::sync::Arc;

impl<T: VectorScalar> DenseVectorArray<T> {
    /// Element-wise addition of two vector arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different dimensions
    /// - Arrays have different lengths
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let a = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0]], 2)?;
    /// let b = DenseVectorArrayF32::from_vecs(&[vec![3.0, 4.0]], 2)?;
    /// let sum = a.add(&b)?;
    /// // sum contains [4.0, 6.0]
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.validate_dimension(other)?;
        self.validate_length(other)?;

        let self_values = self.values();
        let other_values = other.values();

        // Element-wise addition, properly handling Arrow array offsets
        let total_elements = self.len() * self.dimension;
        let result_values: Vec<T::Native> = (0..total_elements)
            .map(|i| self_values.value(i) + other_values.value(i))
            .collect();

        let values_array = PrimitiveArray::<T>::from_iter_values(result_values);
        let field = Arc::new(Field::new("item", T::DATA_TYPE, false));
        let result_array = FixedSizeListArray::new(
            field,
            self.dimension as i32,
            Arc::new(values_array),
            None,
        );

        Self::try_new(result_array)
    }

    /// Element-wise subtraction of two vector arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different dimensions
    /// - Arrays have different lengths
    pub fn subtract(&self, other: &Self) -> Result<Self> {
        self.validate_dimension(other)?;
        self.validate_length(other)?;

        let self_values = self.values();
        let other_values = other.values();

        // Element-wise subtraction, properly handling Arrow array offsets
        let total_elements = self.len() * self.dimension;
        let result_values: Vec<T::Native> = (0..total_elements)
            .map(|i| self_values.value(i) - other_values.value(i))
            .collect();

        let values_array = PrimitiveArray::<T>::from_iter_values(result_values);
        let field = Arc::new(Field::new("item", T::DATA_TYPE, false));
        let result_array = FixedSizeListArray::new(
            field,
            self.dimension as i32,
            Arc::new(values_array),
            None,
        );

        Self::try_new(result_array)
    }

    /// Scalar multiplication of a vector array.
    ///
    /// Multiplies each element of each vector by the scalar value.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let a = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0]], 2)?;
    /// let scaled = a.scalar_multiply(2.0)?;
    /// // scaled contains [2.0, 4.0]
    /// ```
    pub fn scalar_multiply(&self, scalar: T::Native) -> Result<Self> {
        let self_values = self.values();

        // Scalar multiplication, properly handling Arrow array offsets
        let total_elements = self.len() * self.dimension;
        let result_values: Vec<T::Native> = (0..total_elements)
            .map(|i| self_values.value(i) * scalar)
            .collect();

        let values_array = PrimitiveArray::<T>::from_iter_values(result_values);
        let field = Arc::new(Field::new("item", T::DATA_TYPE, false));
        let result_array = FixedSizeListArray::new(
            field,
            self.dimension as i32,
            Arc::new(values_array),
            None,
        );

        Self::try_new(result_array)
    }

    /// Element-wise (Hadamard) product of two vector arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different dimensions
    /// - Arrays have different lengths
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        self.validate_dimension(other)?;
        self.validate_length(other)?;

        let self_values = self.values();
        let other_values = other.values();

        // Element-wise multiplication, properly handling Arrow array offsets
        let total_elements = self.len() * self.dimension;
        let result_values: Vec<T::Native> = (0..total_elements)
            .map(|i| self_values.value(i) * other_values.value(i))
            .collect();

        let values_array = PrimitiveArray::<T>::from_iter_values(result_values);
        let field = Arc::new(Field::new("item", T::DATA_TYPE, false));
        let result_array = FixedSizeListArray::new(
            field,
            self.dimension as i32,
            Arc::new(values_array),
            None,
        );

        Self::try_new(result_array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseVectorArrayF32;
    use approx::assert_relative_eq;

    #[test]
    fn test_add() {
        let a = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ], 3).unwrap();

        let b = DenseVectorArrayF32::from_vecs(&[
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ], 3).unwrap();

        let result = a.add(&b).unwrap();

        assert_eq!(result.get(0).unwrap(), vec![11.0, 22.0, 33.0]);
        assert_eq!(result.get(1).unwrap(), vec![44.0, 55.0, 66.0]);
    }

    #[test]
    fn test_subtract() {
        let a = DenseVectorArrayF32::from_vecs(&[
            vec![10.0, 20.0, 30.0],
        ], 3).unwrap();

        let b = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
        ], 3).unwrap();

        let result = a.subtract(&b).unwrap();

        assert_eq!(result.get(0).unwrap(), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_scalar_multiply() {
        let a = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ], 3).unwrap();

        let result = a.scalar_multiply(2.0).unwrap();

        assert_eq!(result.get(0).unwrap(), vec![2.0, 4.0, 6.0]);
        assert_eq!(result.get(1).unwrap(), vec![8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_multiply() {
        let a = DenseVectorArrayF32::from_vecs(&[
            vec![2.0, 3.0, 4.0],
        ], 3).unwrap();

        let b = DenseVectorArrayF32::from_vecs(&[
            vec![5.0, 6.0, 7.0],
        ], 3).unwrap();

        let result = a.multiply(&b).unwrap();

        assert_eq!(result.get(0).unwrap(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0]], 2).unwrap();
        let b = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0, 3.0]], 3).unwrap();

        let result = a.add(&b);
        assert!(matches!(result, Err(NarrowError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_length_mismatch() {
        let a = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0]], 2).unwrap();
        let b = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ], 2).unwrap();

        let result = a.add(&b);
        assert!(matches!(result, Err(NarrowError::LengthMismatch { .. })));
    }

    #[test]
    fn test_add_sliced_arrays() {
        use arrow::array::Array;

        // Create arrays and slice them
        let a = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ], 2).unwrap();

        let b = DenseVectorArrayF32::from_vecs(&[
            vec![10.0, 20.0],
            vec![30.0, 40.0],
            vec![50.0, 60.0],
        ], 2).unwrap();

        // Slice both arrays to take only the middle element
        let a_sliced = a.as_arrow().slice(1, 1);
        let b_sliced = b.as_arrow().slice(1, 1);

        let a_narrow = DenseVectorArrayF32::try_new(a_sliced).unwrap();
        let b_narrow = DenseVectorArrayF32::try_new(b_sliced).unwrap();

        // Perform addition on sliced arrays
        let result = a_narrow.add(&b_narrow).unwrap();

        // Should get [33.0, 44.0] (from [3.0, 4.0] + [30.0, 40.0])
        assert_eq!(result.len(), 1);
        assert_eq!(result.get(0).unwrap(), vec![33.0, 44.0]);
    }

    #[test]
    fn test_scalar_multiply_sliced() {
        use arrow::array::Array;

        let a = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ], 2).unwrap();

        // Slice to get the last element
        let a_sliced = a.as_arrow().slice(2, 1);
        let a_narrow = DenseVectorArrayF32::try_new(a_sliced).unwrap();

        let result = a_narrow.scalar_multiply(2.0).unwrap();

        // Should get [10.0, 12.0] (from [5.0, 6.0] * 2.0)
        assert_eq!(result.get(0).unwrap(), vec![10.0, 12.0]);
    }
}
