//! Conversions to nalgebra.
//!
//! This module provides conversions from DenseVectorArray to nalgebra's DMatrix and DVectorSlice,
//! enabling linear algebra operations.

use crate::dense::DenseVectorArray;
use crate::error::{NarrowError, Result};
use crate::scalar::VectorScalar;
use nalgebra::{DMatrix, DVector, DVectorView};

impl<T: VectorScalar> DenseVectorArray<T>
where
    T::Native: nalgebra::Scalar + Copy,
{
    /// Create a nalgebra DMatrix from this vector array.
    ///
    /// Each vector becomes a row in the matrix. This operation copies the data.
    ///
    /// # Returns
    ///
    /// A DMatrix with shape `(num_vectors, dimension)`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vectors = DenseVectorArrayF32::from_vecs(&[
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0],
    /// ], 2)?;
    ///
    /// let matrix = vectors.to_nalgebra_matrix()?;
    /// assert_eq!(matrix.nrows(), 2);
    /// assert_eq!(matrix.ncols(), 2);
    /// ```
    pub fn to_nalgebra_matrix(&self) -> Result<DMatrix<T::Native>> {
        let values = self.values();

        // Handle Arrow array offsets for sliced arrays
        let offset = values.offset();
        let total_elements = self.len() * self.dimension;
        let buffer_slice = values.values();
        let data_slice = &buffer_slice[offset..offset + total_elements];

        // nalgebra matrices are column-major by default, but we can create from row-major data
        Ok(DMatrix::from_row_slice(self.len(), self.dimension, data_slice))
    }

    /// Create a nalgebra DVector from a single vector in this array.
    ///
    /// # Arguments
    ///
    /// - `index`: The index of the vector to convert
    ///
    /// # Returns
    ///
    /// A DVector with length `dimension`, or an error if index is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vectors = DenseVectorArrayF32::from_vecs(&[
    ///     vec![1.0, 2.0, 3.0],
    /// ], 3)?;
    ///
    /// let vec = vectors.to_nalgebra_vector(0)?;
    /// assert_eq!(vec.len(), 3);
    /// ```
    pub fn to_nalgebra_vector(&self, index: usize) -> Result<DVector<T::Native>> {
        if index >= self.len() {
            return Err(NarrowError::ComputationError(format!(
                "index {} out of bounds for array of length {}",
                index,
                self.len()
            )));
        }

        let values = self.values();

        // Handle Arrow array offsets for sliced arrays
        let offset = values.offset();
        let buffer_slice = values.values();
        let start = offset + (index * self.dimension);
        let end = start + self.dimension;

        Ok(DVector::from_row_slice(&buffer_slice[start..end]))
    }

    /// Get a view of a single vector as a nalgebra DVectorView.
    ///
    /// This provides a zero-copy view into the Arrow array.
    ///
    /// # Arguments
    ///
    /// - `index`: The index of the vector to view
    ///
    /// # Returns
    ///
    /// A DVectorView with length `dimension`, or an error if index is out of bounds.
    pub fn nalgebra_view(&self, index: usize) -> Result<DVectorView<T::Native>> {
        if index >= self.len() {
            return Err(NarrowError::ComputationError(format!(
                "index {} out of bounds for array of length {}",
                index,
                self.len()
            )));
        }

        let values = self.values();

        // Handle Arrow array offsets for sliced arrays
        let offset = values.offset();
        let buffer_slice = values.values();
        let start = offset + (index * self.dimension);
        let end = start + self.dimension;

        Ok(DVectorView::from_slice(&buffer_slice[start..end], self.dimension))
    }

    /// Create a DenseVectorArray from a nalgebra matrix.
    ///
    /// Each row of the matrix becomes a vector in the array.
    ///
    /// # Arguments
    ///
    /// - `matrix`: A nalgebra DMatrix with shape `(num_vectors, dimension)`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nalgebra::DMatrix;
    ///
    /// let matrix = DMatrix::from_row_slice(2, 3, &[
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// ]);
    ///
    /// let vectors = DenseVectorArrayF32::from_nalgebra_matrix(&matrix)?;
    /// assert_eq!(vectors.len(), 2);
    /// assert_eq!(vectors.dimension(), 3);
    /// ```
    pub fn from_nalgebra_matrix(matrix: &DMatrix<T::Native>) -> Result<Self> {
        let num_vectors = matrix.nrows();
        let dimension = matrix.ncols();

        if dimension == 0 {
            return Err(NarrowError::InvalidDimension(0));
        }

        let vectors: Vec<Vec<T::Native>> = (0..num_vectors)
            .map(|i| matrix.row(i).iter().copied().collect())
            .collect();

        Self::from_vecs(&vectors, dimension)
    }

    /// Create a DenseVectorArray from a single nalgebra vector.
    ///
    /// The resulting array will contain a single vector.
    ///
    /// # Arguments
    ///
    /// - `vector`: A nalgebra DVector
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nalgebra::DVector;
    ///
    /// let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let vectors = DenseVectorArrayF32::from_nalgebra_vector(&vec)?;
    /// assert_eq!(vectors.len(), 1);
    /// assert_eq!(vectors.dimension(), 3);
    /// ```
    pub fn from_nalgebra_vector(vector: &DVector<T::Native>) -> Result<Self> {
        let dimension = vector.len();

        if dimension == 0 {
            return Err(NarrowError::InvalidDimension(0));
        }

        let vec_data: Vec<T::Native> = vector.iter().copied().collect();
        Self::from_vecs(&[vec_data], dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseVectorArrayF32;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_to_nalgebra_matrix() {
        let vectors = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ], 3).unwrap();

        let matrix = vectors.to_nalgebra_matrix().unwrap();

        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 3);
        assert_relative_eq!(matrix[(0, 0)], 1.0);
        assert_relative_eq!(matrix[(1, 2)], 6.0);
    }

    #[test]
    fn test_to_nalgebra_vector() {
        let vectors = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ], 3).unwrap();

        let vec0 = vectors.to_nalgebra_vector(0).unwrap();
        assert_eq!(vec0.len(), 3);
        assert_relative_eq!(vec0[0], 1.0);

        let vec1 = vectors.to_nalgebra_vector(1).unwrap();
        assert_relative_eq!(vec1[0], 4.0);
    }

    #[test]
    fn test_nalgebra_view() {
        let vectors = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
        ], 3).unwrap();

        let view = vectors.nalgebra_view(0).unwrap();
        assert_eq!(view.len(), 3);
        assert_relative_eq!(view[0], 1.0);
        assert_relative_eq!(view[2], 3.0);
    }

    #[test]
    fn test_from_nalgebra_matrix() {
        let matrix = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);

        let vectors = DenseVectorArrayF32::from_nalgebra_matrix(&matrix).unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors.dimension(), 3);
        assert_eq!(vectors.get(0).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_nalgebra_vector() {
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let vectors = DenseVectorArrayF32::from_nalgebra_vector(&vec).unwrap();

        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors.dimension(), 3);
        assert_eq!(vectors.get(0).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matrix_roundtrip() {
        let original = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ], 2).unwrap();

        let matrix = original.to_nalgebra_matrix().unwrap();
        let roundtrip = DenseVectorArrayF32::from_nalgebra_matrix(&matrix).unwrap();

        assert_eq!(original.len(), roundtrip.len());
        assert_eq!(original.dimension(), roundtrip.dimension());

        for i in 0..original.len() {
            assert_eq!(original.get(i).unwrap(), roundtrip.get(i).unwrap());
        }
    }

    #[test]
    fn test_sliced_array_to_nalgebra_matrix() {
        use arrow::array::Array;

        let original = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ], 3).unwrap();

        // Slice to get the last two elements
        let sliced = original.as_arrow().slice(1, 2);
        let sliced_narrow = DenseVectorArrayF32::try_new(sliced).unwrap();

        let matrix = sliced_narrow.to_nalgebra_matrix().unwrap();

        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 3);
        assert_relative_eq!(matrix[(0, 0)], 4.0);
        assert_relative_eq!(matrix[(1, 2)], 9.0);
    }

    #[test]
    fn test_sliced_array_to_nalgebra_vector() {
        use arrow::array::Array;

        let original = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ], 2).unwrap();

        // Slice to get the middle element
        let sliced = original.as_arrow().slice(1, 1);
        let sliced_narrow = DenseVectorArrayF32::try_new(sliced).unwrap();

        let vec = sliced_narrow.to_nalgebra_vector(0).unwrap();

        assert_eq!(vec.len(), 2);
        assert_relative_eq!(vec[0], 3.0);
        assert_relative_eq!(vec[1], 4.0);
    }

    #[test]
    fn test_sliced_array_nalgebra_view() {
        use arrow::array::Array;

        let original = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ], 3).unwrap();

        // Slice to get the first element
        let sliced = original.as_arrow().slice(0, 1);
        let sliced_narrow = DenseVectorArrayF32::try_new(sliced).unwrap();

        let view = sliced_narrow.nalgebra_view(0).unwrap();

        assert_eq!(view.len(), 3);
        assert_relative_eq!(view[0], 1.0);
        assert_relative_eq!(view[1], 2.0);
        assert_relative_eq!(view[2], 3.0);
    }
}
