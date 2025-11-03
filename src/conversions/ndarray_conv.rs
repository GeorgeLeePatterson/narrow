//! Zero-copy conversions to ndarray.
//!
//! This module provides conversions from DenseVectorArray to ndarray's ArrayView2,
//! enabling efficient batch operations without copying data.

use crate::dense::DenseVectorArray;
use crate::error::{NarrowError, Result};
use crate::scalar::VectorScalar;
use ndarray::{ArrayView1, ArrayView2};

impl<T: VectorScalar> DenseVectorArray<T> {
    /// Get a zero-copy view of this vector array as a 2D ndarray.
    ///
    /// The resulting array has shape `(num_vectors, dimension)` where each row
    /// is a vector.
    ///
    /// # Safety
    ///
    /// This method provides a zero-copy view into the Arrow array's buffer. The view
    /// is valid as long as the DenseVectorArray exists. The returned ArrayView borrows
    /// from this array.
    ///
    /// # Returns
    ///
    /// An ArrayView2 with shape `(len, dimension)`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vectors = DenseVectorArrayF32::from_vecs(&[
    ///     vec![1.0, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    /// ], 3)?;
    ///
    /// let matrix = vectors.as_ndarray()?;
    /// assert_eq!(matrix.shape(), &[2, 3]);
    /// assert_eq!(matrix[[0, 0]], 1.0);
    /// assert_eq!(matrix[[1, 2]], 6.0);
    /// ```
    pub fn as_ndarray(&self) -> Result<ArrayView2<T::Native>> {
        let values = self.values();

        // Handle Arrow array offsets for sliced arrays
        // The values array may have an offset if the FixedSizeListArray was sliced
        let offset = values.offset();
        let total_elements = self.len() * self.dimension;
        let buffer_slice = values.values();

        // Slice the buffer to account for offset
        let data_slice = &buffer_slice[offset..offset + total_elements];

        // Create a 2D view with shape (num_vectors, dimension)
        ArrayView2::from_shape((self.len(), self.dimension), data_slice)
            .map_err(|e| NarrowError::ComputationError(format!("failed to create ndarray view: {}", e)))
    }

    /// Get a zero-copy view of a single vector as a 1D ndarray.
    ///
    /// # Arguments
    ///
    /// - `index`: The index of the vector to view
    ///
    /// # Returns
    ///
    /// An ArrayView1 with length `dimension`, or None if index is out of bounds.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vectors = DenseVectorArrayF32::from_vecs(&[
    ///     vec![1.0, 2.0, 3.0],
    /// ], 3)?;
    ///
    /// let vec = vectors.ndarray_view(0)?;
    /// assert_eq!(vec.len(), 3);
    /// ```
    pub fn ndarray_view(&self, index: usize) -> Result<ArrayView1<T::Native>> {
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

        ArrayView1::from_shape(self.dimension, &buffer_slice[start..end])
            .map_err(|e| NarrowError::ComputationError(format!("failed to create ndarray view: {}", e)))
    }

    /// Create a DenseVectorArray from an ndarray matrix.
    ///
    /// Each row of the matrix becomes a vector in the array.
    ///
    /// # Arguments
    ///
    /// - `array`: A 2D ndarray with shape `(num_vectors, dimension)`
    ///
    /// # Returns
    ///
    /// A new DenseVectorArray containing copies of the data.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ndarray::array;
    ///
    /// let matrix = array![[1.0, 2.0], [3.0, 4.0]];
    /// let vectors = DenseVectorArrayF32::from_ndarray(&matrix)?;
    /// assert_eq!(vectors.len(), 2);
    /// assert_eq!(vectors.dimension(), 2);
    /// ```
    pub fn from_ndarray(array: &ndarray::ArrayView2<T::Native>) -> Result<Self> {
        let (num_vectors, dimension) = array.dim();

        if dimension == 0 {
            return Err(NarrowError::InvalidDimension(0));
        }

        // Convert to row-major if needed and collect into vectors
        let vectors: Vec<Vec<T::Native>> = (0..num_vectors)
            .map(|i| array.row(i).to_vec())
            .collect();

        Self::from_vecs(&vectors, dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseVectorArrayF32;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_as_ndarray() {
        let vectors = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ], 3).unwrap();

        let matrix = vectors.as_ndarray().unwrap();

        assert_eq!(matrix.shape(), &[3, 3]);
        assert_relative_eq!(matrix[[0, 0]], 1.0);
        assert_relative_eq!(matrix[[0, 1]], 2.0);
        assert_relative_eq!(matrix[[0, 2]], 3.0);
        assert_relative_eq!(matrix[[2, 2]], 9.0);
    }

    #[test]
    fn test_ndarray_view() {
        let vectors = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ], 3).unwrap();

        let vec0 = vectors.ndarray_view(0).unwrap();
        assert_eq!(vec0.len(), 3);
        assert_relative_eq!(vec0[0], 1.0);
        assert_relative_eq!(vec0[2], 3.0);

        let vec1 = vectors.ndarray_view(1).unwrap();
        assert_relative_eq!(vec1[0], 4.0);
    }

    #[test]
    fn test_from_ndarray() {
        let matrix = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];

        let vectors = DenseVectorArrayF32::from_ndarray(&matrix.view()).unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors.dimension(), 3);
        assert_eq!(vectors.get(0).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(vectors.get(1).unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_roundtrip() {
        let original = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ], 2).unwrap();

        let matrix = original.as_ndarray().unwrap();
        let roundtrip = DenseVectorArrayF32::from_ndarray(&matrix).unwrap();

        assert_eq!(original.len(), roundtrip.len());
        assert_eq!(original.dimension(), roundtrip.dimension());

        for i in 0..original.len() {
            assert_eq!(original.get(i).unwrap(), roundtrip.get(i).unwrap());
        }
    }

    #[test]
    fn test_sliced_array_as_ndarray() {
        use arrow::array::Array;

        let original = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ], 3).unwrap();

        // Slice to get the middle element
        let sliced = original.as_arrow().slice(1, 1);
        let sliced_narrow = DenseVectorArrayF32::try_new(sliced).unwrap();

        let matrix = sliced_narrow.as_ndarray().unwrap();

        assert_eq!(matrix.shape(), &[1, 3]);
        assert_relative_eq!(matrix[[0, 0]], 4.0);
        assert_relative_eq!(matrix[[0, 1]], 5.0);
        assert_relative_eq!(matrix[[0, 2]], 6.0);
    }

    #[test]
    fn test_sliced_array_ndarray_view() {
        use arrow::array::Array;

        let original = DenseVectorArrayF32::from_vecs(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ], 2).unwrap();

        // Slice to get the last element
        let sliced = original.as_arrow().slice(2, 1);
        let sliced_narrow = DenseVectorArrayF32::try_new(sliced).unwrap();

        let vec = sliced_narrow.ndarray_view(0).unwrap();

        assert_eq!(vec.len(), 2);
        assert_relative_eq!(vec[0], 5.0);
        assert_relative_eq!(vec[1], 6.0);
    }
}
