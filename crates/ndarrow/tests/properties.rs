//! Property-style integration tests for round-trip and invariant coverage.

use arrow_array::{Array, Float32Array, Float64Array, types::Float32Type};
use ndarray::{Array2, ArrayD, IxDyn};
use ndarrow::{
    AsNdarray, IntoArrow, arrayd_to_fixed_shape_tensor, csr_to_extension_array,
    csr_view_from_extension, fixed_shape_tensor_as_array_viewd, fixed_size_list_as_array2,
    helpers::densify_csr_view,
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    #[test]
    fn prop_dense_scalar_roundtrip_f64(values in proptest::collection::vec(-1_000_000.0_f64..1_000_000.0, 0..128)) {
        let original = Float64Array::from(values);
        let view = original.as_ndarray().expect("generated scalar array must be valid");
        let roundtrip: Float64Array = view.to_owned().into_arrow().expect("roundtrip must succeed");

        prop_assert_eq!(roundtrip.len(), original.len());
        for i in 0..original.len() {
            prop_assert_eq!(roundtrip.value(i).to_bits(), original.value(i).to_bits());
        }
    }

    #[test]
    fn prop_dense_matrix_roundtrip_f32(
        rows in 0_usize..16,
        cols in 1_usize..16,
        values in proptest::collection::vec(-10_000.0_f32..10_000.0, 0..(16 * 16))
    ) {
        let required = rows * cols;
        prop_assume!(required <= values.len());

        let matrix_values = values.into_iter().take(required).collect::<Vec<_>>();
        let matrix = Array2::from_shape_vec((rows, cols), matrix_values.clone())
            .expect("shape must match test data");
        let arrow = matrix.into_arrow().expect("matrix to Arrow must succeed");
        let view = fixed_size_list_as_array2::<Float32Type>(&arrow)
            .expect("Arrow to matrix view must succeed");

        prop_assert_eq!(view.shape(), &[rows, cols]);
        for (actual, expected) in view.iter().zip(matrix_values.iter()) {
            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn prop_sparse_csr_roundtrip_and_densify_f32(
        nrows in 1_usize..8,
        ncols in 1_usize..8,
        cells in proptest::collection::vec((any::<bool>(), -1000_i16..1000_i16), 1..(8 * 8))
    ) {
        let cell_count = nrows * ncols;
        prop_assume!(cell_count <= cells.len());

        let mut row_ptrs = Vec::with_capacity(nrows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut expected_dense = vec![0.0_f32; cell_count];

        row_ptrs.push(0_i32);
        for row in 0..nrows {
            for col in 0..ncols {
                let (present, raw) = cells[row * ncols + col];
                if present {
                    let value = f32::from(raw) / 10.0;
                    col_indices.push(u32::try_from(col).expect("column index must fit u32"));
                    values.push(value);
                    expected_dense[row * ncols + col] = value;
                }
            }
            row_ptrs.push(i32::try_from(values.len()).expect("nnz length must fit i32"));
        }

        let (field, array) = csr_to_extension_array("csr", ncols, row_ptrs, col_indices, values)
            .expect("csr construction must succeed");
        let view = csr_view_from_extension::<Float32Type>(&field, &array)
            .expect("csr view extraction must succeed");
        let dense = densify_csr_view(&view).expect("densify from csr view must succeed");

        let dense_values = dense
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("dense values must be f32");
        let actual_dense = dense_values.values().iter().copied().collect::<Vec<_>>();

        prop_assert_eq!(actual_dense.len(), expected_dense.len());
        for (actual, expected) in actual_dense.iter().zip(expected_dense.iter()) {
            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn prop_fixed_tensor_roundtrip_f32(
        batch in 1_usize..4,
        dim1 in 1_usize..4,
        dim2 in 1_usize..4,
        values in proptest::collection::vec(-10_000.0_f32..10_000.0, 0..(4 * 4 * 4))
    ) {
        let required = batch * dim1 * dim2;
        prop_assume!(required <= values.len());

        let tensor_values = values.into_iter().take(required).collect::<Vec<_>>();
        let tensor = ArrayD::from_shape_vec(IxDyn(&[batch, dim1, dim2]), tensor_values.clone())
            .expect("shape must match generated tensor values");

        let (field, fsl) = arrayd_to_fixed_shape_tensor("tensor", tensor)
            .expect("ArrayD to fixed-shape tensor must succeed");
        let view = fixed_shape_tensor_as_array_viewd::<Float32Type>(&field, &fsl)
            .expect("fixed-shape tensor view extraction must succeed");

        prop_assert_eq!(view.shape(), &[batch, dim1, dim2]);
        for (actual, expected) in view.iter().zip(tensor_values.iter()) {
            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}
