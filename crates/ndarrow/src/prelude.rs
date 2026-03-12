//! ndarrow prelude.
//!
//! Import this module to bring the most commonly used ndarrow traits and
//! conversion functions into scope.

pub use crate::{
    AsNdarray, IntoArrow, NdarrowElement, NdarrowError,
    complex::{
        Complex32Extension, Complex32VariableShapeTensorIter, Complex64Extension,
        Complex64VariableShapeTensorIter, array1_complex32_to_extension,
        array1_complex64_to_extension, arrays_complex32_to_variable_shape_tensor,
        arrays_complex64_to_variable_shape_tensor, complex32_as_array_view1,
        complex32_variable_shape_tensor_iter, complex64_as_array_view1,
        complex64_variable_shape_tensor_iter,
    },
    extensions::{
        RegisteredExtension, deserialize_registered_extension, registered_extension_names,
    },
    fixed_size_list_as_array2, fixed_size_list_as_array2_masked,
    fixed_size_list_as_array2_unchecked,
    helpers::{
        cast_f32_to_f64, cast_f64_to_f32, compact_non_null, densify_csr_view, fill_nulls_with_mean,
        fill_nulls_with_zero, reshape_primitive_to_array2, reshape_primitive_to_arrayd,
        to_standard_layout,
    },
    sparse::{
        CsrMatrixBatchExtension, CsrMatrixBatchIter, CsrMatrixExtension, CsrMatrixMetadata,
        CsrView, csr_batch_to_extension_array, csr_matrix_batch_iter, csr_to_extension_array,
        csr_view_from_columns, csr_view_from_extension,
    },
    tensor::{
        VariableShapeTensorIter, arrayd_to_fixed_shape_tensor, arrays_to_variable_shape_tensor,
        fixed_shape_tensor_as_array_viewd, variable_shape_tensor_iter,
    },
};
