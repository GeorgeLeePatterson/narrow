//! Integration tests verifying the zero-copy guarantee.
//!
//! These tests confirm that bridge conversions never allocate or copy data.
//! They compare raw pointer addresses to prove views point directly at Arrow
//! buffers, and ownership transfers preserve the original allocation.

use arrow_array::{
    Float32Array, Float64Array, ListArray, PrimitiveArray,
    types::{Float32Type, Float64Type},
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ndarrow::{
    AsNdarray, IntoArrow, arrayd_to_fixed_shape_tensor, arrays_to_variable_shape_tensor,
    csr_to_extension_array, csr_view_from_extension, fixed_shape_tensor_as_array_viewd,
    variable_shape_tensor_iter,
};

// ─── Inbound: views point to Arrow's buffer ───

#[test]
fn primitive_view_borrows_arrow_buffer_f64() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let arr = Float64Array::from(data);
    let arrow_ptr = arr.values().as_ref().as_ptr();

    let view = arr.as_ndarray().unwrap();
    let view_ptr = view.as_ptr();

    assert_eq!(arrow_ptr, view_ptr, "ArrayView1 must point directly to Arrow's buffer");
}

#[test]
fn primitive_view_borrows_arrow_buffer_f32() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let arr = Float32Array::from(data);
    let arrow_ptr = arr.values().as_ref().as_ptr();

    let view = arr.as_ndarray().unwrap();
    let view_ptr = view.as_ptr();

    assert_eq!(arrow_ptr, view_ptr, "ArrayView1 must point directly to Arrow's buffer");
}

#[test]
fn fsl_view_borrows_inner_buffer() {
    use std::sync::Arc;

    use arrow_schema::Field;

    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let values = Float64Array::from(data);
    let inner_ptr = values.values().as_ref().as_ptr();

    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
    let fsl = arrow_array::FixedSizeListArray::new(field, 3, Arc::new(values), None);

    let view = ndarrow::fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();
    let view_ptr = view.as_ptr();

    assert_eq!(
        inner_ptr, view_ptr,
        "ArrayView2 must point directly to the inner PrimitiveArray's buffer"
    );
}

#[test]
fn unchecked_view_borrows_same_buffer() {
    let arr = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let arrow_ptr = arr.values().as_ref().as_ptr();

    let view = unsafe { arr.as_ndarray_unchecked() };
    let view_ptr = view.as_ptr();

    assert_eq!(arrow_ptr, view_ptr, "unchecked view must also point to Arrow's buffer");
}

#[test]
fn masked_view_borrows_same_buffer() {
    let arr = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let arrow_ptr = arr.values().as_ref().as_ptr();

    let (view, _mask) = arr.as_ndarray_masked();
    let view_ptr = view.as_ptr();

    assert_eq!(arrow_ptr, view_ptr, "masked view must point to Arrow's buffer");
}

// ─── Outbound: ownership transfer preserves the allocation ───

#[test]
fn array1_transfer_preserves_allocation_f64() {
    let data = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
    let original_ptr = data.as_ptr();
    let arr = Array1::from_vec(data);

    let prim: PrimitiveArray<Float64Type> = arr.into_arrow().unwrap();
    let arrow_ptr = prim.values().as_ref().as_ptr();

    assert_eq!(original_ptr, arrow_ptr, "into_arrow must transfer the Vec without copying");
}

#[test]
fn array1_transfer_preserves_allocation_f32() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let original_ptr = data.as_ptr();
    let arr = Array1::from_vec(data);

    let prim: PrimitiveArray<Float32Type> = arr.into_arrow().unwrap();
    let arrow_ptr = prim.values().as_ref().as_ptr();

    assert_eq!(original_ptr, arrow_ptr, "into_arrow must transfer the Vec without copying");
}

#[test]
fn array2_transfer_preserves_allocation() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let original_ptr = data.as_ptr();
    let arr = Array2::from_shape_vec((2, 3), data).unwrap();

    let fsl = arr.into_arrow().unwrap();
    let inner = fsl.values().as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
    let arrow_ptr = inner.values().as_ref().as_ptr();

    assert_eq!(
        original_ptr, arrow_ptr,
        "into_arrow for Array2 must transfer the Vec without copying"
    );
}

// ─── Full pipeline: only the computation allocates ───

#[test]
fn pipeline_only_computation_allocates() {
    // Create Arrow data
    let input_data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let arrow_input = Float64Array::from(input_data);
    let input_ptr = arrow_input.values().as_ref().as_ptr();

    // Step 1: Arrow → ndarray view (zero-copy)
    let view = arrow_input.as_ndarray().unwrap();
    assert_eq!(view.as_ptr(), input_ptr, "inbound view must share input buffer");

    // Step 2: Computation (this IS the allocation — it's expected)
    let result = &view * 2.0;

    // Step 3: ndarray → Arrow (zero-copy ownership transfer)
    let result_ptr = result.as_ptr();
    let arrow_output: Float64Array = result.into_arrow().unwrap();
    let output_ptr = arrow_output.values().as_ref().as_ptr();

    assert_eq!(result_ptr, output_ptr, "outbound transfer must preserve the computation buffer");

    // The only allocation in the pipeline is step 2 (the computation).
    // Steps 1 and 3 are O(1) pointer operations.
}

// ─── Sparse/tensor zero-copy checks ───

#[test]
fn csr_extension_view_borrows_arrow_buffers() {
    let row_ptrs = vec![0_i32, 2, 3];
    let col_indices = vec![0_u32, 2, 1];
    let values = vec![1.0_f64, 5.0, 2.0];
    let (field, array) = csr_to_extension_array("csr", 4, row_ptrs, col_indices, values).unwrap();

    let indices = array.column(0).as_any().downcast_ref::<ListArray>().unwrap();
    let indices_values =
        indices.values().as_any().downcast_ref::<arrow_array::UInt32Array>().unwrap();
    let values_col = array.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    let values_values = values_col.values().as_any().downcast_ref::<Float64Array>().unwrap();

    let view = csr_view_from_extension::<Float64Type>(&field, &array).unwrap();
    assert_eq!(view.col_indices.as_ptr(), indices_values.values().as_ref().as_ptr());
    assert_eq!(view.values.as_ptr(), values_values.values().as_ref().as_ptr());
}

#[test]
fn fixed_shape_tensor_view_borrows_arrow_buffer() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3]), data).unwrap();
    let (field, fsl) = arrayd_to_fixed_shape_tensor("tensor", tensor).unwrap();

    let inner = fsl.values().as_any().downcast_ref::<Float32Array>().unwrap();
    let arrow_ptr = inner.values().as_ref().as_ptr();
    let view = fixed_shape_tensor_as_array_viewd::<Float32Type>(&field, &fsl).unwrap();
    assert_eq!(view.as_ptr(), arrow_ptr);
}

#[test]
fn variable_shape_tensor_iter_borrows_arrow_buffer() {
    let a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![5.0_f32, 6.0]).unwrap();
    let (field, array) = arrays_to_variable_shape_tensor("ragged", vec![a, b], None).unwrap();

    let data_col =
        array.column_by_name("data").unwrap().as_any().downcast_ref::<ListArray>().unwrap();
    let data_values = data_col.values().as_any().downcast_ref::<Float32Array>().unwrap();
    let arrow_ptr = data_values.values().as_ref().as_ptr();

    let mut iter = variable_shape_tensor_iter::<Float32Type>(&field, &array).unwrap();
    let (_row, view) = iter.next().unwrap().unwrap();
    assert_eq!(view.as_ptr(), arrow_ptr);
}
